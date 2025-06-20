"""Pure WebRTC ASGI server for LumiMate live narration.

This server uses FastAPI for WebRTC signalling (SDP offer/answer)
and aiortc for handling WebRTC peer connections and data channels.
Narration tokens are streamed over a WebRTC data channel.

Run with:
    uvicorn lib.services.omni_server:app --host 0.0.0.0 --port 8123 --reload

Endpoints:
    POST /offer       – WebRTC signalling (SDP offer → answer)

Data Channels:
    'narration'       – For sending narration tokens from server to client.
    'control'         – For control messages (e.g., start/stop narration) from client to server.

Media (UDP) ports are negotiated via ICE; ensure relevant UDP ports are open if behind NAT/firewall.

Dependencies (pip install):
    fastapi uvicorn[standard] aiortc opencv-python Pillow

Notes:
- MiniCPM-o model is loaded once and shared.
- Heavy model calls run in a background thread via asyncio.to_thread.
- Video is received via WebRTC, processed, and narration is sent back.
"""
from __future__ import annotations
import asyncio
import logging
import uuid
import time
import json
import re
from typing import Dict, Optional, List, Any
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCDataChannel, MediaStreamError
from aiortc.contrib.media import MediaBlackhole

import torch
from transformers import AutoModel, AutoTokenizer

# ─── Model initialisation ────────────────────────────────────────────────
logging.getLogger("torch").setLevel(logging.ERROR)
print("🔄 Loading MiniCPM-o-2_6 model…", flush=True)

# Use the simple GitHub implementation
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM-o-2_6', trust_remote_code=True)

print("✅ Model loaded", flush=True)

# ─── Global state management ───────────────────────────────────────────────

# Global stores for peer connections and data channels
pcs: Dict[str, RTCPeerConnection] = {}
global_data_channels: Dict[str, Dict[str, RTCDataChannel]] = {}
global_peer_data: Dict[str, Dict] = {}  # For storing peer-specific data
last_narration_per_client: Dict[str, str] = {}
no_change_count: Dict[str, int] = {}  # Track consecutive "no change" responses

# Authentication store
auth_tokens = set()

# ─── Shared session state ───────────────────────────────────────────────
MAX_VIDEO_HISTORY = 30  # tokens of history for each client

# Real-time frame processing system
frame_buffers: Dict[str, List[Image.Image]] = {}
narration_loops: Dict[str, asyncio.Task] = {}
MAX_FRAMES_IN_BATCH = 12  # More frames for better temporal context
NARRATION_INTERVAL = 1.5  # Slightly longer interval to accumulate more frames


# For FPS calculation
last_fps_log_time: Dict[str, float] = {}
frame_counter: Dict[str, int] = {}

# Frame difference threshold
TEXT_SIMILARITY_THRESHOLD = 0.75  # Reduced from 0.85 to allow more variation


# ─── Helper functions ───────────────────────────────────────────────────


def are_texts_similar(a: str, b: str) -> bool:
    """Compare two strings for similarity."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() > TEXT_SIMILARITY_THRESHOLD


def get_peer_id_from_channel(channel: RTCDataChannel) -> str | None:
    """Extract peer ID from data channel by searching global state."""
    for peer_id, channels in global_data_channels.items():
        if channel.label in channels and channels[channel.label] == channel:
            return peer_id
    print("⚠️ [WARNING] Could not find peer_id for channel.")
    return None


async def narration_loop(peer_id: str):
    """Periodically process the latest frames from the buffer."""
    while peer_id in pcs:
        await asyncio.sleep(NARRATION_INTERVAL)

        if peer_id in frame_buffers and frame_buffers[peer_id]:
            # Copy the latest frames and clear the buffer immediately
            # to avoid processing stale frames on the next iteration.
            frames_to_process = frame_buffers[peer_id].copy()
            frame_buffers[peer_id].clear()

            print(
                f"🔄 [NARRATION_LOOP] Processing batch of {len(frames_to_process)} frames for peer {peer_id}")
            # Process the batch without blocking the loop for too long
            asyncio.create_task(process_frame(peer_id, frames_to_process))
        else:
            # This is not an error, just means no new frames arrived.
            pass


def cleanup_peer_data(peer_id: str):
    """Clean up all data for a peer."""
    # Cancel the narration loop
    if peer_id in narration_loops:
        narration_loops[peer_id].cancel()
        narration_loops.pop(peer_id, None)

    # Clear frame buffer
    frame_buffers.pop(peer_id, None)

    # Clear other peer data
    last_narration_per_client.pop(peer_id, None)
    no_change_count.pop(peer_id, None)
    global_data_channels.pop(peer_id, None)
    last_fps_log_time.pop(peer_id, None)
    frame_counter.pop(peer_id, None)

    print(f"🧹 [DEBUG] Cleaned up all data for peer {peer_id}")


async def _process_frame_sync(sid: str, frames: List[Image.Image]):
    """Process a batch of frames with MiniCPM-o and send response.

    Args:
        sid: Session ID for the peer
        frames: List of PIL Images to process as a batch (should be in chronological order)
    """
    if not frames:
        print("⚠️ [DEBUG] No frames provided to _process_frame_sync")
        return None

    print(
        f"🔄 [DEBUG] Starting batch inference for peer {sid} with {len(frames)} frames")
    print(f"🔄 [DEBUG] Frame dimensions: {frames[0].size if frames else 'N/A'}")

    # Get last narration for context
    last_narr = last_narration_per_client.get(sid, "")

    # Build context-aware prompt for multi-frame processing
    if last_narr:
        question = (
            f"Continue narrating for a visually impaired person. Last, you said: '{last_narr}'. "
            "Look carefully at these frames in sequence and describe any new movement, change in position, "
            "new objects appearing, or different actions in one very short sentence. "
            "If nothing has genuinely changed, you may say 'No change detected.' "
            "Examples: 'The person moves to the left.' or 'A cup appears on the table.'"
        )
        print(
            f"🔄 [DEBUG] Using continuation prompt with last narration: '{last_narr[:50]}...'")
    else:
        question = (
            "You are a scene narrator for a visually impaired person. Look at these video frames and describe "
            "the most prominent object or action in a one or two very short sentence. "
            "Focus on what's most important or interesting in the scene. "
            "Examples: 'You are looking at a desk with a laptop.' or 'A person is holding a blue phone.'"
        )
        print("🔄 [DEBUG] Using initial prompt (no previous narration)")

    # Wait for narration channel to be ready (with timeout)
    max_wait = 5.0  # seconds
    start_time = time.time()
    narration_channel = None

    while time.time() - start_time < max_wait:
        narration_channel = global_data_channels.get(sid, {}).get("narration")
        if narration_channel and narration_channel.readyState == "open":
            break
        await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    if not narration_channel:
        print(
            f"⚠️ [DEBUG] No narration channel found for peer {sid} after {max_wait} seconds")
        return None

    if narration_channel.readyState != "open":
        print(
            f"⚠️ [DEBUG] Narration channel not open for peer {sid} after {max_wait} seconds")
        return None

    print(f"✅ [DEBUG] Got open narration channel for peer {sid}")

    try:
        # Prepare messages for the model (using GitHub implementation format)
        msgs = [
            {'role': 'user', 'content': frames + [question]},
        ]

        # Set decode params for video
        params = {}
        params["use_image_id"] = False
        # use 1 if cuda OOM and video resolution > 448*448
        params["max_slice_nums"] = 2

        print("🔄 [DEBUG] Running model.chat()...")

        # Use the simple chat method from GitHub implementation
        answer = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )

        print(f"🔄 [DEBUG] Model response: '{answer}'")

        if answer and answer.strip():
            clean_answer = answer.strip()

            # Take only the first sentence to keep it brief
            sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
            if sentences:
                clean_answer = sentences[0]

            # Handle "No change detected" responses
            if "no change detected" in clean_answer.lower():
                # Increment no-change counter
                no_change_count[sid] = no_change_count.get(sid, 0) + 1

                # If we've had too many "no change" responses, reset context to get fresh perspective
                if no_change_count[sid] >= 3:
                    print(
                        f"🔄 [DEBUG] Resetting context after {no_change_count[sid]} no-change responses")
                    last_narration_per_client[sid] = ""
                    no_change_count[sid] = 0
                    return None  # Skip this response but reset context

                # Allow some "no change" responses through
                print(
                    f"✅ [DEBUG] Allowing no-change response ({no_change_count[sid]}/3)")
            else:
                # Reset no-change counter on successful response
                no_change_count[sid] = 0

            # Filter out unwanted phrases (but allow "No change detected")
            static_phrases = ["static", "unchanged",
                              "similar to before", "remains the same"]
            if any(phrase in clean_answer.lower() for phrase in static_phrases):
                print(
                    f"🗑️ [DEBUG] Discarding static response for peer {sid}: '{clean_answer}'")
                return None

            # Filter out responses that are too similar to the last one
            if are_texts_similar(clean_answer, last_narr):
                print(
                    f"🗑️ [DEBUG] Discarding similar response for peer {sid}: '{clean_answer}'")
                return None

            # Send complete response directly (no word-by-word streaming)
            complete_message = {
                'type': 'complete',
                'full_text': clean_answer,
                'is_final': True
            }

            if narration_channel.readyState == "open":
                try:
                    narration_channel.send(json.dumps(complete_message))
                    print(
                        f"✅ [DEBUG] Sent complete response: '{clean_answer}'")
                except Exception as e:
                    print(
                        f"⚠️ [ERROR] Error sending complete response: {str(e)}")

            # Update last narration for context
            last_narration_per_client[sid] = clean_answer
            print(f"✅ [DEBUG] Updated last narration for peer {sid}")

            return clean_answer
        else:
            print("⚠️ [DEBUG] No response generated")
            return None

    except Exception as e:
        print(f"⚠️ [ERROR] Error during model inference: {str(e)}")
        try:
            narration_channel = global_data_channels.get(
                sid, {}).get("narration")
            if narration_channel and narration_channel.readyState == "open":
                error_message = {
                    "type": "error",
                    "message": f"Error processing video: {str(e)}",
                    "is_final": True
                }
                narration_channel.send(json.dumps(error_message))
        except Exception as send_error:
            print(f"⚠️ [ERROR] Failed to send error message: {send_error}")

        return None


async def process_frame(sid: str, frames: List[Image.Image]):
    """Process a batch of frames and handle streaming.

    Args:
        sid: Session ID for the peer
        frames: List of PIL Images to process as a batch
    """
    print(
        f"🎬 [DEBUG] process_frame called for peer {sid} with {len(frames)} frames")
    print(f"🎬 [DEBUG] Starting batch streaming inference...")

    try:
        # Directly await the async _process_frame_sync
        result = await _process_frame_sync(sid, frames)
        print(
            f"🎬 [DEBUG] Batch streaming inference completed, result: {'Success' if result else 'Skipped/None'}")
        return result
    except Exception as e:
        print(f"🎬 [ERROR] Error in batch processing: {str(e)}")
        return None

# ─── WebRTC signalling & media ingestion ───────────────────────────────
app = FastAPI()

# Data channel message handler


async def handle_start_narration(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle start narration request."""
    try:
        peer_id = get_peer_id_from_channel(channel)
        if not peer_id:
            return

        client_id = data.get('client_id', 'unknown')

        print(
            f"🎬 [DEBUG] Starting narration for peer {peer_id}, client {client_id}")

        # Send acknowledgment
        response = {
            "type": "ack",
            "message": "Server ready for video processing"
        }
        channel.send(json.dumps(response))

    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_start_narration: {e}")


async def handle_end_stream(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle end stream request."""
    try:
        peer_id = get_peer_id_from_channel(channel)
        if not peer_id:
            return

        print(f"🛑 [DEBUG] Ending stream for peer {peer_id}")

        # Clean up any peer-specific data
        cleanup_peer_data(peer_id)

        # Send acknowledgment
        response = {
            "type": "ack",
            "message": "Stream ended successfully"
        }
        channel.send(json.dumps(response))

    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_end_stream: {e}")


async def on_message(channel: RTCDataChannel, message: str):
    """Handle incoming WebRTC data channel messages."""
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        print(f"📨 [DEBUG] Received message: {msg_type}")

        if msg_type == 'start_narration':
            await handle_start_narration(channel, data)
        elif msg_type == 'end_stream':
            await handle_end_stream(channel, data)
        else:
            print(f"⚠️ [DEBUG] Unknown message type: {msg_type}")

    except Exception as e:
        print(f"❌ [DEBUG] Error in on_message: {e}")


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        raise HTTPException(status_code=400, detail="Invalid SDP payload")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())
    pc.id = peer_id  # Assign the peer_id to the connection object
    pcs[peer_id] = pc

    # Initialize data channels dictionary for this peer
    global_data_channels[peer_id] = {}

    @pc.on("datachannel")
    def on_datachannel(channel):
        global_data_channels[peer_id][channel.label] = channel
        print(f"📦 Data channel '{channel.label}' created for peer {peer_id}")

        @channel.on("message")
        async def on_message_wrapper(message):
            await on_message(channel, message)

    @pc.on("track")
    async def on_track(track):
        """Handle incoming WebRTC video frames and process with MiniCPM."""
        if track.kind != "video":
            return

        peer_id = pc.id
        print(f"📺 Video track received for peer {peer_id}")

        # Initialize FPS counter and frame buffer
        last_fps_log_time[peer_id] = time.time()
        frame_counter[peer_id] = 0
        frame_buffers[peer_id] = []
        track_frame_counter = 0  # For skipping frames

        # Start the narration loop for this peer
        if peer_id not in narration_loops:
            loop = asyncio.get_event_loop()
            narration_loops[peer_id] = loop.create_task(
                narration_loop(peer_id))
            print(f"✅ [DEBUG] Started narration loop for peer {peer_id}")

        try:
            while True:
                try:
                    # Get next video frame
                    frame = await track.recv()
                    frame_counter[peer_id] += 1
                    track_frame_counter += 1

                    # Skip frames to reduce processing load (process every 2nd frame)
                    if track_frame_counter % 2 != 0:
                        continue

                    # Log FPS periodically
                    current_time = time.time()
                    if current_time - last_fps_log_time.get(peer_id, 0) >= 5.0:
                        fps = frame_counter.get(peer_id, 0) / 5.0
                        print(
                            f"📊 [INFO] Peer {peer_id} incoming FPS: {fps:.2f}")
                        frame_counter[peer_id] = 0
                        last_fps_log_time[peer_id] = current_time

                    # Convert frame to RGB PIL Image (no resizing - handled by client)
                    try:
                        bgr = frame.to_ndarray(format="bgr24")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)

                        # Validate frame dimensions - skip if too small or inconsistent
                        if pil_img.size[0] < 200 or pil_img.size[1] < 200:
                            print(
                                f"⚠️ [DEBUG] Skipping frame with invalid dimensions: {pil_img.size}")
                            continue

                    except Exception as frame_convert_error:
                        print(
                            f"⚠️ [DEBUG] Failed to convert frame: {frame_convert_error}")
                        continue

                    # Add frame to buffer, keeping it at a max size
                    if peer_id in frame_buffers:
                        frame_buffers[peer_id].append(pil_img)
                        if len(frame_buffers[peer_id]) > MAX_FRAMES_IN_BATCH:
                            # Remove oldest frame
                            frame_buffers[peer_id].pop(0)

                except MediaStreamError as e:
                    print(f"🔌 Video track ended for peer {peer_id}: {e}")
                    break
                except Exception as frame_error:
                    print(
                        f"⚠️ [ERROR] Error processing video frame for peer {peer_id}: {frame_error}")
                    continue

        except Exception as e:
            print(
                f"❌ [ERROR] Error in video track processing for peer {peer_id}: {e}")
        finally:
            # The narration loop will handle remaining frames.
            # We just need to signal that this track is done.
            print(f"📺 Video processing ended for peer {peer_id}")
            # The on_state_change handler will do the final cleanup.

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"ℹ️ Connection state is {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            print(f"🔌 Peer {pc.id} disconnected.")
            if pc.id in pcs:
                await pcs[pc.id].close()
                pcs.pop(pc.id, None)
                cleanup_peer_data(pc.id)

    # Handle offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
