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
import base64
import logging
import os
import uuid
import time
import json
import hashlib
import numpy as np
from collections import deque
from io import BytesIO
from typing import Dict, Optional, List, Any

import cv2
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
previous_frame_per_client: Dict[str, Optional[Image.Image]] = {}

# Authentication store
auth_tokens = set()

# ─── Shared session state ───────────────────────────────────────────────
MAX_VIDEO_HISTORY = 30  # tokens of history for each client

# Frame batching system
frame_buffers: Dict[str, List[Image.Image]] = {}
batch_timers: Dict[str, asyncio.Task] = {}
BATCH_SIZE = 3  # Smaller batch size for faster response
BATCH_TIMEOUT = 2.0  # Process batch every 2 seconds max

# ─── Helper functions ───────────────────────────────────────────────────


def get_peer_id_from_channel(channel: RTCDataChannel) -> str:
    """Extract peer ID from data channel."""
    return f"peer_{id(channel)}"


async def process_batch_timeout(peer_id: str):
    """Process batch after timeout expires."""
    await asyncio.sleep(BATCH_TIMEOUT)

    if peer_id in frame_buffers and frame_buffers[peer_id]:
        print(
            f"⏰ [DEBUG] Timeout triggered batch processing for peer {peer_id}")
        frames = frame_buffers[peer_id].copy()
        frame_buffers[peer_id] = []

        # Clear the timer
        batch_timers.pop(peer_id, None)

        # Process the batch
        asyncio.create_task(process_frame(peer_id, frames))


async def add_frame_to_batch(peer_id: str, frame: Image.Image):
    """Add frame to batch and handle batching logic."""
    # Initialize buffer if needed
    if peer_id not in frame_buffers:
        frame_buffers[peer_id] = []

    # Add frame to buffer
    frame_buffers[peer_id].append(frame)
    print(
        f"🎞️ [DEBUG] Added frame to batch for peer {peer_id} (buffer size: {len(frame_buffers[peer_id])})")

    # Cancel existing timer if present
    if peer_id in batch_timers:
        batch_timers[peer_id].cancel()
        batch_timers.pop(peer_id, None)

    # Check if we should process the batch
    if len(frame_buffers[peer_id]) >= BATCH_SIZE:
        print(
            f"🎞️ [DEBUG] Batch size reached for peer {peer_id}, processing immediately")
        frames = frame_buffers[peer_id].copy()
        frame_buffers[peer_id] = []

        # Process the batch
        asyncio.create_task(process_frame(peer_id, frames))
    else:
        # Start timeout timer
        batch_timers[peer_id] = asyncio.create_task(
            process_batch_timeout(peer_id))


def cleanup_peer_data(peer_id: str):
    """Clean up all data for a peer."""
    # Cancel any pending batch timer
    if peer_id in batch_timers:
        batch_timers[peer_id].cancel()
        batch_timers.pop(peer_id, None)

    # Clear frame buffer
    frame_buffers.pop(peer_id, None)

    # Clear other peer data
    previous_frame_per_client.pop(peer_id, None)
    last_narration_per_client.pop(peer_id, None)
    global_data_channels.pop(peer_id, None)

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
            f"Continue narrating this video sequence. Previously you mentioned: '{last_narr}'. "
            "Describe any new developments or changes in the scene. Keep it brief (1-2 sentences)."
        )
        print(
            f"🔄 [DEBUG] Using continuation prompt with last narration: '{last_narr[:50]}...'")
    else:
        question = (
            "Describe what's happening in this video sequence. "
            "Focus on movement, actions, and important changes. "
            "Keep it brief and descriptive (1-2 sentences)."
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
            # Send complete response directly (no word-by-word streaming)
            complete_message = {
                'type': 'complete',
                'full_text': answer.strip(),
                'is_final': True
            }

            if narration_channel.readyState == "open":
                try:
                    narration_channel.send(json.dumps(complete_message))
                    print(
                        f"✅ [DEBUG] Sent complete response: '{answer.strip()}'")
                except Exception as e:
                    print(
                        f"⚠️ [ERROR] Error sending complete response: {str(e)}")

            # Update last narration for context
            last_narration_per_client[sid] = answer.strip()
            print(f"✅ [DEBUG] Updated last narration for peer {sid}")

            return answer.strip()
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


async def on_message(channel: RTCDataChannel, message: str):
    """Handle incoming WebRTC data channel messages."""
    try:
        data = json.loads(message)
        msg_type = data.get('type')

        print(f"📨 [DEBUG] Received message: {msg_type}")

        if msg_type == 'start_narration':
            # Handle narration start request
            await handle_start_narration(channel, data)
        elif msg_type == 'end_stream':
            # Handle stream end request
            await handle_end_stream(channel, data)
        elif msg_type == 'question':
            # Handle question request
            await handle_question(channel, data)
        else:
            print(f"⚠️ [DEBUG] Unknown message type: {msg_type}")

    except Exception as e:
        print(f"❌ [DEBUG] Error in on_message: {e}")


async def handle_start_narration(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle start narration request."""
    try:
        peer_id = get_peer_id_from_channel(channel)
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

        print(f"🛑 [DEBUG] Ending stream for peer {peer_id}")

        # Clean up any peer-specific data
        if peer_id in previous_frame_per_client:
            del previous_frame_per_client[peer_id]
        if peer_id in last_narration_per_client:
            del last_narration_per_client[peer_id]

        # Send acknowledgment
        response = {
            "type": "ack",
            "message": "Stream ended successfully"
        }
        channel.send(json.dumps(response))

    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_end_stream: {e}")


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

    # Handle any additional data channels created by the client
    @pc.on("datachannel")
    def on_datachannel(channel):
        channel_id = channel.label
        print(
            f"🔌 Client-created data channel '{channel_id}' for peer {peer_id}")
        global_data_channels[peer_id][channel_id] = channel

        # Special handling for narration channel
        if channel_id == "narration":
            print(f"🔌 Narration channel received for peer {peer_id}")

            @channel.on("open")
            def on_narration_open():
                print(f"🔌 Narration channel opened for peer {peer_id}")

            @channel.on("message")
            def on_narration_message(message):
                asyncio.create_task(on_message(channel, message))
        else:
            @channel.on("message")
            def on_channel_message(message):
                asyncio.create_task(on_message(channel, message))

    recorder = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        """Handle incoming WebRTC video frames and process with MiniCPM."""
        if track.kind != "video":
            return

        print(f"📺 Video track received for peer {pc.id}")
        frame_count = 0

        try:
            while True:
                try:
                    # Get next video frame
                    frame = await track.recv()
                    frame_count += 1

                    # Skip frames to reduce processing load (process every 5th frame)
                    if frame_count % 5 != 0:
                        continue

                    # Convert frame to RGB PIL Image
                    bgr = frame.to_ndarray(format="bgr24")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    # Standard size for processing
                    rgb = cv2.resize(rgb, (384, 384))
                    pil_img = Image.fromarray(rgb)

                    # Add frame to batch processing system
                    await add_frame_to_batch(pc.id, pil_img)

                except MediaStreamError as e:
                    print(f"🔌 Video track ended for peer {pc.id}: {e}")
                    break
                except Exception as frame_error:
                    print(
                        f"⚠️ [ERROR] Error processing video frame for peer {pc.id}: {frame_error}")
                    continue

        except Exception as e:
            print(
                f"❌ [ERROR] Error in video track processing for peer {pc.id}: {e}")
        finally:
            # Process any remaining frames in buffer
            if pc.id in frame_buffers and frame_buffers[pc.id]:
                print(
                    f"🔄 [DEBUG] Processing final batch of {len(frame_buffers[pc.id])} frames for peer {pc.id}")
                final_frames = frame_buffers[pc.id].copy()
                frame_buffers[pc.id] = []
                asyncio.create_task(process_frame(pc.id, final_frames))

            print(f"📺 Video processing ended for peer {pc.id}")
            # Note: Don't clean up peer data here as it might still be needed

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.pop(peer_id, None)
            # Clean up all peer-specific data
            cleanup_peer_data(peer_id)
            print(f"🛑 Peer {peer_id} closed and cleaned up")

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
