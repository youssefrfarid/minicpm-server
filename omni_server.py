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
from fuzzywuzzy import fuzz

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

# ─── Global constants and configuration ───────────────────────────────────
GLOBAL_SESSION_ID = "global_session"  # Single session ID for all peers for MVP
model_initialized = False  # Flag to track if the streaming session has been initialized
MAX_STATIC_RESPONSES = 3  # Maximum consecutive static responses before reset
MAX_VIDEO_HISTORY = 30  # Tokens of history for each client
MAX_FRAMES_IN_BATCH = 12  # Size of sliding window for frames
NARRATION_INTERVAL = 2.0  # Interval for streaming_generate calls
TEXT_SIMILARITY_THRESHOLD = 0.65  # Threshold for detecting similar responses

# ─── Global state dictionaries ───────────────────────────────────────────
# Peer connection and WebRTC state
pcs: Dict[str, RTCPeerConnection] = {}  # WebRTC peer connections
global_data_channels: Dict[str, Dict[str, RTCDataChannel]] = {}  # Data channels
global_peer_data: Dict[str, Dict] = {}  # For storing peer-specific data

# Frame and narration state tracking
frame_buffers: Dict[str, List[Image.Image]] = {}  # Maps peer_id -> frames
narration_loops: Dict[str, asyncio.Task] = {}  # Background narration tasks
last_narration_per_client: Dict[str, str] = {}  # Previous narration text

# Response quality tracking
static_response_count: Dict[str, int] = {}  # Track consecutive static responses

# FPS calculation 
frame_counter: Dict[str, int] = {}  # Count frames per peer
last_fps_log_time: Dict[str, float] = {}  # Last FPS calculation timestamp

# Authentication
auth_tokens = set()  # Valid authentication tokens


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
    """Periodically process frames using MiniCPM-o's streaming API."""
    # Initialize streaming session on first run
    if not model_initialized:
        await initialize_streaming_session()
    
    while peer_id in pcs:
        await asyncio.sleep(NARRATION_INTERVAL)

        if peer_id in frame_buffers and frame_buffers[peer_id]:
            # For streaming, we don't completely clear the buffer
            # Instead, we maintain a sliding window of the latest frames
            frames_to_process = frame_buffers[peer_id].copy()
            
            # Keep only the most recent frames for next iteration (sliding window)
            # This maintains context while preventing memory buildup
            if len(frame_buffers[peer_id]) > MAX_FRAMES_IN_BATCH:
                # Remove oldest frames, keep the most recent ones
                frames_to_keep = min(MAX_FRAMES_IN_BATCH // 2, len(frame_buffers[peer_id]))
                frame_buffers[peer_id] = frame_buffers[peer_id][-frames_to_keep:]

            print(
                f"🔄 [NARRATION_LOOP] Processing {len(frames_to_process)} frames for peer {peer_id}")
            
            # Process the frames without blocking the loop
            asyncio.create_task(process_frame(peer_id, frames_to_process))
        else:
            # No new frames arrived
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
    global_data_channels.pop(peer_id, None)
    last_fps_log_time.pop(peer_id, None)
    frame_counter.pop(peer_id, None)

    print(f"🧹 [DEBUG] Cleaned up all data for peer {peer_id}")


async def initialize_streaming_session():
    """Initialize the MiniCPM-o streaming session.
    Should be called once at the beginning of the application.
    """
    global model_initialized
    
    if model_initialized:
        return True
    
    print("🔄 [DEBUG] Initializing MiniCPM-o streaming session...")
    try:
        # Reset the session to start fresh
        model.reset_session()
        
        # Create a system prompt for the assistant role
        sys_prompt = {
            "role": "system", 
            "content": (
                "You are an assistant for a visually impaired user. Your task is to describe the scene from a real-time video stream. Be extremely concise and factual. "
                "- Describe ONLY what you see. Do not infer actions or intentions. "
                "- Use short, direct sentences. "
                "- Start with 'You are looking at...' or 'There is...'. "
                "- Example: 'You are looking at a person.' "
                "- Example: 'There is a cup on the table.' "
                "- Do not heavily describe the scene just mention important objects"
            )
        }
        
        # Prefill the system prompt
        model.streaming_prefill(
            session_id=GLOBAL_SESSION_ID,
            msgs=[sys_prompt],
            tokenizer=tokenizer
        )
        
        model_initialized = True
        print("✅ [DEBUG] MiniCPM-o streaming session initialized")
        return True
    except Exception as e:
        print(f"❌ [ERROR] Failed to initialize streaming session: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def _process_frame_sync(sid: str, frames: List[Image.Image]):
    """Process a batch of frames with MiniCPM-o streaming API.

    Args:
        sid: Session ID for the peer
        frames: List of PIL Images to process as a batch (should be in chronological order)
    """
    global model_initialized
    if not frames:
        print("⚠️ [DEBUG] No frames provided to _process_frame_sync")
        return None

    print(
        f"🔄 [DEBUG] Processing frames for peer {sid} with {len(frames)} frames")
    print(f"🔄 [DEBUG] Frame dimensions: {frames[0].size if frames else 'N/A'}")
    
    # Get last narration for context and filtering
    last_narration = last_narration_per_client.get(sid, "")

    # Wait for narration channel to be ready (with timeout)
    max_wait = 5.0  # seconds
    start_time = time.time()
    narration_channel = None

    while time.time() - start_time < max_wait:
        narration_channel = global_data_channels.get(sid, {}).get("narration")
        if narration_channel and narration_channel.readyState == "open":
            break
        await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    if not narration_channel or narration_channel.readyState != "open":
        print(
            f"⚠️ [DEBUG] Narration channel not open or not found for peer {sid} after {max_wait} seconds")
        return None
    print(f"✅ [DEBUG] Got open narration channel for peer {sid}")

    try:
        # Prepare the multimodal prompt with the new frames
        # The prompt should include the last narration to provide context
        prompt_text = "Continue describing to a visually impaired user focus on any people and their actions and any objects that are moving or changing in the scene."
        msgs = [{"role": "user", "content": prompt_text}]
        
        # Use streaming_prefill to add the new frames to the session context
        model.streaming_prefill(
            session_id=GLOBAL_SESSION_ID,
            msgs=msgs,
            images=frames,
            tokenizer=tokenizer,
            omni_input=True,  # Indicate multimodal input
            max_slice_nums=1, # Recommended for streaming
            use_image_id=True # Recommended for streaming
        )

        # Generate the narration using the streaming API
        print("🔄 [DEBUG] Running model.streaming_generate()...")
        response_iterator = model.streaming_generate(
            session_id=GLOBAL_SESSION_ID,
            tokenizer=tokenizer,
            temperature=0.1,
            generate_audio=False  # Disable audio generation to avoid tts_processor error
        )

        # Process the streaming response
        complete_response = ""
        for new_text in response_iterator:
            if isinstance(new_text, dict):
                text_chunk = new_text.get('text', '')
                complete_response += text_chunk
            else:
                complete_response += new_text

        if not complete_response:
            print("🗑️ [DEBUG] Empty response from model")
            return None

        print(f"🔄 [DEBUG] Model streaming response: '{complete_response}'")

        # Clean up the response
        clean_answer = complete_response.replace('<|tts_eos|>', '').strip()
        if not clean_answer:
            return None

        # Compare with the last narration to filter static/repetitive responses
        last_narration = last_narration_per_client.get(sid, "")
        if last_narration:
            similarity = fuzz.ratio(clean_answer.lower(), last_narration.lower()) / 100.0
            print(f"🔎 [SIMILARITY] Comparing:\n"
                  f"  - New: '{clean_answer}'\n"
                  f"  - Last: '{last_narration}'\n"
                  f"  - Similarity: {similarity:.2f}")

            # If the new narration is too similar, increment static counter
            if similarity > TEXT_SIMILARITY_THRESHOLD:
                static_response_count[sid] = static_response_count.get(sid, 0) + 1
                print(f"⚠️ [DEBUG] Static response detected for peer {sid} (Count: {static_response_count[sid]})")

                # Check if we've had too many static responses in a row
                if static_response_count[sid] >= MAX_STATIC_RESPONSES:
                    print(f"⚠️ [WARNING] Too many static responses for peer {sid}, resetting streaming session")
                    try:
                        model_initialized = False
                        model.reset_session()
                        await initialize_streaming_session()
                        
                        if sid in frame_buffers and len(frame_buffers[sid]) > 0:
                            frames_to_keep = min(5, len(frame_buffers[sid]))
                            frame_buffers[sid] = frame_buffers[sid][-frames_to_keep:]
                            print(f"🔄 [DEBUG] Frame buffer partially cleared for peer {sid}, keeping {frames_to_keep} frames")
                        
                        static_response_count[sid] = 0
                    except Exception as e:
                        print(f"❌ [ERROR] Failed to reset session: {str(e)}")
                    return None # Stop processing this response after reset

                # Also skip sending the message if it's almost identical to prevent spam
                if similarity > TEXT_SIMILARITY_THRESHOLD: 
                    print(f"🗑️ [DEBUG] Skipping identical narration for peer {sid}.")
                    return None
            else:
                # Reset counter if response is different enough
                static_response_count[sid] = 0

        # Send the narration to the client
        if narration_channel and narration_channel.readyState == "open":
            # Truncate for display if too long
            display_text = (clean_answer[:60] + '..') if len(clean_answer) > 60 else clean_answer
            
            # Send the complete message over the data channel
            complete_message = {
                "type": "narration",
                "text": clean_answer
            }
            narration_channel.send(json.dumps(complete_message))
            print(
                f"📢 [DEBUG] Sent narration to peer {sid}: '{display_text}'")

            # Update global last narration
            last_narration_per_client[sid] = clean_answer
            static_response_count[sid] = 0  # Reset static response counter when we get a good one

            # Return the processed text (for tracking/debugging)
            return clean_answer

            return None

    except Exception as e:
        print(f"⚠️ [ERROR] Frame processing error for peer {sid}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def process_frame(sid: str, frames: List[Image.Image]):
    """Process frames using MiniCPM-o's streaming API.

    Args:
        sid: Session ID for the peer
        frames: List of PIL Images to process as a continuous stream
    """
    print(
        f"🎥 [DEBUG] process_frame called for peer {sid} with {len(frames)} frames")
    print(f"🎥 [DEBUG] Starting streaming inference...")

    try:
        # Process frames with streaming API
        result = await _process_frame_sync(sid, frames)
        print(
            f"🎥 [DEBUG] Streaming inference completed, result: {'Success' if result else 'Skipped/None'}")
        return result
    except Exception as e:
        print(f"🎥 [ERROR] Error in streaming process: {str(e)}")
        return None

# ─── WebRTC signalling & media ingestion ───────────────────────────────
app = FastAPI()

# Data channel message handler


async def handle_start_narration(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle start narration request."""
    global model_initialized  # Add global declaration at the top of function
    try:
        peer_id = get_peer_id_from_channel(channel)
        if not peer_id:
            return

        client_id = data.get('client_id', 'unknown')

        print(
            f"🎬 [DEBUG] Starting narration for peer {peer_id}, client {client_id}")
        
        # Initialize streaming session if needed
        if not model_initialized:
            init_success = await initialize_streaming_session()
            if not init_success:
                print("❌ [ERROR] Failed to initialize streaming session")
                channel.send(json.dumps({
                    "type": "error",
                    "message": "Failed to initialize model"
                }))
                return

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
                # Gracefully exit if peer connection was closed
                if peer_id not in pcs:
                    print(f"🛑 [DEBUG] Peer {peer_id} disconnected, stopping video processing.")
                    break

                try:
                    # Get next video frame
                    frame = await track.recv()
                    frame_counter[peer_id] += 1
                    track_frame_counter += 1

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

                    # Add frame to buffer for streaming processing
                    # We'll maintain a sliding window in the narration_loop
                    if peer_id in frame_buffers:
                        frame_buffers[peer_id].append(pil_img)
                        # We allow buffer to grow slightly beyond MAX_FRAMES_IN_BATCH
                        # narration_loop will trim it periodically

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
