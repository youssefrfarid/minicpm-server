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
MAX_FRAMES_IN_BATCH = 10  # Number of frames to batch for narration before processing
NARRATION_INTERVAL = 1.0  # Interval to check for a new batch of frames
TEXT_SIMILARITY_THRESHOLD = 0.85  # Threshold for detecting similar responses to avoid repetition

# ─── Global state dictionaries ───────────────────────────────────────────
# Peer connection and WebRTC state
pcs_data: Dict[str, Dict[str, Any]] = {}  # WebRTC peer connections
peer_data: Dict[str, Dict[str, Any]] = {}  # Peer-specific data
global_data_channels: Dict[str, RTCDataChannel] = {}  # Data channels
model_inference_lock = asyncio.Lock()  # Global lock for model inference

# Frame and narration state tracking
last_narration_per_client: Dict[str, str] = {}  # Previous narration text

# FPS calculation
frame_counter: Dict[str, int] = {}  # Count frames per peer
last_fps_log_time: Dict[str, float] = {}  # Last FPS calculation timestamp

# Authentication
auth_tokens = set()  # Valid authentication tokens

# Executor for running blocking tasks
executor = None

# ─── Helper functions ───────────────────────────────────────────────────


def are_texts_similar(a: str, b: str) -> bool:
    """Compare two strings for similarity."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() > TEXT_SIMILARITY_THRESHOLD


def get_peer_id_from_channel(channel: RTCDataChannel) -> Optional[str]:
    """Extract peer ID from data channel by searching global state."""
    for peer_id, data in global_data_channels.items():
        if data.get("control") == channel or data.get("narration") == channel:
            return peer_id
    return None


def cleanup_peer_data(peer_id: str):
    """Clean up all data for a peer."""
    print(f"🧹 Cleaning up data for peer {peer_id}")
    if peer_id in peer_data:
        # Cancel any running narration task
        narration_task = peer_data[peer_id].get("narration_task")
        if narration_task and not narration_task.done():
            narration_task.cancel()
        
        # Remove all peer data
        peer_data.pop(peer_id, None)
    
    # Clean up other global state
    global_data_channels.pop(peer_id, None)
    frame_counter.pop(peer_id, None)
    last_fps_log_time.pop(peer_id, None)
    last_narration_per_client.pop(peer_id, None)
    
    print(f"🧹 [DEBUG] Cleaned up all data for peer {peer_id}")


async def narration_loop(peer_id: str):
    """Periodically processes frames from the buffer for narration."""
    while peer_id in pcs_data and not peer_data[peer_id]["narration_task"].cancelled():
        if model_inference_lock.locked():
            logger.debug(f"🔄 [NARRATION_LOOP] Skipping batch for peer {peer_id}, as model is busy.")
            await asyncio.sleep(NARRATION_INTERVAL)
            continue

        buffer = peer_data[peer_id]["frame_buffer"]
        
        frames_to_process = []
        if len(buffer) >= MAX_FRAMES_IN_BATCH:
            frames_to_process = list(buffer)
            peer_data[peer_id]["frame_buffer"].clear()

        if frames_to_process:
            logger.info(f"🔄 [NARRATION_LOOP] Processing a batch of {len(frames_to_process)} frames for peer {peer_id}")
            asyncio.create_task(process_frames_batch(peer_id, frames_to_process))
        
        await asyncio.sleep(NARRATION_INTERVAL)


def _process_frames_batch_sync(peer_id: str, frames: List[Image.Image]):
    """Process a batch of frames with MiniCPM-o chat API. Runs in a separate thread."""
    if not frames:
        print("⚠️ [DEBUG] No frames provided to _process_frames_batch_sync")
        return None

    last_narration = last_narration_per_client.get(peer_id, "")
    prompt_text = (
        f"You are an assistant for a visually impaired user. "
        "Your task is to summarize the scene from a real-time video stream in a single, brief sentence. "
        "- Focus on the most important object or person. "
        "- Do NOT list multiple items or describe them in detail."
        f"Describe only what has changed since the previous description: '{last_narration}'. "
        "One concise sentence."
    )

    # The user message is a combination of the prompt and the frames
    msgs = [{'role': 'user', 'content': [prompt_text] + frames}]

    try:
        # Call the model's chat function
        answer = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            temperature=0.1,
        )

        if not answer:
            print("🗑️ [DEBUG] Empty response from model")
            return None

        # Clean up the response
        clean_answer = answer.replace('<|tts_eos|>', '').strip()
        if not clean_answer:
            return None

        # Compare with the last narration to filter static/repetitive responses
        if are_texts_similar(clean_answer, last_narration):
            print(f"🗑️ [DEBUG] Skipping similar narration for peer {peer_id}.")
            return None

        # Send the narration to the client
        narration_channel = global_data_channels.get(peer_id, {}).get("narration")
        if narration_channel and narration_channel.readyState == "open":
            complete_message = {
                "type": "narration",
                "text": clean_answer
            }
            narration_channel.send(json.dumps(complete_message))
            display_text = (clean_answer[:60] + '..') if len(clean_answer) > 60 else clean_answer
            print(
                f"📢 [DEBUG] Sent narration to peer {peer_id}: '{display_text}'")

            # Update global last narration
            last_narration_per_client[peer_id] = clean_answer
            return clean_answer

    except Exception as e:
        print(f"⚠️ [ERROR] Frame processing error for peer {peer_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def process_frames_batch(peer_id: str, frames: List[Image.Image]):
    """Processes a batch of frames to generate narration."""
    if model_inference_lock.locked():
        print(f"🚦 [WARN] Concurrency issue: process_frames_batch called while lock was already held. Aborting.")
        return

    async with model_inference_lock:
        print(f"🎥 [DEBUG] process_frames_batch called for peer {peer_id} with {len(frames)} frames")
        loop = asyncio.get_running_loop()
        result = None
        try:
            result = await loop.run_in_executor(
                None, _process_frames_batch_sync, peer_id, frames
            )
        except Exception as e:
            print(f"⚠️ [ERROR] Frame processing error for peer {peer_id}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"🎥 [DEBUG] Batch inference completed, result: {'Success' if result else 'Skipped/None'}")
        return result


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

        logger.info(
            f"🎬 [DEBUG] Starting narration for peer {peer_id}, client {client_id}")
        
        # Initialize streaming session if needed
        if not model_initialized:
            init_success = await initialize_streaming_session()
            if not init_success:
                logger.error("❌ [ERROR] Failed to initialize streaming session")
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
    pcs_data[pc.id] = {"pc": pc}  # Now we can safely use pc.id

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

        # Initialize peer-specific data
        peer_data[peer_id] = {
            "frame_buffer": [],
            "narration_task": None,
            "last_narration": "",
            "static_response_counter": 0
        }
        frame_counter[peer_id] = 0
        last_fps_log_time[peer_id] = time.time()
        last_narration_per_client[peer_id] = ""
        track_frame_counter = 0
        # Start the narration loop for this peer
        if peer_id not in peer_data or peer_data[peer_id]["narration_task"] is None or peer_data[peer_id]["narration_task"].done():
            loop = asyncio.get_event_loop()
            peer_data[peer_id]["narration_task"] = asyncio.create_task(
                narration_loop(peer_id))
            print(f"✅ [DEBUG] Started narration loop for peer {peer_id}")

        try:
            while True:
                # Gracefully exit if peer connection was closed
                if peer_id not in pcs_data:
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
                    if peer_id in peer_data:
                        peer_data[peer_id]["frame_buffer"].append(pil_img)
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
            if pc.id in pcs_data:
                await pcs_data[pc.id]["pc"].close()
                pcs_data.pop(pc.id, None)
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
