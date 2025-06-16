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
from typing import Dict, Optional

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
print("🔄 Loading MiniCPM-o-2_6 omni model…", flush=True)
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-o-2_6",
    trust_remote_code=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=False,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-o-2_6", trust_remote_code=True)
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

# ─── Helper functions ───────────────────────────────────────────────────

def get_peer_id_from_channel(channel: RTCDataChannel) -> str:
    """Extract peer ID from data channel."""
    return f"peer_{id(channel)}"


async def _process_frame_sync(sid: str, frames: List[Image.Image]):
    """Blocking call that processes a batch of frames with MiniCPM-o and streams tokens.
    
    Args:
        sid: Session ID for the peer
        frames: List of PIL Images to process as a batch (should be in chronological order)
    """
    if not frames:
        print("⚠️ [DEBUG] No frames provided to _process_frame_sync")
        return None
        
    print(f"🔄 [DEBUG] Starting batch inference for peer {sid} with {len(frames)} frames")
    print(f"🔄 [DEBUG] Frame dimensions: {frames[0].size if frames else 'N/A'}")
    
    # Get last narration for context
    last_narr = last_narration_per_client.get(sid, "")
    
    # Build context-aware prompt for multi-frame processing
    if last_narr:
        prompt_text = (
            "Continue narrating the video sequence while focusing on changes and movement. "
            f"Previously you mentioned: '{last_narr}'. "
            "Describe any new developments or changes in the scene."
        )
        print(f"🔄 [DEBUG] Using continuation prompt with last narration: '{last_narr[:50]}...'")
    else:
        prompt_text = (
            "You are seeing a sequence of video frames. "
            "Provide a concise narration of what's happening in the video. "
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
        print(f"⚠️ [DEBUG] No narration channel found for peer {sid} after {max_wait} seconds")
        return None
    
    if narration_channel.readyState != "open":
        print(f"⚠️ [DEBUG] Narration channel not open for peer {sid} after {max_wait} seconds")
        return None
        
    print(f"✅ [DEBUG] Got open narration channel for peer {sid}")
    
    full_response = ""
    
    try:
        # 1. Reset session for this peer
        print(f"🔄 [DEBUG] Resetting session for peer {sid}")
        model.reset_session(session_id=sid)
        
        # 2. Prefill system prompt
        print("🔄 [DEBUG] Prefilling system prompt...")
        sys_msg = model.get_sys_prompt(mode='omni', language='en')
        model.streaming_prefill(
            session_id=sid,
            msgs=[sys_msg],
            tokenizer=tokenizer
        )
        
        # 3. Prefill user message with frames
        print(f"🔄 [DEBUG] Prefilling user message with {len(frames)} frames...")
        user_msg = {
            'role': 'user',
            'content': [*frames, prompt_text]
        }
        model.streaming_prefill(
            session_id=sid,
            msgs=[user_msg],
            tokenizer=tokenizer
        )
        
        # 4. Stream generate tokens
        print("🔄 [DEBUG] Starting token generation...")
        response_stream = model.streaming_generate(
            session_id=sid,
            tokenizer=tokenizer,
            temperature=0.2,
            max_new_tokens=100,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            device=model.device,
            omni_input=True
        )
        
        # Process streaming response
        print("🔄 [DEBUG] Streaming tokens to client...")
        for token_data in response_stream:
            if isinstance(token_data, dict):
                # Handle text tokens
                token = token_data.get('text', '')
                if not token:
                    continue
                    
                full_response += token
                
                # Send token to client
                message = {
                    'type': 'token',
                    'token': token,
                    'is_final': False
                }
                
                if narration_channel.readyState == "open":
                    try:
                        narration_channel.send(json.dumps(message))
                        print(f"📤 [DEBUG] Sent token: {token}", end='', flush=True)
                    except Exception as e:
                        print(f"⚠️ [ERROR] Error sending token: {str(e)}")
                
                full_response += token
            
            # Send completion message if we have a response
            if full_response.strip():
                completion_message = {
                    'type': 'complete',
                    'full_text': full_response.strip(),
                    'is_final': True
                }
                if narration_channel.readyState == "open":
                    narration_channel.send(json.dumps(completion_message))
                
                print(f"✅ [DEBUG] Streaming complete: '{full_response.strip()}'")
                
                # Update last narration for context
                last_narration_per_client[sid] = full_response.strip()
                print(f"✅ [DEBUG] Updated last narration for peer {sid}")
                
                return full_response.strip()
            
            return None
            
    except Exception as e:
        print(f"⚠️ [ERROR] Error during streaming generation: {str(e)}")
        try:
            model.reset_session(session_id=sid)
        except Exception as cleanup_error:
            print(f"⚠️ [ERROR] Error cleaning up session: {str(cleanup_error)}")
        try:
            narration_channel = global_data_channels.get(f"{sid}_narration")
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
    print(f"🎬 [DEBUG] process_frame called for peer {sid} with {len(frames)} frames")
    print(f"🎬 [DEBUG] Starting batch streaming inference...")
    
    try:
        # Directly await the async _process_frame_sync
        result = await _process_frame_sync(sid, frames)
        print(f"🎬 [DEBUG] Batch streaming inference completed, result: {'Success' if result else 'Skipped/None'}")
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
        else:
            print(f"⚠️ [DEBUG] Unknown message type: {msg_type}")
            
    except Exception as e:
        print(f"❌ [DEBUG] Error in on_message: {e}")


async def handle_start_narration(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle start narration request."""
    try:
        peer_id = get_peer_id_from_channel(channel)
        client_id = data.get('client_id', 'unknown')
        
        print(f"🎬 [DEBUG] Starting narration for peer {peer_id}, client {client_id}")
        
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
    

    
    # Handle any additional data channels created by the client
    @pc.on("datachannel")
    def on_datachannel(channel):
        channel_id = channel.label
        # Ensure peer dictionary exists now that the first data channel arrived
        if peer_id not in global_data_channels:
            global_data_channels[peer_id] = {}
        if channel_id == "narration":
            print(f"🔌 Client-created narration channel for peer {peer_id}")
            # Update our reference to use the client-created channel
            global_data_channels[peer_id]["narration"] = channel
        else:
            print(f"🔌 New data channel {channel_id} for peer {peer_id}")
            global_data_channels[peer_id][channel_id] = channel
        
        @channel.on("message")
        def on_message(message):
            asyncio.run(on_message(channel, message))
            
    recorder = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        """Handle incoming WebRTC video frames and process with MiniCPM."""
        if track.kind != "video":
            return
            
        print(f"📺 Video track received for peer {pc.id}")
        frame_buffer = []
        BATCH_SIZE = 10
        
        try:
            while True:
                try:
                    # Get next video frame
                    frame = await track.recv()
                    
                    # Convert frame to RGB PIL Image
                    bgr = frame.to_ndarray(format="bgr24")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (384, 384))  # Increased size for better detail
                    pil_img = Image.fromarray(rgb)
                    
                    # Add to buffer
                    frame_buffer.append(pil_img)
                    
                    # Process batch when we have BATCH_SIZE frames
                    if len(frame_buffer) >= BATCH_SIZE:
                        # Process batch in background thread
                        print(f"🔄 [DEBUG] Collected {len(frame_buffer)} frames, processing batch...")
                        asyncio.create_task(process_frame(pc.id, frame_buffer.copy()))
                        frame_buffer = []  # Clear buffer for next batch
                        print(f"🔄 [DEBUG] Started processing batch for peer {pc.id}")
                    
                except MediaStreamError as e:
                    print(f"🔌 Video track ended for peer {pc.id}")
                    break
                    
        except Exception as e:
            print(f"Error in video track processing: {e}")
        finally:
            print(f"Video processing ended for peer {pc.id}")

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.pop(peer_id, None)
            print(f"🛑 Peer {peer_id} closed")

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
