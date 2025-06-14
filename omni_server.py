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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCDataChannel
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

# ─── Shared session state ───────────────────────────────────────────────
MAX_VIDEO_HISTORY = 30  # tokens of history for each client
last_narration_per_client: Dict[str, str] = {}
previous_frame_per_client: Dict[str, Optional[Image.Image]] = {}

# Store peer connections and other client-specific data
pcs: Dict[str, RTCPeerConnection] = {}

# ─── Helper functions ───────────────────────────────────────────────────

def _decode_image_from_base64(image_b64: str) -> Optional[Image.Image]:
    try:
        data = base64.b64decode(image_b64)
        img_arr = cv2.imdecode(cv2.UMat(data).get(), cv2.IMREAD_COLOR)
        if img_arr is None:
            return None
        img_arr = cv2.resize(img_arr, (320, 320), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    except Exception as exc:
        print(f"Image decode error: {exc}")
        return None


def _compute_frame_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """Compute similarity between two PIL Images using histograms."""
    try:
        # Convert to numpy arrays for faster processing
        arr1 = np.array(img1.resize((64, 64)))  # Resize for speed
        arr2 = np.array(img2.resize((64, 64)))
        
        # Compute histogram correlation
        hist1 = np.histogram(arr1.flatten(), bins=50)[0]
        hist2 = np.histogram(arr2.flatten(), bins=50)[0]
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Compute correlation coefficient
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    except Exception as e:
        print(f"Error computing frame similarity: {e}")
        return 0.0


def _process_frame_sync(sid: str, curr_img: Image.Image, prev_img: Optional[Image.Image]):
    """Blocking call that interacts with MiniCPM-o and streams tokens out."""
    last_narr = last_narration_per_client.get(sid, "")
    if last_narr:
        prompt_text = (
            "Continue narrating while ignoring static background details. "
            f"Previously you said: '{last_narr}'. What is happening now?"
        )
    else:
        prompt_text = (
            "Focus on the main subjects and their actions. Ignore static "
            "background details. What's happening now?"
        )

    # Use official MiniCPM-o-2.6 API format
    msgs = [{'role': 'user', 'content': [curr_img, prompt_text]}]
    
    try:
        # Use official chat method with streaming (if available) or full response
        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True
        )
        
        # Clean up response tokens
        if response:
            cleaned_response = (
                response.replace("<|audio_sep|>", "")
                .replace("<|tts_eos|>", "")
                .replace("<|tts|>", "")
                .strip()
            )
            
            if cleaned_response:
                last_narration_per_client[sid] = cleaned_response
                
                # Get the data channel for narration output  
                narration_channel = data_channels.get('narration')
                
                # Send the complete response as narration tokens
                # For real-time feel, split into words and send progressively
                words = cleaned_response.split()
                for i, word in enumerate(words):
                    message = {
                        "type": "narration",
                        "token": word + (" " if i < len(words) - 1 else ""),
                        "is_final": i == len(words) - 1
                    }
                    
                    # Send through data channel if available
                    if narration_channel:
                        try:
                            narration_channel.send(json.dumps(message))
                            time.sleep(0.05)  # Small delay for streaming effect
                        except Exception as e:
                            print(f"Error sending via data channel: {e}")
                            break
                
                # Send final completion message
                if narration_channel:
                    try:
                        completion_message = {
                            "type": "narration_complete",
                            "full_text": cleaned_response
                        }
                        narration_channel.send(json.dumps(completion_message))
                    except Exception as e:
                        print(f"Error sending completion via data channel: {e}")
        
    except Exception as e:
        print(f"Error in MiniCPM-o inference: {e}")
        # Send error message via data channel
        narration_channel = data_channels.get('narration')
        if narration_channel:
            try:
                error_message = {
                    "type": "error",
                    "message": f"Inference error: {str(e)}"
                }
                narration_channel.send(json.dumps(error_message))
            except Exception as send_e:
                print(f"Error sending error message: {send_e}")


async def process_frame(sid: str, curr_img: Image.Image, prev_img: Optional[Image.Image]):
    return await asyncio.to_thread(_process_frame_sync, sid, curr_img, prev_img)

# ─── WebRTC signalling & media ingestion ───────────────────────────────
app = FastAPI()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        raise HTTPException(status_code=400, detail="Invalid SDP payload")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())
    pcs[peer_id] = pc
    
    # Dictionary to store data channels per peer
    data_channels = {}
    
    # Handle data channel creation
    @pc.on("datachannel")
    def on_datachannel(channel):
        channel_id = channel.label
        print(f"🔌 New data channel {channel_id} for peer {peer_id}")
        data_channels[channel_id] = channel
        
        @channel.on("message")
        def on_message(message):
            # Handle different message types
            try:
                if isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "start_narration":
                        client_id = data.get("client_id", peer_id)
                        print(f"🎙️ Starting narration for client {client_id} via data channel")
                        # Save client identifier for this peer connection
                        channel.client_id = client_id
                        # Send acknowledgment
                        channel.send(json.dumps({"type": "ack", "message": "Narration started"}))
                    
                    elif msg_type == "end_stream":
                        print(f"🛑 End stream request received for peer {peer_id}")
                        channel.send(json.dumps({"type": "ack", "message": "Stream ended"}))
                        # Note: Connection cleanup happens in the track handler when it exits
            except Exception as e:
                print(f"❌ Error handling data channel message: {e}")
                channel.send(json.dumps({"type": "error", "message": str(e)}))

    recorder = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print(f"📺 Video track for peer {peer_id}")
            prev_img = None
            prev_frame_data = None
            frame_counter = 0
            duplicate_frames = 0
            start_time = time.time()
            
            while True:
                frame = await track.recv()
                bgr = frame.to_ndarray(format="bgr24")
                bgr = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
                # Enhanced frame debugging with sample points and difference detection
                frame_counter += 1
                ts = time.time()
                
                # Take sample points from frame to create a reliable signature
                # We sample at multiple points to catch small movements
                h, w, _ = bgr.shape
                sample_points = [
                    bgr[h//4, w//4].mean(),      # top-left quadrant
                    bgr[h//4, 3*w//4].mean(),   # top-right quadrant
                    bgr[h//2, w//2].mean(),     # center
                    bgr[3*h//4, w//4].mean(),   # bottom-left quadrant
                    bgr[3*h//4, 3*w//4].mean(), # bottom-right quadrant
                ]
                
                # Create hash of sample points
                frame_hash = hashlib.md5(str(sample_points).encode()).hexdigest()[:8]
                
                # Calculate frame difference if we have a previous frame
                is_duplicate = False
                if prev_frame_data is not None:
                    # Calculate Mean Squared Error between frames
                    if frame_counter % 30 == 0:
                        frame_hash = hashlib.md5(rgb.tobytes()).hexdigest()
                        if prev_frame_data == frame_hash:
                            duplicate_frames += 1
                        prev_frame_data = frame_hash
                        
                        # Report stats every 30 frames
                        elapsed = ts - start_time
                        fps = frame_counter / elapsed if elapsed > 0 else 0
                        duplicate_pct = (duplicate_frames / frame_counter * 100) if frame_counter > 0 else 0
                        print(f"WebRTC stats for {peer_id}: {frame_counter} frames in {elapsed:.1f}s "  
                              f"({fps:.1f} FPS), {duplicate_pct:.1f}% duplicates")
                
                # Skip duplicate frames
                if prev_img is not None:
                    # Use point samples for quick similarity check
                    if prev_frame_data == sample_points:
                        continue
                
                # Get the data channel for narration output
                narration_channel = data_channels.get('narration')
                
                # Process this frame (generates narration) and send via data channel
                rgb_pil = Image.fromarray(rgb)
                
                # Process frame with the narration channel
                last_narration = last_narration_per_client.get(peer_id, "")
                if prev_img is None or _compute_frame_similarity(prev_img, rgb_pil) < 0.95:  # Skip very similar frames
                    # Proper async call to run MiniCPM in thread
                    narration = await process_frame(peer_id, rgb_pil, prev_img)
                    
                    # Only send if the narration has changed
                    if narration and narration != last_narration:
                        # Send chunks for better real-time experience
                        if len(narration) > 80:
                            words = narration.split()
                            chunks = []
                            current = []
                            total = 0
                            
                            # Break into ~50 char chunks on word boundaries 
                            for word in words:
                                if total + len(word) + 1 > 50 and current:
                                    chunks.append(" ".join(current))
                                    current = [word]
                                    total = len(word)
                                else:
                                    current.append(word)
                                    total += len(word) + 1
                            if current:
                                chunks.append(" ".join(current))
                            
                            # Send chunks with is_end flag on the last one
                            for i, chunk in enumerate(chunks):
                                is_last = i == len(chunks) - 1
                                message = {
                                    "type": "narration",
                                    "token": chunk,
                                    "is_end": is_last
                                }
                                
                                # Send through data channel if available
                                if narration_channel:
                                    try:
                                        narration_channel.send(json.dumps(message))
                                    except Exception as e:
                                        print(f"Error sending via data channel: {e}")
                                        # No WebSocket fallback here - this is pure WebRTC
                                
                                await asyncio.sleep(0.1)  # Slight delay between chunks 
                        else:
                            message = {
                                "type": "narration",
                                "token": narration,
                                "is_end": True
                            }
                            
                            # Send through data channel if available
                            if narration_channel:
                                try:
                                    narration_channel.send(json.dumps(message))
                                except Exception as e:
                                    print(f"Error sending via data channel: {e}")
                        
                        last_narration_per_client[peer_id] = narration
                
                prev_img = rgb_pil
                prev_frame_data = sample_points
                
        elif track.kind == "audio":
            await recorder.start()
            track.add_sink(recorder)

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
