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

# ─── Global state management ───────────────────────────────────────────────

# Track peer connections
pcs: Dict[str, RTCPeerConnection] = {}
last_narration_per_client: Dict[str, str] = {}

# Global data channels storage (keyed by peer_id then channel_label)
global_data_channels: Dict[str, Dict[str, RTCDataChannel]] = {}

# ─── Shared session state ───────────────────────────────────────────────
MAX_VIDEO_HISTORY = 30  # tokens of history for each client
previous_frame_per_client: Dict[str, Optional[Image.Image]] = {}

# Frame memory for incoming frames from Flutter client
frame_memory: Dict[str, Image.Image] = {}

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
    """Blocking call that interacts with MiniCPM-o and returns streaming tokens."""
    print(f"🔄 [DEBUG] Starting inference for peer {sid}")
    print(f"🔄 [DEBUG] Current image size: {curr_img.size}")
    print(f"🔄 [DEBUG] Previous image: {'Available' if prev_img else 'None'}")
    
    # Skip processing if frames are too similar (static scene)
    if prev_img is not None:
        similarity = _compute_frame_similarity(curr_img, prev_img)
        print(f"🔄 [DEBUG] Frame similarity: {similarity:.3f}")
        if similarity > 0.95:  # 95% similar = skip processing
            print(f"⏭️ [DEBUG] Frames too similar ({similarity:.3f}), skipping inference")
            return None
    
    last_narr = last_narration_per_client.get(sid, "")
    
    # Build context-aware prompt
    if last_narr:
        prompt_text = (
            "Continue narrating while ignoring static background details. "
            f"Previously you said: '{last_narr}'. What is happening now?"
        )
        print(f"🔄 [DEBUG] Using continuation prompt with last narration: '{last_narr[:50]}...'")
    else:
        prompt_text = (
            "Focus on the main subjects and their actions. Ignore static "
            "background details. What's happening now?"
        )
        print(f"🔄 [DEBUG] Using initial prompt (no previous narration)")

    # Use official MiniCPM-o-2.6 API format with system instruction
    msgs = [
        {
            'role': 'system', 
            'content': [
                "You are a live video narration assistant for visually impaired users. "
                "Provide concise, real-time descriptions of what's happening in the video. "
                "Focus on movement, actions, and important changes. "
                "Ignore static backgrounds and irrelevant details. "
                "Keep responses to 1-2 sentences maximum. "
                "Be descriptive but brief."
            ]
        },
        {
            'role': 'user', 
            'content': [curr_img, prompt_text]
        }
    ]
    
    # Add previous frame for better context if available
    if prev_img is not None:
        msgs[1]['content'].insert(-1, prev_img)  # Add prev_img before prompt_text
        print(f"🔄 [DEBUG] Added previous frame to message")
    
    print(f"🔄 [DEBUG] Calling model streaming with {len(msgs)} messages...")
    
    try:
        # Get the data channel for narration output from global storage
        peer_channels = global_data_channels.get(sid, {})
        narration_channel = peer_channels.get('narration')
        
        if not narration_channel:
            print(f"⚠️ [DEBUG] No narration channel found for peer {sid}")
            return None
        
        # Try to use streaming if available, otherwise fall back to chat
        try:
            # Attempt to use streaming methods (if available)
            response_stream = model.streaming_generate(
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )
            
            print(f"🔄 [DEBUG] Using streaming_generate for real-time tokens")
            full_response = ""
            
            # Stream tokens as they're generated
            for token_data in response_stream:
                if isinstance(token_data, dict) and 'token' in token_data:
                    token = token_data['token']
                elif isinstance(token_data, str):
                    token = token_data
                else:
                    continue
                
                # Clean token
                cleaned_token = (
                    token.replace("<|audio_sep|>", "")
                    .replace("<|tts_eos|>", "")
                    .replace("<|tts|>", "")
                )
                
                if cleaned_token.strip():
                    full_response += cleaned_token
                    
                    # Send token immediately
                    message = {
                        "type": "narration",
                        "token": cleaned_token,
                        "is_final": False
                    }
                    narration_channel.send(json.dumps(message))
                    print(f"🔄 [DEBUG] Streamed token: '{cleaned_token}'")
            
            # Send completion
            if full_response.strip():
                completion_message = {
                    "type": "narration_complete",
                    "full_text": full_response.strip()
                }
                narration_channel.send(json.dumps(completion_message))
                last_narration_per_client[sid] = full_response.strip()
                print(f"✅ [DEBUG] Streaming complete: '{full_response.strip()}'")
                return full_response.strip()
            
        except (AttributeError, TypeError) as streaming_error:
            print(f"⚠️ [DEBUG] Streaming not available ({streaming_error}), falling back to chat")
            
            # Fallback to regular chat method
            response = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )
            
            print(f"🔄 [DEBUG] Raw model response: '{response}'")
            
            # Clean up response tokens
            if response:
                cleaned_response = (
                    response.replace("<|audio_sep|>", "")
                    .replace("<|tts_eos|>", "")
                    .replace("<|tts|>", "")
                    .strip()
                )
                
                print(f"🔄 [DEBUG] Cleaned response: '{cleaned_response}'")
                
                if cleaned_response:
                    # Send complete response immediately (not word by word)
                    message = {
                        "type": "narration",
                        "token": cleaned_response,
                        "is_final": True
                    }
                    narration_channel.send(json.dumps(message))
                    
                    completion_message = {
                        "type": "narration_complete",
                        "full_text": cleaned_response
                    }
                    narration_channel.send(json.dumps(completion_message))
                    
                    last_narration_per_client[sid] = cleaned_response
                    print(f"✅ [DEBUG] Sent complete response: '{cleaned_response}'")
                    return cleaned_response
        
        return None
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in MiniCPM-o inference: {e}")
        print(f"❌ [DEBUG] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Send error via data channel
        peer_channels = global_data_channels.get(sid, {})
        narration_channel = peer_channels.get('narration')
        if narration_channel:
            error_message = {
                "type": "error",
                "message": f"Inference error: {str(e)}"
            }
            narration_channel.send(json.dumps(error_message))
        
        return f"ERROR: {str(e)}"


async def process_frame(sid: str, curr_img: Image.Image, prev_img: Optional[Image.Image]):
    """Process frame and handle streaming in thread pool."""
    print(f"🎬 [DEBUG] process_frame called for peer {sid}")
    
    # Run streaming inference in thread pool (data channel sends happen inside)
    print(f"🎬 [DEBUG] Starting streaming inference in thread pool...")
    result = await asyncio.to_thread(_process_frame_sync, sid, curr_img, prev_img)
    print(f"🎬 [DEBUG] Streaming inference completed, result: {'Success' if result else 'Skipped/None'}")
    
    return result

# ─── WebRTC signalling & media ingestion ───────────────────────────────
app = FastAPI()

# Data channel message handler
async def on_message(channel: RTCDataChannel, message: str):
    """Handle incoming messages from Flutter client via WebRTC data channel."""
    try:
        data = json.loads(message)
        msg_type = data.get('type', 'unknown')
        
        print(f"📨 [DEBUG] Received message type: {msg_type}")
        
        if msg_type == 'frame':
            # Handle incoming frame from Flutter client
            await handle_frame_message(channel, data)
        elif msg_type == 'start_narration':
            # Handle start narration request
            await handle_start_narration(channel, data)
        elif msg_type == 'end_stream':
            # Handle end stream request
            await handle_end_stream(channel, data)
        elif msg_type == 'question':
            # Handle question request
            await handle_question(channel, data)
        else:
            print(f"⚠️ [DEBUG] Unknown message type: {msg_type}")
            
    except json.JSONDecodeError as e:
        print(f"❌ [DEBUG] Error parsing JSON message: {e}")
        print(f"❌ [DEBUG] Raw message: {message[:100]}...")
    except Exception as e:
        print(f"❌ [DEBUG] Error handling message: {e}")
        import traceback
        traceback.print_exc()


async def handle_frame_message(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle incoming frame data from Flutter client."""
    try:
        # Get peer ID from channel
        peer_id = f"peer_{id(channel)}"
        
        # Extract base64 image data
        base64_image = data.get('image', '')
        timestamp = data.get('timestamp', 0)
        
        if not base64_image:
            print(f"⚠️ [DEBUG] Received frame message without image data")
            return
        
        print(f"🖼️ [DEBUG] Processing frame from {peer_id} (timestamp: {timestamp})")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            print(f"🖼️ [DEBUG] Decoded image: {image.size} - {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get previous frame for comparison
            prev_image = frame_memory.get(peer_id)
            
            # Store current frame as previous for next iteration
            frame_memory[peer_id] = image
            
            # Process the frame
            await process_frame(peer_id, image, prev_image)
            
        except Exception as decode_error:
            print(f"❌ [DEBUG] Error decoding image: {decode_error}")
            
    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_frame_message: {e}")
        import traceback
        traceback.print_exc()


async def handle_start_narration(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle start narration request."""
    try:
        peer_id = f"peer_{id(channel)}"
        client_id = data.get('client_id', 'unknown')
        
        print(f"🎬 [DEBUG] Start narration request from {peer_id} (client: {client_id})")
        
        # Send acknowledgment
        ack_message = {
            "type": "start_narration_ack",
            "status": "ready",
            "message": "Server ready for frame processing"
        }
        channel.send(json.dumps(ack_message))
        print(f"✅ [DEBUG] Sent start narration acknowledgment to {peer_id}")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_start_narration: {e}")


async def handle_end_stream(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle end stream request."""
    try:
        peer_id = f"peer_{id(channel)}"
        print(f"🛑 [DEBUG] End stream request from {peer_id}")
        
        # Clean up peer data
        if peer_id in frame_memory:
            del frame_memory[peer_id]
        if peer_id in last_narration_per_client:
            del last_narration_per_client[peer_id]
        
        # Send acknowledgment
        ack_message = {
            "type": "end_stream_ack",
            "status": "stopped"
        }
        channel.send(json.dumps(ack_message))
        print(f"✅ [DEBUG] Processed end stream for {peer_id}")
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_end_stream: {e}")


async def handle_question(channel: RTCDataChannel, data: Dict[str, Any]):
    """Handle question request."""
    try:
        peer_id = f"peer_{id(channel)}"
        question = data.get('text', '')
        
        print(f"❓ [DEBUG] Question from {peer_id}: {question}")
        
        # For now, just acknowledge the question
        # TODO: Implement question processing with current frame
        ack_message = {
            "type": "question_ack",
            "status": "received",
            "question": question
        }
        channel.send(json.dumps(ack_message))
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in handle_question: {e}")


@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        raise HTTPException(status_code=400, detail="Invalid SDP payload")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())
    pcs[peer_id] = pc
    
    # Handle data channel creation
    @pc.on("datachannel")
    def on_datachannel(channel):
        channel_id = channel.label
        print(f"🔌 New data channel {channel_id} for peer {peer_id}")
        if peer_id not in global_data_channels:
            global_data_channels[peer_id] = {}
        global_data_channels[peer_id][channel_id] = channel
        
        @channel.on("message")
        def on_message(message):
            asyncio.run(on_message(channel, message))
            
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
                
                # Get the data channel for narration output from global storage
                peer_channels = global_data_channels.get(peer_id, {})
                narration_channel = peer_channels.get('narration')
                
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
