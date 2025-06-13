"""Unified ASGI server for LumiMate live narration.

This file merges:
  • Flask-SocketIO text/tokens server (previous `server.py`)
  • WebRTC video ingress server (previous `webrtc_server.py`)
into one asyncio-friendly FastAPI / python-socketio ASGI application.

Run with:
    uvicorn lib.services.omni_server:asgi_app --host 0.0.0.0 --port 8123

Socket paths / ports
    /socket.io        – Socket.IO websocket (narration tokens, control)
    POST /offer       – WebRTC signalling (SDP offer → answer)
Media (UDP) ports are negotiated via ICE; open 10000-10100/UDP on the VM.

Dependencies (pip install):
    fastapi uvicorn[standard] python-socketio[asgi] aiortc opencv-python

Notes
-----
• MiniCPM-o model is loaded once and shared by both paths.
• All heavy model calls run in a background thread via asyncio.to_thread so
  the event-loop stays responsive.
• Only video is streamed over WebRTC; audio is dropped.  The existing
  sentence-level TTS buffer stays unchanged on the phone.
"""
from __future__ import annotations
import asyncio
import base64
import logging
import uuid
from collections import deque
from io import BytesIO
from typing import Dict, Optional

import cv2
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole
import socketio  # python-socketio ASGI implementation

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

    # Model expects list[images] + text
    content = ([] if prev_img is None else [prev_img]) + [curr_img] + [prompt_text]
    model.streaming_prefill(sid, [{"role": "user", "content": content}], tokenizer)

    current_segment = ""
    for r in model.streaming_generate(
        sid,
        images_tensor=None,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        generate_audio=False,
    ):
        token = r.get("text", getattr(r, "text", "")).replace("<|audio_sep|>", "")
        if token.strip():
            current_segment += token
            asyncio.run_coroutine_threadsafe(
                sio.emit("token", {"text": token}, room=sid), sio.loop
            )

    if current_segment:
        last_narration_per_client[sid] = current_segment.strip()

    asyncio.run_coroutine_threadsafe(sio.emit("narration_segment_end", room=sid), sio.loop)
    print(f"[{sid}] 📤 Narration segment sent.")


async def process_frame(sid: str, curr_img: Image.Image, prev_img: Optional[Image.Image]):
    await asyncio.to_thread(_process_frame_sync, sid, curr_img, prev_img)

# ─── Socket.IO setup (ASGI) ─────────────────────────────────────────────

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()

@sio.event
async def connect(sid, environ, auth):
    print(f"🟢 Client connected: {sid}")
    last_narration_per_client[sid] = ""
    previous_frame_per_client[sid] = None

    model.reset_session()
    sys_msg = model.get_sys_prompt(mode="omni", language="en")
    model.streaming_prefill(sid, [sys_msg], tokenizer)

    live_instr = (
        "You are a live video narrator for a visually impaired user, acting as their eyes. "
        "Your descriptions should be immediate, concise, and in the present tense. "
        "Start sentences with 'You are looking at…' or 'You see…'."
    )
    model.streaming_prefill(sid, [{"role": "user", "content": live_instr}], tokenizer)

@sio.event
async def disconnect(sid):
    print(f"🔴 Client disconnected: {sid}")
    last_narration_per_client.pop(sid, None)
    previous_frame_per_client.pop(sid, None)

@sio.on("message")
async def handle_message(sid, msg):
    msg_type = msg.get("type", "unknown")
    if msg_type == "video_frame":
        img_b64 = msg.get("image_b64")
        if not img_b64:
            return
        img = _decode_image_from_base64(img_b64)
        if img is None:
            return
        prev_img = previous_frame_per_client.get(sid)
        await process_frame(sid, img, prev_img)
        previous_frame_per_client[sid] = img
    elif msg_type == "init":
        txt = msg.get("text", "")
        model.streaming_prefill(sid, [{"role": "user", "content": txt}], tokenizer)
        await sio.emit("ack", {"text": "Instruction received"}, room=sid)
    elif msg_type == "question":
        q = msg.get("text", "")
        model.streaming_prefill(sid, [{"role": "user", "content": q}], tokenizer)
        # simple blocking answer generation
        answer_tokens = []
        for r in model.streaming_generate(
            sid,
            images=None,
            tokenizer=tokenizer,
            generate_audio=False,
        ):
            t = r.get("text", getattr(r, "text", ""))
            if t:
                answer_tokens.append(t)
        answer = "".join(answer_tokens).strip()
        await sio.emit("answer", {"text": answer}, room=sid)
    elif msg_type == "end_stream_request":
        await sio.emit("stream_ended_ack", room=sid)

# ─── WebRTC signalling & media ingestion ───────────────────────────────
pcs: Dict[str, RTCPeerConnection] = {}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    if "sdp" not in params or "type" not in params:
        raise HTTPException(status_code=400, detail="Invalid SDP payload")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    peer_id = str(uuid.uuid4())
    pcs[peer_id] = pc

    recorder = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            print(f"📺 Video track for peer {peer_id}")
            prev_img = None
            while True:
                frame = await track.recv()
                bgr = frame.to_ndarray(format="bgr24")
                bgr = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                await process_frame(peer_id, pil, prev_img)
                prev_img = pil
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

# ─── Combine ASGI apps ─────────────────────────────────────────────────
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)
