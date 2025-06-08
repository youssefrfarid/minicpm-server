import logging
from flask import Flask
from flask_socketio import SocketIO, emit
import base64, numpy as np
from io import BytesIO
from PIL import Image
import torch, librosa
import time
from transformers import AutoModel, AutoTokenizer
from collections import deque

# ─── Configure Flask + SocketIO ─────────────────────────────
logging.getLogger('werkzeug').setLevel(logging.WARNING)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# ─── Load MiniCPM-o-2_6 omni model once (no TTS) ───────────────────
print("🔄 Loading MiniCPM-o-2_6 omni model…", flush=True)
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16,
    init_vision=True, init_audio=True, init_tts=False
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)
print("✅ Model loaded", flush=True)

SESSION = "session_file"

# ─── Stream Processing Configuration ────────────────────────────
# Keep track of recent frames and their descriptions to avoid repetition
MAX_RECENT_FRAMES = 5  # Number of recent frames to track for similarity checks
MAX_VIDEO_HISTORY = 30  # Number of tokens to include from previous narration
recent_frames = deque(maxlen=MAX_RECENT_FRAMES)
frame_timestamps = deque(maxlen=MAX_RECENT_FRAMES)
last_narration = ""
stream_start_time = None

def decode_chunk(pkt):
    """Decode image and audio from base64 encoded packet."""
    img = Image.open(BytesIO(base64.b64decode(pkt["image_b64"]))).convert("RGB")
    aud = np.frombuffer(base64.b64decode(pkt["audio_b64"]), dtype=np.int16).astype(np.float32)
    return ["<unit>", img, aud]


def frame_too_similar(new_frame):
    """Check if a new frame is too similar to recently processed frames to avoid redundant narration."""
    # Skip similarity check if no frames processed yet
    if not recent_frames:
        return False
        
    # Very basic similarity check - this could be improved with better metrics
    # Here we just resize to smaller dimensions for faster comparison
    new_small = new_frame.resize((64, 64))
    new_array = np.array(new_small)
    
    for old_frame in recent_frames:
        old_small = old_frame.resize((64, 64))
        old_array = np.array(old_small)
        
        # Calculate mean squared error as simple similarity metric
        mse = np.mean((old_array - new_array) ** 2)
        
        # If MSE is below threshold, frames are too similar
        if mse < 100:  # This threshold can be tuned
            return True
            
    return False

# ─── WebSocket handlers ───────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print("🟢 Client connected → sending system and narrator prompts", flush=True)
    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    model.reset_session()
    model.streaming_prefill(SESSION, [sys_msg], tokenizer)
    live_instr = (
        "You are a live video narrator. Speak in short, present-tense phrases, "
        "describing only the most important actions as they happen, and emit them token-by-token."
    )
    model.streaming_prefill(SESSION, [{"role":"user","content": live_instr}], tokenizer)

@socketio.on('message')
def handle_message(msg):
    t = msg.get("type","init")
    print(f"📥 Received message type={t}", flush=True)

    if t == "init":
        txt = msg["text"]
        print(f"📝 Instruction: {txt}", flush=True)
        model.streaming_prefill(SESSION, [{"role":"user","content": txt}], tokenizer)
        emit('ack', {"text": "Instruction received"})

    elif t == "chunk":
        global last_narration, stream_start_time, recent_frames, frame_timestamps
        
        # Initialize start time if this is the first chunk
        if stream_start_time is None:
            stream_start_time = time.time()
        
        # Mark the timestamp of this frame (seconds since stream started)
        current_time = time.time() - stream_start_time
        
        # Decode the frame and audio
        chunk_data = decode_chunk(msg)
        img = chunk_data[1]  # Image is the second element in the list
        
        # Check if this frame is too similar to recent ones
        if frame_too_similar(img):
            print("📊 Frame too similar to recent ones, skipping narration", flush=True)
            emit('chunk_end')
            return

        # Store this frame for future similarity checks
        recent_frames.append(img)
        frame_timestamps.append(current_time)
        
        # Construct a streaming-optimized prompt that includes previous context
        # This helps maintain continuity in the narration
        context_prompt = f"Continue narrating the video. Previously described: '{last_narration[-MAX_VIDEO_HISTORY:]}'" \
                        if last_narration else "Describe what you see in this video frame for a visually impaired person"
        
        # Send the frame to the model with context
        model.streaming_prefill(SESSION, [{
            "role": "user", 
            "content": [context_prompt, img]
        }], tokenizer)
        
        # Collect the narration for this frame
        frame_narration = ""
        
        # Stream tokens back to client
        for r in model.streaming_generate(
            session_id=SESSION,
            tokenizer=tokenizer,
            temperature=0.5,
            generate_audio=False
        ):
            token = r.get("text", getattr(r, "text", ""))
            if token:
                # Filter out TTS markers that might be in the output
                if "<|tts_eos|>" in token:
                    token = token.replace("<|tts_eos|>", "")
                    
                if token.strip():
                    frame_narration += token
                    emit('token', {"text": token})
        
        # Update the last narration with this frame's description
        if frame_narration:
            last_narration += " " + frame_narration
            
        emit('chunk_end')
        print(f"📤 Chunk #{len(recent_frames)} narrated; end signal sent", flush=True)

    elif t == "question":
        question = msg.get("text", "")
        print(f"❓ Question received: {question}", flush=True)
        model.streaming_prefill(SESSION, [{"role":"user","content": question}], tokenizer)
        ans_tokens = []
        for r in model.streaming_generate(
            session_id=SESSION,
            tokenizer=tokenizer,
            temperature=0.0,
            max_new_tokens=32,
            generate_audio=False
        ):
            tok = r.get("text", getattr(r, "text", ""))
            if tok:
                ans_tokens.append(tok)
        answer = "".join(ans_tokens).strip()
        print(f"📤 Answer: {answer}", flush=True)
        emit('answer', {"text": answer})

    elif t == "end":
        print("🔴 End of stream: final flush", flush=True)
        for r in model.streaming_generate(
            session_id=SESSION,
            tokenizer=tokenizer,
            temperature=0.5,
            max_new_tokens=256,
            generate_audio=False
        ):
            token = r.get("text", getattr(r, "text", ""))
            if token:
                emit('token', {"text": token})
        emit('end')

    else:
        print("⚠️ Unknown message type", flush=True)
        emit('error', {"msg": "unknown type"})

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO Omni server on port 8123", flush=True)
    socketio.run(app, host='0.0.0.0', port=8123)