import logging
from flask import Flask
from flask_socketio import SocketIO, emit
import base64, numpy as np
from io import BytesIO
from PIL import Image
import torch, librosa
from transformers import AutoModel, AutoTokenizer

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

def decode_chunk(pkt):
    img = Image.open(BytesIO(base64.b64decode(pkt["image_b64"]))).convert("RGB")
    aud = np.frombuffer(base64.b64decode(pkt["audio_b64"]), dtype=np.int16).astype(np.float32)
    return ["<unit>", img, aud]

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
        chunk = decode_chunk(msg)
        model.streaming_prefill(SESSION, [{"role":"user","content": chunk}], tokenizer)
        for r in model.streaming_generate(
            session_id=SESSION,
            tokenizer=tokenizer,
            temperature=0.5,
            max_new_tokens=32,
            generate_audio=False
        ):
            token = r.get("text", getattr(r, "text", ""))
            if token:
                emit('token', {"text": token})
        emit('chunk_end')
        print("📤 Chunk streamed; end signal sent", flush=True)

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