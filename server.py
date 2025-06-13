import logging
from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64, numpy as np
from io import BytesIO
from PIL import Image
import torch
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

# SESSION_ID_PLACEHOLDER = "session_file" # This might need to be dynamic per client

# ─── Stream Processing Configuration ────────────────────────────
MAX_VIDEO_HISTORY = 30  # Number of tokens to include from previous narration (can be revisited)
# recent_frames = deque(maxlen=MAX_RECENT_FRAMES) # Replaced by client_frame_buffers
# frame_timestamps = deque(maxlen=MAX_RECENT_FRAMES) # Replaced by client_frame_buffers
last_narration_per_client = {} # Store last narration per client session
stream_start_time_per_client = {} # Store stream start time per client

# New: Per-client frame buffers
client_frame_buffers = {}
FRAMES_PER_BATCH = 3  # Number of frames to collect before sending to model (tune this)
PROCESSING_TIMEOUT_SECONDS = 3 # Max time to wait before processing a smaller batch

def decode_image_from_base64(image_b64_string):
    """Decode image from base64 encoded string."""
    try:
        return Image.open(BytesIO(base64.b64decode(image_b64_string))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# frame_too_similar function might be removed or adapted later for sequences.
# For now, let's simplify and remove it to focus on frame batching.

# ─── WebSocket handlers ───────────────────────────────────────
@socketio.on('connect')
def on_connect(auth=None): # Add auth=None to accept optional argument
    sid = request.sid
    print(f"🟢 Client connected: {sid} → initializing session", flush=True)
    
    # Initialize resources for this session
    client_frame_buffers[sid] = deque()
    last_narration_per_client[sid] = ""
    stream_start_time_per_client[sid] = time.time() # Initialize stream start time

    # Reset model session state for this client. reset_session() clears KV cache for a new logical conversation.
    # The session_id (sid) is then used in subsequent streaming_prefill/generate calls.
    model.reset_session()

    sys_msg = model.get_sys_prompt(mode='omni', language='en')
    model.streaming_prefill(sid, [sys_msg], tokenizer)
    
    live_instr = (
        "You are a live video narrator for a visually impaired user, acting as their eyes. Your descriptions should be immediate, concise, and in the present tense. "
        "Start your sentences with phrases like 'You are looking at...' or 'You see...'. "
        "Focus on describing actions, movements, and changes in the scene rather than just listing static objects. "
        "For example, instead of 'There is a cup on the table,' say 'You see a cup sitting on the table.' or if it's moving 'Someone is picking up the cup.' "
        "Narrate the events as they unfold, token-by-token, to provide a real-time understanding of the environment."
    )
    model.streaming_prefill(sid, [{"role":"user","content": live_instr}], tokenizer)
    print(f"✅ Session initialized for {sid}", flush=True)

@socketio.on('disconnect')
def on_disconnect(auth=None): # Add auth=None to accept optional argument
    sid = request.sid
    print(f"🔴 Client disconnected: {sid} → cleaning up session resources", flush=True)
    if sid in client_frame_buffers:
        del client_frame_buffers[sid]
    if sid in last_narration_per_client:
        del last_narration_per_client[sid]
    if sid in stream_start_time_per_client:
        del stream_start_time_per_client[sid]
    # Potentially inform the model to clear any session state if necessary,
    # though often this is handled by just not using the session_id anymore.

def process_frame_batch(sid):
    if sid not in client_frame_buffers or not client_frame_buffers[sid]:
        return

    frames_to_process = list(client_frame_buffers[sid])
    client_frame_buffers[sid].clear() # Clear buffer for this client

    if not frames_to_process:
        print(f"[{sid}] No frames to process.", flush=True)
        return

    print(f"[{sid}] Processing batch of {len(frames_to_process)} frames.", flush=True)

    # Construct prompt for the model
    last_narration = last_narration_per_client.get(sid, "")
    if last_narration:
        prompt_text = f"(Previously: {last_narration}) What's happening now?"
    else:
        prompt_text = "What's happening now?"

    # The model expects a list of images and text in the content
    # Ensure frames_to_process contains PIL.Image objects
    model_input_content = frames_to_process + [prompt_text]

    model.streaming_prefill(sid, [{
        "role": "user",
        "content": model_input_content
    }], tokenizer)

    current_narration_segment = ""
    # Pass sid as the first argument for session_id
    for r in model.streaming_generate(
        sid, 
        images_tensor=None,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.1, # Lower temperature for more factual, less creative narration
        do_sample=True, # Allow sampling based on temperature
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        generate_audio=False, # Audio generation handled by client or separate TTS
        # history=last_narration_per_client.get(sid, "") # Pass previous narration as history
        # max_video_history=MAX_VIDEO_HISTORY # Control history length
    ):
        token = r.get("text", getattr(r, "text", "")).replace("<|audio_sep|>", "")
        if token.strip():
            current_narration_segment += token
            emit('token', {"text": token}, room=sid)
    
    if current_narration_segment:
        last_narration_per_client[sid] = (last_narration_per_client.get(sid, "") + " " + current_narration_segment).strip()
        
    emit('narration_segment_end', room=sid) # New event to signal end of a narration segment
    print(f"[{sid}] 📤 Narration segment sent. Total frames processed for this batch: {len(frames_to_process)}", flush=True)


@socketio.on('message')
def handle_message(msg):
    sid = request.sid
    msg_type = msg.get("type", "unknown")
    print(f"[{sid}] 📥 Received message type={msg_type}", flush=True)

    if msg_type == "init":
        txt = msg.get("text", "")
        print(f"[{sid}] 📝 Instruction: {txt}")
        # This initial instruction might be for general interaction,
        # separate from the live narration prompt set in on_connect
        model.streaming_prefill(sid, [{"role":"user","content": txt}], tokenizer)
        emit('ack', {"text": "Instruction received"}, room=sid)

    elif msg_type == "video_frame":
        if sid not in client_frame_buffers:
            print(f"[{sid}] ⚠️ Received video_frame for unknown session. Ignoring.", flush=True)
            return

        image_b64 = msg.get("image_b64")
        if not image_b64:
            print(f"[{sid}] ⚠️ video_frame message missing image_b64. Ignoring.", flush=True)
            return

        img = decode_image_from_base64(image_b64)
        if img:
            client_frame_buffers[sid].append(img)
            # Log frame addition without printing the full base64 string
            print(f"[{sid}] 🖼️ Frame added to buffer (image data not shown). Buffer size: {len(client_frame_buffers[sid])}", flush=True)

            if len(client_frame_buffers[sid]) >= FRAMES_PER_BATCH:
                process_frame_batch(sid)
        else:
            print(f"[{sid}] ⚠️ Failed to decode image from video_frame. Ignoring.", flush=True)


    elif msg_type == "question": # Existing question handling
        question = msg.get("text", "")
        print(f"[{sid}] ❓ Question received: {question}")
        # Ensure the question is processed in the context of the current session
        model.streaming_prefill(sid, [{"role":"user","content": question}], tokenizer)
        ans_tokens = []
        # Pass sid as the first argument for session_id, and images=None for text-only questions
        for r in model.streaming_generate(
            sid,
            images=None, 
            tokenizer=tokenizer,
            temperature=0.1, # Typically lower temp for factual answers
            do_sample=True, # Allow sampling based on temperature
            generate_audio=False
        ):
            tok = r.get("text", getattr(r, "text", ""))
            if tok:
                ans_tokens.append(tok)
        answer = "".join(ans_tokens).strip()
        print(f"[{sid}] 📤 Answer: {answer}")
        emit('answer', {"text": answer}, room=sid)

    elif msg_type == "end_stream_request": # Client signals end of their video stream
        print(f"[{sid}] 🔴 Client requested end of stream. Processing any remaining frames.", flush=True)
        process_frame_batch(sid) # Process any frames left in the buffer
        # Optionally, emit a final confirmation or perform other cleanup
        emit('stream_ended_ack', room=sid)

    else:
        print(f"[{sid}] ⚠️ Unknown message type: {msg_type}", flush=True)
        emit('error', {"msg": f"Unknown message type: {msg_type}"}, room=sid)

# Timer to process frames if batch size isn't met within timeout
# This requires a background scheduler or a more complex async setup.
# For now, we'll rely on FRAMES_PER_BATCH.
# A simple alternative could be a check within the "video_frame" handler
# based on time since last processing, but that's less robust.

if __name__ == '__main__':
    print("🚀 Starting Flask-SocketIO Omni server on port 8123")
    # Consider using a more production-ready WSGI server like gunicorn for SocketIO
    # e.g., gunicorn --worker-class eventlet -w 1 module:app
    socketio.run(app, host='0.0.0.0', port=8123, debug=False, use_reloader=False)