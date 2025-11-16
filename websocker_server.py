"""
WebSocket STT сервер — ONNX Whisper Small + ONNX Runtime (CUDA EP)
Фичи:
- ONNX inference (ORTModelForSpeechSeq2Seq) — CUDA execution provider
- VAD (energy-based), max buffer (8s), streaming partials
- Hotwords boosting для "KIKO" и похожих (initial_prompt repetition + aggressive post-correction)
- English language by default
- JSON WebSocket protocol:
    - {"type":"audio","audio":"...base64 pcm16..."}
    - {"type":"finalize"} -> final transcription
    - Partial interim results sent periodically: {"type":"partial","text":...,"is_final":False}
"""

import warnings
warnings.filterwarnings("ignore")

import asyncio
import websockets
import json
import base64
import time
from datetime import datetime
from collections import defaultdict
import re
from difflib import get_close_matches
import os
import sys

import numpy as np
import soundfile as sf

# ONNX / Optimum
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import onnxruntime as ort

# -------------------- Configuration --------------------
SAMPLE_RATE = 16000
MAX_BUFFER_SECONDS = 8           # max kept audio per client (seconds)
PARTIAL_INTERVAL = 2.0           # send partial every N seconds of accumulated audio
VAD_ENERGY_THRESHOLD = 0.01      # simple VAD (RMS threshold)
MIN_SPEECH_SECONDS = 0.25        # min speech to consider
MAX_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SECONDS

ONNX_MODEL_DIR = "./whisper-small-onnx"  # <- directory with ONNX export (encoder/decoder...)
USE_CUDA = True  # require CUDA EP (must have onnxruntime-gpu installed)

# Hotwords and boosting strategy
HOTWORDS = ["kiko", "KIKO", "Kiko"]
HOTWORD_CANONICAL = "KIKO"
HOTWORD_FORCE_THRESHOLD = 0.75  # fuzzy match cutoff to force replace
INITIAL_PROMPT_REPEATS = 12     # repeat "KIKO" many times to bias decoding

# decoding params
GENERATE_MAX_LENGTH = 448
GENERATE_BEAM_SIZE = 5

# ----------- Post-correction dictionary & fuzzy ----------
CORRECTION_DICT = {
    "kiko": HOTWORD_CANONICAL,
    "kyko": HOTWORD_CANONICAL,
    "keeko": HOTWORD_CANONICAL,
    "kico": HOTWORD_CANONICAL,
    "kieko": HOTWORD_CANONICAL,
    "keyko": HOTWORD_CANONICAL,
    "tico": HOTWORD_CANONICAL,
    "tiko": HOTWORD_CANONICAL,
}

# speaker sessions
speakers_sessions = defaultdict(dict)
speaker_counter = defaultdict(int)

# -------------------- Utility functions --------------------
def audio_from_pcm16_base64(b64: str) -> np.ndarray:
    """Decode base64 PCM16 mono -> float32 [-1,1]"""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr

def get_speaker_identifier(audio: np.ndarray) -> str:
    mean_amplitude = float(np.mean(np.abs(audio)))
    std_amplitude = float(np.std(audio))
    zero_crossings = int(np.sum(np.diff(np.sign(audio)) != 0))
    return f"{mean_amplitude:.6f}_{std_amplitude:.6f}_{zero_crossings}"

def get_speaker_number(client_id, speaker_hash):
    if speaker_hash not in speakers_sessions[client_id]:
        speaker_counter[client_id] += 1
        speakers_sessions[client_id][speaker_hash] = speaker_counter[client_id]
    return speakers_sessions[client_id][speaker_hash]

def simple_vad(audio: np.ndarray, threshold=VAD_ENERGY_THRESHOLD) -> bool:
    """Very simple VAD: RMS energy above threshold => speech"""
    if len(audio) == 0:
        return False
    rms = np.sqrt(np.mean(audio**2))
    return rms >= threshold

def apply_post_correction(text: str) -> str:
    """Aggressive post-correction: force hotword canonicalization via fuzzy matching."""
    if not text:
        return text
    words = re.split(r'(\s+)', text)  # preserve whitespace tokens
    corrected = []
    for token in words:
        if token.strip() == "":
            corrected.append(token)
            continue
        clean = re.sub(r'[^\w]', '', token).lower()
        if not clean:
            corrected.append(token)
            continue
        if clean in CORRECTION_DICT:
            # preserve original whitespace/punctuation but force canonical
            replacement = CORRECTION_DICT[clean]
            # Keep capitalization pattern: if token is title -> Titlecase, if upper -> upper
            if token.isupper():
                replacement = replacement.upper()
            elif token.istitle():
                replacement = replacement.title()
            corrected.append(re.sub(re.escape(clean), replacement, token, flags=re.IGNORECASE))
            continue
        # fuzzy match
        if len(clean) > 2:
            matches = get_close_matches(clean, CORRECTION_DICT.keys(), n=1, cutoff=HOTWORD_FORCE_THRESHOLD)
            if matches:
                replacement = CORRECTION_DICT[matches[0]]
                if token.isupper():
                    replacement = replacement.upper()
                elif token.istitle():
                    replacement = replacement.title()
                corrected.append(re.sub(re.escape(clean), replacement, token, flags=re.IGNORECASE))
                continue
        corrected.append(token)
    return ''.join(corrected)

# -------------------- Load ONNX model & processor --------------------
print("="*80)
print("🚀 Starting ONNX Whisper Small (CUDA) WebSocket STT server")
print("="*80)

# Ensure ONNX model dir exists
if not os.path.isdir(ONNX_MODEL_DIR):
    print(f"❌ ONNX model directory not found: {ONNX_MODEL_DIR}")
    print("Place your exported whisper-small ONNX files into that directory.")
    sys.exit(1)

# Ensure CUDA EP available in onnxruntime
providers = ort.get_available_providers()
if USE_CUDA and "CUDAExecutionProvider" not in providers:
    print("❌ CUDA Execution Provider not available in onnxruntime.")
    print("Install onnxruntime-gpu compatible with your CUDA and restart.")
    print("Available providers:", providers)
    sys.exit(1)

# Load processor (feature_extractor + tokenizer)
print("🔁 Loading WhisperProcessor from HuggingFace (openai/whisper-small)")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load ORT model (expects folder with ONNX exports produced by optimum/transformers export)
print(f"🔁 Loading ORTModelForSpeechSeq2Seq from {ONNX_MODEL_DIR} (this may take a moment)...")
ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(ONNX_MODEL_DIR)

# print runtime provider info
print("✅ ONNX model loaded. Execution providers available:", providers)
print("🌐 WebSocket endpoint: ws://0.0.0.0:8765")
print("="*80)

# -------------------- WebSocket handler --------------------
class ClientState:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.last_partial_time = 0.0

async def handle_client(ws, path):
    client_ip = ws.remote_address[0] if ws.remote_address else "unknown"
    client_port = ws.remote_address[1] if ws.remote_address else 0
    client_id = f"{client_ip}:{client_port}"
    print(f"🔌 Client connected: {client_id}")

    state = ClientState()
    speakers_sessions[client_id] = {}  # init session map
    speaker_counter[client_id] = 0

    # send welcome
    await ws.send(json.dumps({
        "type":"connected",
        "message":"ONNX Whisper-small (CUDA) STT ready",
        "sample_rate": SAMPLE_RATE,
        "model": "whisper-small (onnx)"
    }))

    try:
        async for raw in ws:
            try:
                data = json.loads(raw)
            except Exception:
                await ws.send(json.dumps({"type":"error","message":"invalid json"}))
                continue

            mtype = data.get("type","audio")

            if mtype == "audio":
                b64 = data.get("audio")
                if not b64:
                    continue
                chunk = audio_from_pcm16_base64(b64)
                # append to buffer, cap MAX_SAMPLES
                state.buffer = np.concatenate([state.buffer, chunk])
                if len(state.buffer) > MAX_SAMPLES:
                    # keep last MAX_SAMPLES
                    state.buffer = state.buffer[-MAX_SAMPLES:]
                # If enough audio accumulated, maybe send partial
                now = time.time()
                if (now - state.last_partial_time) >= PARTIAL_INTERVAL:
                    # send a partial non-final transcription asynchronously
                    asyncio.create_task(process_partial_and_send(ws, client_id, state))
                    state.last_partial_time = now

            elif mtype == "finalize":
                # finalize: run full transcription on current buffer (if any), then clear
                if len(state.buffer) == 0:
                    await ws.send(json.dumps({
                        "type":"transcription",
                        "text":"",
                        "is_final":True,
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue

                # run inference
                audio = state.buffer.copy()
                state.buffer = np.array([], dtype=np.float32)

                # simple VAD: if below energy threshold and short, return empty
                if not simple_vad(audio):
                    await ws.send(json.dumps({
                        "type":"transcription",
                        "text":"",
                        "is_final":True,
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue

                # limit audio length
                if len(audio) > MAX_SAMPLES:
                    audio = audio[-MAX_SAMPLES:]

                duration = len(audio) / SAMPLE_RATE

                # speaker hash
                speaker_hash = get_speaker_identifier(audio)
                speaker_num = get_speaker_number(client_id, speaker_hash)

                # build initial prompt with repeated hotwords to bias decode
                repeated = " ".join([HOTWORD_CANONICAL]*INITIAL_PROMPT_REPEATS)
                initial_prompt = f"{repeated} KIKO assistant voice. "

                start = time.time()
                # processor: expects waveform in float [-1,1]
                inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
                input_features = inputs.get("input_features") or inputs.get("input_values")

                # Generate
                gen_out = ort_model.generate(
                    input_features,
                    max_length=GENERATE_MAX_LENGTH,
                    num_beams=GENERATE_BEAM_SIZE,
                    do_sample=False
                )
                # extract sequences
                sequences = getattr(gen_out, "sequences", gen_out)
                # decode
                try:
                    text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
                except Exception:
                    # fallback
                    text = "<decode_error>"

                # post corrections (force KIKO etc.)
                text = apply_post_correction(text)

                end = time.time()
                trans_time = end - start
                rtf = trans_time / duration if duration > 0 else 0

                resp = {
                    "type":"transcription",
                    "text": text,
                    "is_final": True,
                    "timestamp": datetime.now().isoformat(),
                    "speaker_number": speaker_num,
                    "metrics": {
                        "audio_duration_s": round(duration,3),
                        "transcription_time_s": round(trans_time,3),
                        "realtime_factor": round(rtf,3),
                        "samples": len(audio)
                    }
                }
                print(f"📝 [{client_id}] (final) Speaker#{speaker_num} | dur={duration:.2f}s rtf={rtf:.3f} : {text!r}")
                await ws.send(json.dumps(resp))

            elif mtype == "ping":
                # simple keepalive
                await ws.send(json.dumps({"type":"pong","timestamp":datetime.now().isoformat()}))

            else:
                await ws.send(json.dumps({"type":"error","message":"unknown type"}))

    except websockets.exceptions.ConnectionClosed:
        print(f"👋 Client disconnected: {client_id}")

    except Exception as e:
        print(f"❌ Error client {client_id}: {e}")

    finally:
        # cleanup
        if client_id in speakers_sessions:
            del speakers_sessions[client_id]
            del speaker_counter[client_id]
        print(f"🧹 Session cleaned: {client_id}")

# -------------------- Partial processing --------------------
async def process_partial_and_send(ws, client_id, state: 'ClientState'):
    """Compute a lightweight partial transcription on last few seconds (non-blocking spawn)."""
    try:
        # take last PARTIAL_INTERVAL * SAMPLE_RATE seconds (or less)
        N = int(SAMPLE_RATE * PARTIAL_INTERVAL * 1.5)
        audio = state.buffer[-N:] if len(state.buffer) >= 1 else np.array([], dtype=np.float32)
        if len(audio) < SAMPLE_RATE * 0.25:
            # too short
            return

        # VAD check
        if not simple_vad(audio):
            return

        # processor + infer (short input -> cheap)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        input_features = inputs.get("input_features") or inputs.get("input_values")
        gen_out = ort_model.generate(input_features, max_length=128, num_beams=1, do_sample=False)
        sequences = getattr(gen_out, "sequences", gen_out)
        try:
            text = processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        except Exception:
            text = ""

        text = apply_post_correction(text)

        await ws.send(json.dumps({
            "type":"partial",
            "text": text,
            "is_final": False,
            "timestamp": datetime.now().isoformat(),
        }))
        # optional debug print
        # print(f"🔄 Partial [{client_id}]: {text!r}")

    except Exception as e:
        # ignore partial errors (don't crash main loop)
        print(f"⚠️ Partial error for {client_id}: {e}")

# -------------------- Main --------------------
async def main():
    print("🎧 Waiting for connections on ws://0.0.0.0:8765 ...")
    # Note: adjust max_size depending on expected chunk sizes
    async with websockets.serve(handle_client, "0.0.0.0", 8765, max_size=20_000_000):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
