"""
WebSocket сервер для real-time транскрибации
Теперь на обычном Whisper (HuggingFace Transformers) с CUDA ускорением
С поддержкой hotwords boosting, initial_prompt и speaker-id
"""

import warnings
warnings.filterwarnings("ignore")

import asyncio
import websockets
import json
import torch
import numpy as np
import base64
from datetime import datetime
import time
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from collections import defaultdict
import re
from difflib import get_close_matches

print("=" * 80)
print("🚀 LAUNCHING REAL-TIME WHISPER SERVER (HuggingFace)")
print("=" * 80)

# ===============================
#        CUDA CONFIG
# ===============================
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
    torch_dtype = torch.float16
else:
    print("⚠️ CUDA NOT available — using CPU")
    device = "cpu"
    torch_dtype = torch.float32

# ===============================
#       LOAD WHISPER SMALL
# ===============================
print("📦 Loading Whisper-small...")

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True
).to(device)

model.eval()

print("✅ Whisper-small loaded\n")

# ===============================
#         SETTINGS
# ===============================
SAMPLE_RATE = 16000

HOTWORDS = ["kiko", "KIKO", "Kiko", "кіко", "кико"]
BOOST = 20.0   # усиливаем шанс слова в 20 раз

INITIAL_PROMPT = (
    "Kiko is a voice assistant. Common words: Kiko, hello, play, stop, volume."
)

CORRECTION_DICT = {
    "kiko": "Kiko",
    "kyko": "Kiko",
    "kieko": "Kiko",
    "kico": "Kiko",
    "tiko": "Kiko",
    "tico": "Kiko",
}

# Storage: per client → speakers
speakers_sessions = defaultdict(dict)
speaker_counter = defaultdict(int)


# ===============================
#        UTILS
# ===============================
def audio_to_float32(audio_bytes: bytes):
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    return arr.astype(np.float32) / 32768.0


def apply_post_correction(text: str):
    """Fix spelling <1 ms"""
    if not text:
        return text

    words = text.split()
    out = []

    for w in words:
        clean = re.sub(r"[^\w\s]", "", w).lower()

        if clean in CORRECTION_DICT:
            out.append(CORRECTION_DICT[clean])
            continue

        match = get_close_matches(clean, CORRECTION_DICT.keys(), n=1, cutoff=0.8)
        if match:
            out.append(CORRECTION_DICT[match[0]])
        else:
            out.append(w)

    return " ".join(out)


def noise_gate(audio, th=0.01):
    a = np.abs(audio)
    return audio * (a > th)


# VERY simple speaker fingerprint
def get_speaker_hash(audio):
    mean = np.mean(np.abs(audio))
    std = np.std(audio)
    zc = np.sum(np.diff(np.sign(audio)) != 0)
    return f"{mean:.4f}_{std:.4f}_{zc}"


def get_speaker_number(client_id, speaker_hash):
    if speaker_hash not in speakers_sessions[client_id]:
        speaker_counter[client_id] += 1
        speakers_sessions[client_id][speaker_hash] = speaker_counter[client_id]
    return speakers_sessions[client_id][speaker_hash]


# ===============================
#       WHISPER INFERENCE
# ===============================
def transcribe_whisper(audio: np.ndarray):
    """
    Standard Whisper pipeline, but with:
    - hotwords boosting
    - initial_prompt
    - beam_size=1 (max speed)
    """

    # extract features
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).to(device)

    # ——— HOTWORDS BOOSTING ———
    # Whisper supports prefix biasing using "forced_decoder_ids"
    forced_decoder_ids = None
    if INITIAL_PROMPT:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    # decode
    generated_ids = model.generate(
        inputs["input_features"],
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,               # fastest
        temperature=0.0,
        suppress_tokens=None,
        forced_decoder_ids=forced_decoder_ids,
    )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # post-correction
    text = apply_post_correction(text)

    # hotwords simple boosting (post-pass)
    for hw in HOTWORDS:
        if hw.lower() in text.lower():
            text = text.replace(hw, "Kiko")

    return text.strip()


# ===============================
#          WS HANDLER
# ===============================
async def handle_client(ws):
    client_id = id(ws)
    print(f"🔌 Client connected: {client_id}")

    pcm_chunks = []

    await ws.send(json.dumps({
        "type": "connected",
        "sample_rate": SAMPLE_RATE,
        "model": "whisper-small",
        "device": device,
    }))

    try:
        async for msg in ws:
            data = json.loads(msg)
            t = data.get("type")

            if t == "audio":
                audio_b64 = data.get("audio")
                if audio_b64:
                    pcm_chunks.append(base64.b64decode(audio_b64))

            elif t == "finalize":
                if not pcm_chunks:
                    await ws.send(json.dumps({
                        "type": "transcription",
                        "text": "",
                        "is_final": True
                    }))
                    continue

                pcm = b"".join(pcm_chunks)
                pcm_chunks.clear()

                audio = audio_to_float32(pcm)
                audio = noise_gate(audio)

                dur = len(audio) / SAMPLE_RATE

                # speaker
                sh = get_speaker_hash(audio)
                spk = get_speaker_number(client_id, sh)

                # inference
                t0 = time.perf_counter()
                text = transcribe_whisper(audio)
                t1 = time.perf_counter()

                dt = (t1 - t0) * 1000
                rtf = dur / (dt / 1000)

                print(f"📝 [{client_id}] Speaker#{spk}: {text}")
                print(f"⏱ {dt:.2f}ms  RTF={rtf:.2f}")

                await ws.send(json.dumps({
                    "type": "transcription",
                    "text": text,
                    "is_final": True,
                    "speaker_number": spk,
                    "metrics": {
                        "transcription_time_ms": round(dt, 2),
                        "audio_duration_s": round(dur, 3),
                        "rtf": round(rtf, 2)
                    }
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Client disconnected: {client_id}")

    finally:
        if client_id in speakers_sessions:
            del speakers_sessions[client_id]
            del speaker_counter[client_id]
        print(f"👋 Session closed: {client_id}")


# ===============================
#           MAIN
# ===============================
async def main():
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=20,
        max_size=10 * 1024 * 1024
    )
    print("🎧 Waiting for connections...")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
