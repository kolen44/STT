"""
WebSocket server for real-time Whisper transcription
Optimized for RTX 4050:
- OpenAI Whisper (проверенный, работающий)
- CUDA FP16
- Hotwords + Initial prompt
- Speaker tracking
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
import whisper
from collections import defaultdict
import re
from difflib import get_close_matches

print("="*80)
print("🚀 LAUNCHING MAX-SPEED REAL-TIME WHISPER SERVER")
print("="*80)

# ===============================
# CUDA CONFIG
# ===============================
if not torch.cuda.is_available():
    print("❌ CUDA не доступна! Установите CUDA Toolkit и GPU драйверы")
    print("   https://developer.nvidia.com/cuda-downloads")
    import sys
    sys.exit(1)

device = "cuda"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# ===============================
# LOAD MODEL
# ===============================
print("📦 Загружаем Whisper Small...")
start_time = time.time()
whisper_model = whisper.load_model("small", device=device)
load_time = time.time() - start_time
print(f"✅ Whisper загружен за {load_time:.2f}с\n")

# ===============================
# SETTINGS
# ===============================
SAMPLE_RATE = 16000
HOTWORDS = ["kiko", "KIKO", "Kiko", "кіко", "кико"]
BOOST = 20.0
INITIAL_PROMPT = "Kiko is a voice assistant. Common words: Kiko, hello, play, stop, volume."
CORRECTION_DICT = {
    "kiko": "Kiko", "kyko": "Kiko", "kieko": "Kiko", "kico": "Kiko", "tiko": "Kiko", "tico": "Kiko"
}

speakers_sessions = defaultdict(dict)
speaker_counter = defaultdict(int)
# encoder_cache = {}  # KV-cache per client

# ===============================
# UTILS
# ===============================
def audio_to_float32(audio_bytes: bytes):
    arr = np.frombuffer(audio_bytes, dtype=np.int16)
    return arr.astype(np.float32)/32768.0

def apply_post_correction(text: str):
    if not text: return text
    words = text.split()
    out = []
    for w in words:
        clean = re.sub(r"[^\w\s]","",w).lower()
        if clean in CORRECTION_DICT:
            out.append(CORRECTION_DICT[clean])
            continue
        match = get_close_matches(clean, CORRECTION_DICT.keys(), n=1, cutoff=0.8)
        out.append(CORRECTION_DICT[match[0]] if match else w)
    return " ".join(out)

def noise_gate(audio, th=0.01):
    return audio * (np.abs(audio) > th)

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
# WHISPER INFERENCE with KV-cache
# ===============================
def transcribe_whisper(audio: np.ndarray, client_id: int):
    # --- защита от тишины / мусора ---
    if audio.size == 0:
        return ""

    max_amp = float(np.max(np.abs(audio)))
    if not np.isfinite(max_amp) or max_amp < 1e-4:
        # почти тишина → сразу пусто
        return ""

    # 1) Готовим вход для Whisper — processor всегда даёт float32 на CPU
    raw_inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    # 2) Признаки → на нужный девайс и в нужный dtype (float16 на CUDA / float32 на CPU)
    input_features = raw_inputs["input_features"].to(device=device, dtype=dtype)

    # 3) Каноничный способ для Whisper: forced_decoder_ids от языка и задачи
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="en",      # если говоришь по-русски — можешь поставить "ru"
        task="transcribe",
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = text.strip()

    # Пост-коррекция "Kiko"
    text = apply_post_correction(text)

    # Hotwords post-pass — добиваем “Kiko” в нужный вид
    for hw in HOTWORDS:
        if hw.lower() in text.lower():
            text = text.replace(hw, "Kiko")

    return text



# ===============================
# WS HANDLER
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
                    await ws.send(json.dumps({"type": "transcription", "text": "", "is_final": True}))
                    continue

                pcm = b"".join(pcm_chunks)
                pcm_chunks.clear()

                audio = audio_to_float32(pcm)
                audio = noise_gate(audio)

                dur = len(audio)/SAMPLE_RATE

                # speaker
                sh = get_speaker_hash(audio)
                spk = get_speaker_number(client_id, sh)

                # inference
                t0 = time.perf_counter()
                text = transcribe_whisper(audio, client_id)
                t1 = time.perf_counter()

                dt = (t1 - t0)*1000
                rtf = dur/(dt/1000)

                print(f"📝 [{client_id}] Speaker#{spk}: {text}")
                print(f"⏱ {dt:.2f}ms  RTF={rtf:.2f}")

                await ws.send(json.dumps({
                    "type": "transcription",
                    "text": text,
                    "is_final": True,
                    "speaker_number": spk,
                    "metrics": {
                        "transcription_time_ms": round(dt,2),
                        "audio_duration_s": round(dur,3),
                        "rtf": round(rtf,2)
                    }
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Client disconnected: {client_id}")
    finally:
        if client_id in speakers_sessions:
            del speakers_sessions[client_id]
            del speaker_counter[client_id]
        # if client_id in encoder_cache:
        #     del encoder_cache[client_id]
        print(f"👋 Session closed: {client_id}")

# ===============================
# MAIN
# ===============================
async def main():
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=20,
        max_size=10*1024*1024
    )
    print("🎧 Waiting for connections...")
    await server.wait_closed()

if __name__=="__main__":
    asyncio.run(main())
