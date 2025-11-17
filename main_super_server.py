"""
WebSocket STT сервер - максимальная скорость
OpenAI Whisper Small на GPU
"""
import warnings
warnings.filterwarnings("ignore")

import asyncio
import websockets
import json
import whisper
import torch
import numpy as np
from datetime import datetime
import base64
import time
from collections import defaultdict
import re
from difflib import get_close_matches

# ============ ЗАГРУЗКА МОДЕЛИ ============
print("=" * 80)
print("🚀 WEBSOCKET STT SUPER SERVER (WHISPER SMALL)")
print("=" * 80)

# Форсируем GPU режим
if not torch.cuda.is_available():
    print("❌ CUDA не доступна! Установите CUDA Toolkit и GPU драйверы")
    print("   https://developer.nvidia.com/cuda-downloads")
    import sys
    sys.exit(1)

device = "cuda"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Загружаем Whisper Small
print(f"📦 Загружаем Whisper Small ({device.upper()})...")
start_time = time.time()
whisper_model = whisper.load_model("small", device=device)
load_time = time.time() - start_time
print(f"✅ Whisper загружен за {load_time:.2f}с\n")

print("=" * 80)
print(f"🌐 WebSocket сервер готов на ws://0.0.0.0:8765")
print(f"📊 Режим: {device.upper()}")
print("=" * 80)
print()

# ===============================
# НАСТРОЙКИ
# ===============================
SAMPLE_RATE = 16000

# Hotwords для boosting с весами
HOTWORDS = ["Kiko", "kiko", "KIKO", "кико", "кіко"]

# Initial prompt для контекста
INITIAL_PROMPT = "Kiko is a voice assistant. Common words: Kiko, hello, play, stop, volume, turn on, turn off."

# Словарь для post-correction (<1ms нагрузки)
CORRECTION_DICT = {
    "kiko": "Kiko",
    "kyko": "Kiko",
    "keeko": "Kiko",
    "kico": "Kiko",
    "kieko": "Kiko",
    "keyko": "Kiko",
    "tico": "Kiko",
    "tiko": "Kiko",
}

# Словарь спикеров (локальное хранение на время работы)
speakers_sessions = defaultdict(dict)
speaker_counter = defaultdict(int)

# ===============================
# UTILS
# ===============================
def apply_post_correction(text):
    """Применяем пост-коррекцию текста (<1ms)"""
    if not text:
        return text
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Убираем пунктуацию для проверки
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        
        # Проверяем точное совпадение
        if clean_word in CORRECTION_DICT:
            corrected = word.replace(clean_word, CORRECTION_DICT[clean_word])
            corrected = corrected.replace(clean_word.capitalize(), CORRECTION_DICT[clean_word])
            corrected_words.append(corrected)
        # Fuzzy match для опечаток
        elif len(clean_word) > 2:
            matches = get_close_matches(clean_word, CORRECTION_DICT.keys(), n=1, cutoff=0.8)
            if matches:
                corrected = word.replace(clean_word, CORRECTION_DICT[matches[0]])
                corrected = corrected.replace(clean_word.capitalize(), CORRECTION_DICT[matches[0]])
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)


def simple_noise_gate(audio_data, threshold=0.01):
    """Простой noise gate - обнуляем тихие участки (почти 0ms)"""
    audio_abs = np.abs(audio_data)
    mask = audio_abs > threshold
    return audio_data * mask


def get_speaker_hash(audio_data):
    """Улучшенный идентификатор спикера на основе характеристик голоса"""
    # Базовые характеристики
    mean_amplitude = np.mean(np.abs(audio_data))
    std_amplitude = np.std(audio_data)
    zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
    
    # Дополнительные характеристики для лучшего различия
    # Спектральный центроид (приблизительно)
    fft = np.fft.rfft(audio_data)
    magnitude = np.abs(fft)
    spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
    
    # Энергия в разных частотных диапазонах
    low_freq_energy = np.sum(magnitude[:len(magnitude)//4])
    high_freq_energy = np.sum(magnitude[3*len(magnitude)//4:])
    
    # Создаём более детальный "отпечаток"
    speaker_hash = f"{mean_amplitude:.5f}_{std_amplitude:.5f}_{zero_crossings}_{spectral_centroid:.2f}_{low_freq_energy:.2f}_{high_freq_energy:.2f}"
    return speaker_hash


def get_speaker_number(client_id, speaker_hash):
    """Получаем номер спикера или создаём новый"""
    if speaker_hash not in speakers_sessions[client_id]:
        speaker_counter[client_id] += 1
        speakers_sessions[client_id][speaker_hash] = speaker_counter[client_id]
    return speakers_sessions[client_id][speaker_hash]


# ===============================
# WS HANDLER
# ===============================
async def handle_client(websocket):
    """Обработка подключения клиента"""
    client_id = id(websocket)
    print(f"🔌 Клиент подключился: {client_id}")

    audio_buffer = []

    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "Real-time transcription server ready",
            "sample_rate": SAMPLE_RATE,
            "model": "small",
            "device": device,
        }))

        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "audio":
                    audio_b64 = data.get("audio") or ""
                    if not audio_b64:
                        continue
                    
                    audio_chunk = np.frombuffer(
                        base64.b64decode(audio_b64),
                        dtype=np.int16
                    ).astype(np.float32) / 32768.0
                    
                    audio_buffer.append(audio_chunk)

                elif msg_type == "finalize":
                    if not audio_buffer:
                        await websocket.send(json.dumps({
                            "type": "transcription",
                            "text": "",
                            "is_final": True,
                            "timestamp": datetime.now().isoformat(),
                        }))
                        continue

                    # Объединяем аудио
                    audio = np.concatenate(audio_buffer)
                    audio_buffer = []
                    
                    # Применяем простой noise gate (почти 0ms нагрузки)
                    audio = simple_noise_gate(audio, threshold=0.01)
                    
                    audio_duration = len(audio) / SAMPLE_RATE
                    
                    # Определяем спикера
                    speaker_hash = get_speaker_hash(audio)
                    speaker_num = get_speaker_number(client_id, speaker_hash)
                    
                    print(f"🎧 [{client_id}] Speaker #{speaker_num} | samples={len(audio)} duration={audio_duration:.3f}s")

                    # Транскрибация с hotwords и initial_prompt
                    start_time = time.perf_counter()
                    
                    result = whisper_model.transcribe(
                        audio,
                        language="en",
                        initial_prompt=INITIAL_PROMPT,
                        fp16=True
                    )
                    
                    text = result["text"].strip()
                    
                    # Применяем post-correction (<1ms)
                    text = apply_post_correction(text)
                    
                    end_time = time.perf_counter()
                    transcription_time = (end_time - start_time) * 1000  # в миллисекундах
                    
                    rtf = audio_duration / (transcription_time / 1000) if transcription_time > 0 else 0

                    print(f"📝 [{client_id}] Speaker #{speaker_num}: {text!r}")
                    print(f"⏱️  Время транскрибации: {transcription_time:.2f}ms ({transcription_time/1000:.3f}s) | RTF: {rtf:.2f}x")

                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": text,
                        "is_final": True,
                        "timestamp": datetime.now().isoformat(),
                        "speaker_number": speaker_num,
                        "metrics": {
                            "transcription_time_ms": round(transcription_time, 2),
                            "transcription_time_s": round(transcription_time / 1000, 3),
                            "audio_duration_s": round(audio_duration, 3),
                            "realtime_factor": round(rtf, 2),
                            "samples": len(audio)
                        }
                    }))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON",
                }))
            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Клиент отключился: {client_id}")
    except Exception as e:
        print(f"❌ Ошибка в handle_client: {e}")
    finally:
        # Очищаем данные сессии при отключении
        if client_id in speakers_sessions:
            total_speakers = len(speakers_sessions[client_id])
            print(f"👋 Сессия завершена: {client_id} | Всего спикеров: {total_speakers}")
            del speakers_sessions[client_id]
            del speaker_counter[client_id]
        else:
            print(f"👋 Сессия завершена: {client_id}")


async def main():
    """Запуск WebSocket сервера"""
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=20,
        max_size=10 * 1024 * 1024  # 10MB max message size
    )
    
    print("🎧 Ожидаю подключений...")
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Сервер остановлен")
