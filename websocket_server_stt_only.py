"""
WebSocket STT сервер - ТОЛЬКО транскрибация
Обычный OpenAI Whisper Small (не faster-whisper)
Максимальная скорость на GPU
"""
import warnings
warnings.filterwarnings("ignore")

# Автоустановка зависимостей
import subprocess
import sys

def install_if_missing(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        print(f"📦 Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"✅ {package} установлен")

# Проверяем и устанавливаем необходимые пакеты
install_if_missing("openai-whisper", "whisper")
install_if_missing("torch")
install_if_missing("numpy")
install_if_missing("websockets")
print()

import asyncio
import websockets
import json
import whisper
import torch
import numpy as np
from datetime import datetime
import base64
import time

# ============ ЗАГРУЗКА МОДЕЛИ ============
print("=" * 80)
print("🚀 WEBSOCKET STT СЕРВЕР (WHISPER SMALL)")
print("=" * 80)

# Форсируем GPU режим
if not torch.cuda.is_available():
    print("❌ CUDA не доступна! Установите CUDA Toolkit и GPU драйверы")
    print("   https://developer.nvidia.com/cuda-downloads")
    import sys
    sys.exit(1)

device = "cuda"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Загружаем обычный Whisper Small
print(f"📦 Загружаем Whisper Small ({device.upper()})...")
start_time = time.time()
whisper_model = whisper.load_model("small", device=device)
load_time = time.time() - start_time
print(f"✅ Модель загружена за {load_time:.2f}с\n")

print("=" * 80)
print(f"🌐 WebSocket сервер готов на ws://0.0.0.0:8765")
print(f"📊 Режим: {device.upper()}")
print("=" * 80)
print()

# Конфигурация
SAMPLE_RATE = 16000


async def handle_client(websocket, path):
    """Обработка клиента"""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"🎙️  Подключился: {client_id}")
    
    audio_buffer = []
    
    try:
        # Отправляем приветствие
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "STT server ready",
            "device": device,
            "sample_rate": SAMPLE_RATE
        }))
        
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type", "audio")
            
            # ============ АУДИО ============
            if msg_type == "audio":
                audio_chunk = np.frombuffer(
                    base64.b64decode(data["audio"]),
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                
                audio_buffer.append(audio_chunk)
            
            # ============ ФИНАЛИЗАЦИЯ ============
            elif msg_type == "finalize":
                if not audio_buffer:
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": "",
                        "is_final": True,
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                # Объединяем аудио
                audio = np.concatenate(audio_buffer)
                audio_buffer = []
                
                duration = len(audio) / SAMPLE_RATE
                
                # Транскрибируем (обычный Whisper с hotwords)
                start_time = time.time()
                result = whisper_model.transcribe(
                    audio,
                    language="en",
                    initial_prompt="Kiko, kiko, KIKO, Kiko assistant, voice assistant Kiko",
                    fp16=True
                )
                
                text = result["text"].strip()
                transcribe_time = time.time() - start_time
                rtf = transcribe_time / duration if duration > 0 else 0
                
                print(f"🧠 [{client_id}] {text!r}")
                print(f"⏱️  {duration:.2f}s аудио → {transcribe_time*1000:.0f}ms обработки (RTF: {rtf:.3f}x)")
                
                # Отправляем результат
                response = {
                    "type": "transcription",
                    "text": text,
                    "is_final": True,
                    "language": result.get("language", "ru"),
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "audio_duration_s": round(duration, 3),
                        "transcription_time_s": round(transcribe_time, 3),
                        "transcription_time_ms": round(transcribe_time * 1000, 2),
                        "realtime_factor": round(rtf, 3),
                        "samples": len(audio)
                    }
                }
                
                await websocket.send(json.dumps(response))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"👋 Отключился: {client_id}")
    except Exception as e:
        print(f"❌ Ошибка [{client_id}]: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Запуск сервера"""
    host = "0.0.0.0"
    port = 8765
    
    print(f"🎧 Ожидаю подключений...")
    
    async with websockets.serve(handle_client, host, port, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Сервер остановлен")
