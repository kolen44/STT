"""
WebSocket STT сервер С распознаванием спикеров
Обычный OpenAI Whisper Small + SpeechBrain ECAPA
Максимальная скорость на GPU
"""
import warnings
warnings.filterwarnings("ignore")

# Патчи для совместимости (ДО импорта speechbrain!)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["sox", "soundfile"]

from huggingface_hub import hf_hub_download as original_hf_hub_download
def patched_hf_hub_download(*args, **kwargs):
    kwargs.pop('use_auth_token', None)
    return original_hf_hub_download(*args, **kwargs)

import huggingface_hub
huggingface_hub.hf_hub_download = patched_hf_hub_download

import speechbrain.utils.fetching as fetching_module
original_link = fetching_module.link_with_strategy

def patched_link_with_strategy(src, dst, strategy="auto"):
    import shutil
    from pathlib import Path
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_file():
        shutil.copy2(src_path, dst_path)
    return dst_path

fetching_module.link_with_strategy = patched_link_with_strategy

import asyncio
import websockets
import json
import whisper
import torch
import numpy as np
from datetime import datetime
import base64
import time
from speechbrain.inference.speaker import SpeakerRecognition
import pickle
from pathlib import Path

# ============ ЗАГРУЗКА МОДЕЛЕЙ ============
print("=" * 80)
print("🚀 WEBSOCKET STT + SPEAKER RECOGNITION")
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
print(f"✅ Whisper загружен за {load_time:.2f}с")

# Загружаем SpeechBrain для speaker recognition
print(f"📦 Загружаем SpeechBrain ECAPA...")
start_time = time.time()

# Предзагрузка файлов модели
target_dir = Path("pretrained_models/spkrec-ecapa-voxceleb")
target_dir.mkdir(parents=True, exist_ok=True)

required_files = ["hyperparams.yaml", "embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt", "mean_var_norm_emb.ckpt"]
print(f"📋 Загружаем файлы модели...")
for filename in required_files:
    target_file = target_dir / filename
    if not target_file.exists():
        try:
            cached_file = huggingface_hub.hf_hub_download(
                repo_id="speechbrain/spkrec-ecapa-voxceleb",
                filename=filename,
                cache_dir=str(Path.home() / ".cache" / "huggingface")
            )
            import shutil
            shutil.copy2(cached_file, target_file)
        except Exception as e:
            if "404" not in str(e):
                print(f"  ⚠️  {filename}: {e}")

speaker_model = SpeakerRecognition.from_hparams(
    source="pretrained_models/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
load_time = time.time() - start_time
print(f"✅ SpeechBrain загружен за {load_time:.2f}с\n")

print("=" * 80)
print(f"🌐 WebSocket сервер готов на ws://0.0.0.0:8766")
print(f"📊 Режим: {device.upper()}")
print(f"🎭 Speaker Recognition: ENABLED")
print("=" * 80)
print()

# Конфигурация
SAMPLE_RATE = 16000

# Hotwords для boosting с весами (чем выше вес, тем сильнее приоритет)
HOTWORDS = "Kiko:10.0, kiko:10.0, кико:8.0, кіко:8.0"

# Initial prompt для контекста (0% нагрузки)
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

SPEAKERS_DB_FILE = "speakers_database.pkl"

# База спикеров
speakers_database = {}
if os.path.exists(SPEAKERS_DB_FILE):
    try:
        with open(SPEAKERS_DB_FILE, 'rb') as f:
            speakers_database = pickle.load(f)
        print(f"📂 Загружено {len(speakers_database)} сохранённых голосов\n")
    except Exception as e:
        print(f"⚠️  Не удалось загрузить базу: {e}\n")


def save_speakers_database():
    try:
        with open(SPEAKERS_DB_FILE, 'wb') as f:
            pickle.dump(speakers_database, f)
    except Exception as e:
        print(f"❌ Ошибка сохранения базы: {e}")


def apply_post_correction(text):
    """Применяем пост-коррекцию текста (<1ms)"""
    if not text:
        return text
    
    import re
    from difflib import get_close_matches
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Убираем пунктуацию для проверки
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        
        # Проверяем точное совпадение
        if clean_word in CORRECTION_DICT:
            # Заменяем, сохраняя пунктуацию
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


def get_speaker_embedding(audio_np):
    try:
        audio_tensor = torch.from_numpy(audio_np).float()
        if device == "cuda":
            audio_tensor = audio_tensor.cuda()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        embedding = speaker_model.encode_batch(audio_tensor)
        return embedding.squeeze()
    except Exception as e:
        print(f"❌ Ошибка получения эмбеддинга: {e}")
        return None


def identify_speaker(embedding, threshold=0.25):
    if embedding is None or len(speakers_database) == 0:
        return None, 0.0
    
    best_match = None
    best_similarity = -1.0
    
    for speaker_id, data in speakers_database.items():
        similarity = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0),
            data["embedding"].unsqueeze(0)
        ).item()
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker_id
    
    if best_similarity > threshold:
        return best_match, best_similarity
    
    return None, best_similarity


def register_speaker(name, embedding):
    speaker_id = f"speaker_{len(speakers_database) + 1}"
    speakers_database[speaker_id] = {
        "name": name,
        "embedding": embedding,
        "samples_count": 1,
        "created_at": datetime.now().isoformat()
    }
    save_speakers_database()
    return speaker_id


def update_speaker_embedding(speaker_id, new_embedding, alpha=0.3):
    if speaker_id in speakers_database:
        old_embedding = speakers_database[speaker_id]["embedding"]
        updated_embedding = alpha * new_embedding + (1 - alpha) * old_embedding
        speakers_database[speaker_id]["embedding"] = updated_embedding
        speakers_database[speaker_id]["samples_count"] += 1
        save_speakers_database()


async def handle_client(websocket):
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f"🎙️  Подключился: {client_id}")
    
    audio_buffer = []
    
    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "STT + Speaker Recognition server ready",
            "device": device,
            "sample_rate": SAMPLE_RATE,
            "speakers_count": len(speakers_database)
        }))
        
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type", "audio")
            
            if msg_type == "audio":
                audio_chunk = np.frombuffer(
                    base64.b64decode(data["audio"]),
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                audio_buffer.append(audio_chunk)
            
            elif msg_type == "finalize":
                if not audio_buffer:
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": "",
                        "is_final": True,
                        "timestamp": datetime.now().isoformat()
                    }))
                    continue
                
                audio = np.concatenate(audio_buffer)
                audio_buffer = []
                
                # Применяем простой noise gate (почти 0ms нагрузки)
                audio = simple_noise_gate(audio, threshold=0.01)
                
                duration = len(audio) / SAMPLE_RATE
                
                # Транскрибация с hotwords и initial_prompt
                start_time = time.time()
                result = whisper_model.transcribe(
                    audio,
                    language="en",
                    initial_prompt=INITIAL_PROMPT,  # Контекст для модели (+0-2ms)
                    fp16=True
                )
                text = result["text"].strip()
                
                # Применяем post-correction (<1ms)
                text = apply_post_correction(text)
                
                transcribe_time = time.time() - start_time
                
                # Speaker recognition
                speaker_start = time.time()
                embedding = get_speaker_embedding(audio)
                speaker_id, similarity = identify_speaker(embedding)
                speaker_time = time.time() - speaker_start
                
                speaker_info = None
                if speaker_id:
                    speaker_info = {
                        "id": speaker_id,
                        "name": speakers_database[speaker_id]["name"],
                        "similarity": round(similarity, 3),
                        "is_known": True
                    }
                    update_speaker_embedding(speaker_id, embedding)
                else:
                    speaker_info = {
                        "id": "unknown",
                        "name": "Unknown Speaker",
                        "similarity": round(similarity, 3),
                        "is_known": False
                    }
                
                rtf = transcribe_time / duration if duration > 0 else 0
                
                print(f"🧠 [{client_id}] {speaker_info['name']}: {text!r}")
                print(f"⏱️  {duration:.2f}s аудио → {transcribe_time*1000:.0f}ms STT + {speaker_time*1000:.0f}ms speaker (RTF: {rtf:.3f}x)")
                
                response = {
                    "type": "transcription",
                    "text": text,
                    "is_final": True,
                    "language": result.get("language", "ru"),
                    "timestamp": datetime.now().isoformat(),
                    "speaker": speaker_info,
                    "metrics": {
                        "audio_duration_s": round(duration, 3),
                        "transcription_time_s": round(transcribe_time, 3),
                        "transcription_time_ms": round(transcribe_time * 1000, 2),
                        "speaker_recognition_time_ms": round(speaker_time * 1000, 2),
                        "realtime_factor": round(rtf, 3),
                        "samples": len(audio)
                    }
                }
                
                await websocket.send(json.dumps(response))
            
            elif msg_type == "register_speaker":
                speaker_name = data.get("name", "Unknown")
                
                if not audio_buffer:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "No audio data for speaker registration"
                    }))
                    continue
                
                audio = np.concatenate(audio_buffer)
                audio_buffer = []
                
                embedding = get_speaker_embedding(audio)
                if embedding is not None:
                    speaker_id = register_speaker(speaker_name, embedding)
                    print(f"✅ Зарегистрирован: {speaker_name} (ID: {speaker_id})")
                    
                    await websocket.send(json.dumps({
                        "type": "speaker_registered",
                        "speaker_id": speaker_id,
                        "name": speaker_name
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Failed to extract speaker embedding"
                    }))
            
            elif msg_type == "list_speakers":
                speakers_list = [
                    {
                        "id": sid,
                        "name": sdata["name"],
                        "samples_count": sdata["samples_count"],
                        "created_at": sdata["created_at"]
                    }
                    for sid, sdata in speakers_database.items()
                ]
                
                await websocket.send(json.dumps({
                    "type": "speakers_list",
                    "speakers": speakers_list,
                    "total": len(speakers_list)
                }))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"👋 Отключился: {client_id}")
    except Exception as e:
        print(f"❌ Ошибка [{client_id}]: {e}")
        import traceback
        traceback.print_exc()


async def main():
    host = "0.0.0.0"
    port = 8766
    
    print(f"🎧 Ожидаю подключений...")
    
    async with websockets.serve(handle_client, host, port, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Сервер остановлен")
