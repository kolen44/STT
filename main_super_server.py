"""
WebSocket STT сервер v2.0 - ChatGPT-style диалоговый режим
OpenAI Whisper Small на GPU

Основные улучшения:
- Интеллектуальное определение пауз (адаптивное 0.5-1.5 сек вместо 6 сек)
- Streaming partial results во время речи
- Proper sentence boundary detection
- Корректная обработка множественных "Kiko"
- Real-time feedback как в ChatGPT
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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

# ============ ЗАГРУЗКА МОДЕЛИ ============
print("=" * 80)
print("🚀 WEBSOCKET STT SUPER SERVER v2.0 (ChatGPT-style)")
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
print(f"📊 Режим: {device.upper()} | ChatGPT-style диалог")
print("=" * 80)
print()

# ===============================
# НАСТРОЙКИ - ОПТИМИЗИРОВАННЫЕ ДЛЯ ДИАЛОГА
# ===============================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16


# === VAD настройки - ОПТИМИЗИРОВАНО ДЛЯ МАКСИМАЛЬНОЙ СКОРОСТИ ===
class VADConfig:
    # Порог энергии для определения речи
    ENERGY_THRESHOLD = 0.010  # Снижен для более быстрого старта
    
    # Минимальная энергия для транскрибации (защита от галлюцинаций)
    MIN_AUDIO_ENERGY = 0.012  # Снижен
    
    # Адаптивные паузы - УВЕЛИЧЕНЫ чтобы не обрезать речь
    MIN_PAUSE_MS = 600        # 600мс для коротких фраз (было 400)
    DEFAULT_PAUSE_MS = 800    # 800мс стандарт (было 550)
    MAX_PAUSE_MS = 1200       # 1200мс макс для длинных предложений (было 800)
    QUESTION_PAUSE_MS = 650   # 650мс для вопросов (было 450)
    
    # Минимальная длительность речи
    MIN_SPEECH_MS = 150       # 150мс - быстрее реагируем
    
    # Максимальная длительность сегмента
    MAX_SEGMENT_MS = 60000    # 60 секунд
    
    # Порог для мягкой финализации
    SOFT_SEGMENT_MS = 25000   # 25 сек
    
    # Частота partial - УВЕЛИЧЕНА для real-time
    PARTIAL_INTERVAL_MS = 200  # Каждые 200мс - очень быстро!
    
    # Размер VAD фрейма
    FRAME_MS = 20             # 20мс фреймы - быстрее!
    
    # Количество фреймов для начала речи
    SPEECH_START_FRAMES = 2   # 2 фрейма = 40мс для старта
    
    # ДЕДУПЛИКАЦИЯ - минимальный интервал между одинаковыми текстами
    DEDUP_WINDOW_MS = 3000    # 3 секунды


# === Hotwords для boosting ===
HOTWORDS = ["Kiko", "kiko", "KIKO", "кико", "кіко", "Кико"]

# Initial prompt - МИНИМАЛЬНЫЙ чтобы избежать галлюцинаций!
# НЕ используем "assistant", "voice assistant" и т.д. - Whisper повторяет их!
INITIAL_PROMPT = None  # Отключён для предотвращения галлюцинаций

# Расширенный словарь для post-correction - МНОГО вариантов для надёжного распознавания
CORRECTION_DICT = {
    # Английские варианты (все возможные произношения)
    "kiko": "Kiko", "kyko": "Kiko", "keeko": "Kiko", "kico": "Kiko",
    "kieko": "Kiko", "keyko": "Kiko", "tico": "Kiko", "tiko": "Kiko",
    "keco": "Kiko", "cico": "Kiko", "qico": "Kiko", "kika": "Kiko",
    "kikko": "Kiko", "keko": "Kiko", "chico": "Kiko", "kiku": "Kiko",
    "keeco": "Kiko", "kigo": "Kiko", "kijo": "Kiko", "kito": "Kiko",
    "kikou": "Kiko", "kicco": "Kiko", "kyco": "Kiko", "kiko's": "Kiko",
    "quico": "Kiko", "keko": "Kiko", "keekou": "Kiko", "kikov": "Kiko",
    "kikoу": "Kiko", "keiko": "Kiko", "kico": "Kiko", "kykou": "Kiko",
    # Whisper часто слышит как "key call" или "he called"
    "keycall": "Kiko", "key-call": "Kiko",
    # Русские варианты
    "кико": "Kiko", "кіко": "Kiko", "кика": "Kiko", "кеко": "Kiko", "тико": "Kiko",
    "кику": "Kiko", "кіку": "Kiko", "кико.": "Kiko", "кико,": "Kiko",
    "киго": "Kiko", "кійко": "Kiko", "кийко": "Kiko",
}

# Паттерны для определения типа фразы
QUESTION_PATTERNS = [
    r'\?$',
    r'^(what|who|where|when|why|how|can|could|would|should|is|are|do|does|did)\b',
    r'^(что|кто|где|когда|почему|как|можно|могу)\b',
]

COMMAND_PATTERNS = [
    r'^(play|stop|pause|next|previous|volume|mute|unmute)\b',
    r'^(включи|выключи|поставь|следующ|предыдущ|громкость)\b',
    r'^(search|find|show|open|close|start|turn)\b',
    r'^(найди|покажи|открой|закрой|запусти)\b',
]

SHORT_RESPONSE_PATTERNS = [
    r'^(yes|no|ok|okay|sure|yeah|yep|nope|maybe)$',
    r'^(да|нет|окей|ладно|хорошо|может)$',
]


# ===============================
# СОСТОЯНИЕ КЛИЕНТА
# ===============================
class SpeechState(Enum):
    SILENCE = 0      # Тишина, ожидание
    SPEECH = 1       # Активная речь
    PAUSE = 2        # Пауза в речи (может продолжиться)


@dataclass
class ClientSession:
    """Состояние клиентской сессии"""
    client_id: str
    
    # Аудио буферы
    audio_buffer: List[np.ndarray] = field(default_factory=list)
    speech_buffer: List[np.ndarray] = field(default_factory=list)
    
    # Состояние VAD
    state: SpeechState = SpeechState.SILENCE
    speech_frames: int = 0
    silence_frames: int = 0
    
    # Timing
    speech_start_time: float = 0.0
    last_partial_time: float = 0.0
    pause_start_time: float = 0.0
    
    # Контекст для адаптивных пауз
    last_transcript: str = ""
    conversation_context: List[str] = field(default_factory=list)
    
    # Speaker tracking
    speaker_sessions: Dict[str, int] = field(default_factory=dict)
    speaker_counter: int = 0
    
    # Метрики
    total_speech_ms: float = 0.0
    total_segments: int = 0
    
    # ДЕДУПЛИКАЦИЯ - предотвращает повторную отправку одинакового текста
    last_sent_text: str = ""
    last_sent_time: float = 0.0


# Глобальное хранилище сессий
sessions: Dict[str, ClientSession] = {}
sessions_lock = threading.Lock()


# ===============================
# УТИЛИТЫ
# ===============================

def calculate_energy(audio: np.ndarray) -> float:
    """Вычисляет RMS энергию аудио"""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def calculate_zero_crossings(audio: np.ndarray) -> int:
    """Подсчёт пересечений нуля"""
    if len(audio) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(np.sign(audio))) > 0))


def is_speech_frame(audio: np.ndarray) -> bool:
    """Определяет, является ли фрейм речью"""
    if len(audio) == 0:
        return False
    
    energy = calculate_energy(audio)
    if energy < VADConfig.ENERGY_THRESHOLD:
        return False
    
    # Шум имеет много пересечений нуля, речь - меньше
    zc_rate = calculate_zero_crossings(audio) / len(audio) if len(audio) > 0 else 0
    if zc_rate > 0.5:
        return False
    
    return True


def determine_pause_duration(text: str, speech_duration_ms: float) -> int:
    """
    ИНТЕЛЛЕКТУАЛЬНОЕ определение паузы - как у ChatGPT Voice.
    Анализирует контекст и структуру предложения для быстрого отклика.
    """
    text_lower = text.lower().strip()
    words = text_lower.split()
    word_count = len(words)
    
    # 1. Очень короткие ответы (1-2 слова) - минимальная пауза
    if word_count <= 2:
        return VADConfig.MIN_PAUSE_MS
    
    # 2. Команды с wake word - быстрая реакция
    has_kiko = any(w in text_lower for w in ['kiko', 'кико'])
    if has_kiko and word_count <= 5:
        return VADConfig.MIN_PAUSE_MS
    
    # 3. Явно завершённые предложения (точка, !, ?)
    if re.search(r'[.!?]$', text_lower):
        return VADConfig.MIN_PAUSE_MS + 50  # Немного больше для уверенности
    
    # 4. Короткие ответы - быстро
    for pattern in SHORT_RESPONSE_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return VADConfig.MIN_PAUSE_MS
    
    # 5. Команды - быстро
    for pattern in COMMAND_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return VADConfig.MIN_PAUSE_MS + 50
    
    # 6. Вопросы без знака вопроса в конце
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return VADConfig.QUESTION_PAUSE_MS
    
    # 7. Незавершённые предложения - ждём дольше
    # Признаки незавершённости:
    #   - заканчивается на союз/предлог
    #   - заканчивается на запятую
    #   - последнее слово < 3 букв и не существительное
    incomplete_endings = ['и', 'а', 'но', 'или', 'что', 'как', 'где', 'когда', 
                         'and', 'or', 'but', 'that', 'which', 'who', 'where',
                         'the', 'a', 'an', 'to', 'for', 'with', 'in', 'on']
    
    if words and words[-1] in incomplete_endings:
        return VADConfig.MAX_PAUSE_MS
    
    if text_lower.endswith(','):
        return VADConfig.MAX_PAUSE_MS
    
    # 8. По длине речи и количеству слов
    if word_count <= 4:
        return VADConfig.MIN_PAUSE_MS + 100
    elif word_count <= 8:
        return VADConfig.DEFAULT_PAUSE_MS
    else:
        # Длинные фразы - проверяем структуру
        # Если есть знак препинания в конце - готово
        if re.search(r'[.!?;]$', text_lower):
            return VADConfig.DEFAULT_PAUSE_MS
        return VADConfig.MAX_PAUSE_MS


# Фонетические варианты Kiko для fuzzy matching
KIKO_PHONETIC_VARIANTS = [
    "kiko", "kyko", "keeko", "kico", "kieko", "keyko", "tico", "tiko",
    "keco", "cico", "qico", "kika", "kikko", "keko", "chico", "kiku",
    "keeco", "kigo", "kijo", "kito", "kikou", "kicco", "kyco", "keiko",
    "quico", "keekou", "kikov", "key co", "key go", "ki ko", "ki go",
    "kee ko", "kee go", "key cool", "kekko", "geko", "gecko",
]


def fuzzy_match_kiko(word: str) -> bool:
    """Проверяет, похоже ли слово на 'Kiko' (fuzzy matching)"""
    clean = re.sub(r'[^\w]', '', word).lower()
    
    # Прямое совпадение
    if clean in CORRECTION_DICT:
        return True
    
    # Фонетические варианты
    if clean in KIKO_PHONETIC_VARIANTS:
        return True
    
    # Расстояние Левенштейна (простая версия)
    if len(clean) >= 3 and len(clean) <= 6:
        for variant in ["kiko", "keko", "kico", "kiku"]:
            diff = sum(1 for a, b in zip(clean, variant) if a != b)
            diff += abs(len(clean) - len(variant))
            if diff <= 1:  # Максимум 1 ошибка
                return True
    
    return False


def apply_post_correction(text: str) -> str:
    """
    Применяем пост-коррекцию текста с АГРЕССИВНЫМ поиском Kiko.
    Ищем похожие слова и заменяем на Kiko.
    """
    if not text:
        return text
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        punctuation_after = re.sub(r'^[\w\s]+', '', word)
        punctuation_before = re.sub(r'[\w\s]+$', '', word)
        
        corrected = None
        
        # 1. Прямое совпадение в словаре
        if clean_word in CORRECTION_DICT:
            corrected = CORRECTION_DICT[clean_word]
        # 2. Fuzzy matching для "похожих на Kiko" слов
        elif fuzzy_match_kiko(word):
            corrected = "Kiko"
        # 3. Близкие совпадения через difflib
        elif len(clean_word) > 2:
            matches = get_close_matches(clean_word, CORRECTION_DICT.keys(), n=1, cutoff=0.70)  # Снижен порог
            if matches:
                corrected = CORRECTION_DICT[matches[0]]
        
        if corrected:
            final_word = punctuation_before + corrected + punctuation_after
            corrected_words.append(final_word)
        else:
            corrected_words.append(word)
    
    result = ' '.join(corrected_words)
    
    # Убираем дубликаты Kiko рядом: "Kiko Kiko включи" -> "Kiko, включи"
    result = re.sub(r'\bKiko\s+Kiko\b', 'Kiko,', result, flags=re.IGNORECASE)
    
    # Убираем "Kiko assistant" галлюцинации
    result = re.sub(r'\bKiko\s+assistant\b', 'Kiko', result, flags=re.IGNORECASE)
    
    return result


def get_speaker_hash(audio_data: np.ndarray) -> str:
    """Улучшенный идентификатор спикера"""
    if len(audio_data) == 0:
        return "unknown"
    
    mean_amplitude = np.mean(np.abs(audio_data))
    std_amplitude = np.std(audio_data)
    zero_crossings = calculate_zero_crossings(audio_data)
    
    fft = np.fft.rfft(audio_data)
    magnitude = np.abs(fft)
    
    spectral_centroid = 0
    if np.sum(magnitude) > 0:
        spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
    
    n = len(magnitude)
    low_freq = np.sum(magnitude[:n//4]) if n >= 4 else 0
    high_freq = np.sum(magnitude[3*n//4:]) if n >= 4 else 0
    
    return f"{mean_amplitude:.5f}_{std_amplitude:.5f}_{zero_crossings}_{spectral_centroid:.2f}_{low_freq:.2f}_{high_freq:.2f}"


# Паттерны галлюцинаций Whisper (текст из промпта или повторения)
HALLUCINATION_PATTERNS = [
    r'^kiko[\s,\.]*kiko[\s,\.]*kiko',  # Повторяющееся Kiko
    r'^(kiko[\s,\.]*){{3,}}',  # Kiko 3+ раз подряд
    r'^кико[\s,\.]*кико[\s,\.]*кико',  # То же на русском
    r'voice assistant',  # ГЛАВНЫЙ источник галлюцинаций!
    r'kiko assistant',   # Частая галлюцинация
    r'kiko is a',        # Галлюцинация из промпта
    r'assistant kiko',   # Ещё вариант
    r'common phrases',   # Из промпта  
    r'having a conversation',  # Из промпта
    r'^\s*\.+\s*$',     # Только точки
    r'^\s*,+\s*$',      # Только запятые
    r'thank you for watching',  # Типичная галлюцинация YouTube
    r'thanks for watching',
    r'subscribe',
    r'like and subscribe',
    r'please subscribe',
    r'^kiko\.?$',        # Просто "Kiko" без контекста (часто галлюцинация)
    r'^kiko,?\.?$',     # "Kiko," или "Kiko."
    r'\bthe\s+kiko\b',  # "the Kiko" - неестественно
]

def is_noise_or_garbage(text: str) -> bool:
    """Определяет, является ли текст шумом или мусором"""
    if not text:
        return True
    
    t = text.strip()
    if len(t) < 2:
        return True
    if re.match(r'^\[[^\]]+\]$', t):  # [BLANK_AUDIO], [MUSIC], etc.
        return True
    if re.match(r'^[\s\.,!?\-\—\–\'\"…]+$', t):
        return True
    if re.match(r'^(.)\1{2,}$', t):
        return True
    
    return False


def is_hallucination(text: str) -> bool:
    """Проверяет, является ли текст галлюцинацией Whisper"""
    if not text:
        return False
    
    t = text.lower().strip()
    
    # Проверка паттернов галлюцинаций
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return True
    
    # Проверка на повторяющиеся слова ("kiko kiko kiko" или "the the the")
    words = t.split()
    if len(words) >= 3:
        # Если одно слово повторяется 3+ раза подряд
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
    
    # Проверка на слишком много Kiko (больше 2 в коротком тексте)
    kiko_count = len(re.findall(r'\bkiko\b', t, re.IGNORECASE))
    word_count = len(words)
    if word_count > 0 and kiko_count > 2 and kiko_count / word_count > 0.5:
        return True
    
    return False


def has_sufficient_audio_energy(audio: np.ndarray) -> bool:
    """Проверяет, есть ли в аудио достаточно энергии для речи"""
    if len(audio) == 0:
        return False
    
    energy = calculate_energy(audio)
    
    # Если средняя энергия слишком низкая - это тишина
    if energy < VADConfig.MIN_AUDIO_ENERGY:
        return False
    
    # Проверяем, что есть хотя бы какие-то "пики" (речь имеет динамику)
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude < 0.05:  # Если макс амплитуда < 5% - это тишина/шум
        return False
    
    return True


# ===============================
# ГЛАВНАЯ ЛОГИКА ОБРАБОТКИ
# ===============================

async def transcribe_audio(audio: np.ndarray, session: ClientSession) -> Tuple[str, dict]:
    """Транскрибация аудио с метриками и защитой от галлюцинаций."""
    audio_duration = len(audio) / SAMPLE_RATE
    
    # ЗАЩИТА ОТ ГАЛЛЮЦИНАЦИЙ: проверяем энергию аудио перед транскрибацией
    if not has_sufficient_audio_energy(audio):
        print(f"⚠️ [{session.client_id}] Audio energy too low, skipping transcription")
        return "", {"transcription_time_ms": 0, "audio_duration_s": round(audio_duration, 3), 
                   "realtime_factor": 0, "samples": len(audio), "skipped": "low_energy"}
    
    # Noise gate
    audio = audio * (np.abs(audio) > 0.008)  # Увеличен порог
    
    start_time = time.perf_counter()
    
    # Контекст - МИНИМАЛЬНЫЙ! "assistant" вызывает галлюцинации
    # Используем только предыдущие реплики БЕЗ слова "assistant"
    context_prompt = None  # По умолчанию без промпта
    if session.conversation_context:
        recent = session.conversation_context[-2:]  # Только 2 последние фразы
        # Фильтруем слово "assistant" из контекста
        clean_context = ' '.join(recent).replace('assistant', '').replace('Assistant', '')
        if clean_context.strip():
            context_prompt = clean_context.strip()
    
    result = whisper_model.transcribe(
        audio,
        language="en",
        initial_prompt=context_prompt,
        fp16=True,
        condition_on_previous_text=False,  # ОТКЛЮЧЕНО для предотвращения галлюцинаций
        no_speech_threshold=0.6,  # Увеличен порог "нет речи"
        logprob_threshold=-0.8,   # Более строгий порог вероятности
    )
    
    text = result["text"].strip()
    text = apply_post_correction(text)
    
    end_time = time.perf_counter()
    transcription_time = (end_time - start_time) * 1000
    rtf = audio_duration / (transcription_time / 1000) if transcription_time > 0 else 0
    
    # ПРОВЕРКА НА ГАЛЛЮЦИНАЦИИ
    if is_hallucination(text):
        print(f"🚫 [{session.client_id}] Hallucination filtered: {text!r}")
        return "", {"transcription_time_ms": round(transcription_time, 2), 
                   "audio_duration_s": round(audio_duration, 3),
                   "realtime_factor": round(rtf, 2), "samples": len(audio), 
                   "filtered": "hallucination", "original_text": text}
    
    metrics = {
        "transcription_time_ms": round(transcription_time, 2),
        "audio_duration_s": round(audio_duration, 3),
        "realtime_factor": round(rtf, 2),
        "samples": len(audio),
    }
    
    return text, metrics


async def process_vad_frame(session: ClientSession, frame: np.ndarray, websocket) -> Optional[dict]:
    """Обрабатывает один VAD фрейм. Возвращает результат для отправки или None."""
    is_speech = is_speech_frame(frame)
    current_time = time.time()
    
    result = None
    
    if is_speech:
        session.speech_frames += 1
        session.silence_frames = 0
        
        if session.state == SpeechState.SILENCE:
            if session.speech_frames >= VADConfig.SPEECH_START_FRAMES:
                session.state = SpeechState.SPEECH
                session.speech_start_time = current_time
                session.speech_buffer = list(session.audio_buffer[-5:])  # Preroll
                print(f"🎤 [{session.client_id}] Speech started")
        
        elif session.state == SpeechState.PAUSE:
            session.state = SpeechState.SPEECH
            print(f"🎤 [{session.client_id}] Speech resumed")
        
        if session.state == SpeechState.SPEECH:
            session.speech_buffer.append(frame)
            
            # Отправляем partial результаты
            if current_time - session.last_partial_time > VADConfig.PARTIAL_INTERVAL_MS / 1000:
                session.last_partial_time = current_time
                
                if len(session.speech_buffer) > 0:
                    audio = np.concatenate(session.speech_buffer)
                    # Уменьшено до 200мс для более быстрых partial
                    if len(audio) > SAMPLE_RATE * 0.2:
                        text, _ = await transcribe_audio(audio, session)
                        if text and not is_noise_or_garbage(text):
                            session.last_transcript = text
                            result = {
                                "type": "partial",
                                "text": text,
                                "is_final": False,
                                "timestamp": datetime.now().isoformat(),
                            }
    else:
        session.silence_frames += 1
        session.speech_frames = max(0, session.speech_frames - 1)
        
        if session.state == SpeechState.SPEECH:
            session.speech_buffer.append(frame)
            
            if session.silence_frames >= 2:  # ~40мс тишины -> переход в паузу (быстрее!)
                session.state = SpeechState.PAUSE
                session.pause_start_time = current_time
        
        elif session.state == SpeechState.PAUSE:
            session.speech_buffer.append(frame)
            
            speech_duration_ms = (current_time - session.speech_start_time) * 1000
            pause_duration_ms = session.silence_frames * VADConfig.FRAME_MS
            
            # Определяем необходимую паузу на основе текста
            required_pause = VADConfig.DEFAULT_PAUSE_MS
            if session.last_transcript:
                required_pause = determine_pause_duration(session.last_transcript, speech_duration_ms)
            
            # Финализируем если пауза достаточная
            if pause_duration_ms >= required_pause:
                result = await finalize_segment(session)
            
            # CONTINUOUS MODE: если сегмент длинный и есть короткая пауза - разбиваем
            # Это позволяет отправлять частями длинные монологи без прерывания
            elif speech_duration_ms > VADConfig.SOFT_SEGMENT_MS and pause_duration_ms >= 400:
                print(f"📤 [{session.client_id}] Soft split at {speech_duration_ms:.0f}ms (continuous mode)")
                result = await finalize_segment(session, continue_listening=True)
    
    # Ограничиваем размер буфера - принудительное разбиение при превышении MAX - принудительное разбиение при превышении MAX
    max_samples = int(VADConfig.MAX_SEGMENT_MS * SAMPLE_RATE / 1000)
    if session.state in (SpeechState.SPEECH, SpeechState.PAUSE):
        if sum(len(b) for b in session.speech_buffer) > max_samples:
            print(f"📤 [{session.client_id}] Hard split at {VADConfig.MAX_SEGMENT_MS}ms (continuous mode)")
            result = await finalize_segment(session, continue_listening=True)
    
    # Сохраняем фрейм для preroll
    session.audio_buffer.append(frame)
    if len(session.audio_buffer) > 10:
        session.audio_buffer.pop(0)
    
    return result


async def finalize_segment(session: ClientSession, continue_listening: bool = False) -> Optional[dict]:
    """Финализирует текущий сегмент речи
    
    Args:
        session: Клиентская сессия
        continue_listening: Если True - продолжаем слушать после финализации (continuous mode)
    """
    if not session.speech_buffer:
        return None
    
    audio = np.concatenate(session.speech_buffer)
    duration_ms = len(audio) / SAMPLE_RATE * 1000
    
    # Сброс состояния
    session.speech_buffer = []
    session.speech_frames = 0
    session.silence_frames = 0
    
    # CONTINUOUS MODE: остаёмся в режиме прослушивания если нужно
    if continue_listening:
        session.state = SpeechState.SPEECH
        session.speech_start_time = time.time()
        print(f"🔄 [{session.client_id}] Continuing to listen after segment...")
    else:
        session.state = SpeechState.SILENCE
    
    if duration_ms < VADConfig.MIN_SPEECH_MS:
        print(f"⏭️ [{session.client_id}] Segment too short ({duration_ms:.0f}ms), skipping")
        return None
    
    # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА энергии перед транскрибацией
    if not has_sufficient_audio_energy(audio):
        print(f"⚠️ [{session.client_id}] Audio energy too low ({duration_ms:.0f}ms), skipping")
        return None
    
    # Спикер
    speaker_hash = get_speaker_hash(audio)
    if speaker_hash not in session.speaker_sessions:
        session.speaker_counter += 1
        session.speaker_sessions[speaker_hash] = session.speaker_counter
    speaker_num = session.speaker_sessions[speaker_hash]
    
    # Транскрибация
    text, metrics = await transcribe_audio(audio, session)
    
    # Проверка на пустой результат или мусор
    if not text or is_noise_or_garbage(text):
        print(f"🗑️ [{session.client_id}] Empty or garbage filtered: {text!r}")
        return None
    
    # Дополнительная проверка на галлюцинации после транскрибации
    if is_hallucination(text):
        print(f"🚫 [{session.client_id}] Hallucination in final: {text!r}")
        return None
    
    # ДЕДУПЛИКАЦИЯ: проверяем не отправляли ли мы этот текст недавно
    current_time = time.time()
    text_normalized = text.lower().strip()
    last_normalized = session.last_sent_text.lower().strip() if session.last_sent_text else ""
    
    # Проверка на полное совпадение или очень похожий текст
    if last_normalized and text_normalized:
        time_since_last = (current_time - session.last_sent_time) * 1000
        
        # Если текст идентичен и прошло меньше DEDUP_WINDOW_MS
        if text_normalized == last_normalized and time_since_last < VADConfig.DEDUP_WINDOW_MS:
            print(f"🔁 [{session.client_id}] Duplicate skipped: {text!r} (sent {time_since_last:.0f}ms ago)")
            return None
        
        # Если текст очень похож (один содержит другой) и прошло меньше времени
        if time_since_last < VADConfig.DEDUP_WINDOW_MS:
            if text_normalized in last_normalized or last_normalized in text_normalized:
                # Если новый текст короче или равен - дубликат
                if len(text_normalized) <= len(last_normalized):
                    print(f"🔁 [{session.client_id}] Partial duplicate skipped: {text!r}")
                    return None
    
    # Обновляем последний отправленный текст
    session.last_sent_text = text
    session.last_sent_time = current_time
    
    # Обновляем контекст
    session.last_transcript = text
    session.conversation_context.append(text)
    if len(session.conversation_context) > 10:
        session.conversation_context.pop(0)
    
    session.total_speech_ms += duration_ms
    session.total_segments += 1
    
    print(f"📝 [{session.client_id}] Speaker #{speaker_num}: {text!r}")
    print(f"⏱️  Duration: {duration_ms:.0f}ms | Transcription: {metrics['transcription_time_ms']:.0f}ms | RTF: {metrics['realtime_factor']:.1f}x")
    
    return {
        "type": "transcription",
        "text": text,
        "is_final": True,
        "timestamp": datetime.now().isoformat(),
        "speaker_number": speaker_num,
        "metrics": metrics,
    }


# ===============================
# WS HANDLER
# ===============================

async def handle_client(websocket):
    """Обработка подключения клиента"""
    client_id = str(id(websocket))
    
    # Создаём чистую сессию (защита от проблем при перезагрузке)
    session = ClientSession(client_id=client_id)
    with sessions_lock:
        # Удаляем старую сессию если была (при быстром переподключении)
        if client_id in sessions:
            print(f"♻️ [{client_id}] Cleaning up previous session")
            del sessions[client_id]
        sessions[client_id] = session
    
    print(f"🔌 Клиент подключился: {client_id}")
    
    frame_samples = int(SAMPLE_RATE * VADConfig.FRAME_MS / 1000)
    pcm_buffer = np.array([], dtype=np.float32)
    
    # Счётчик тишины для предотвращения галлюцинаций при перезагрузке
    silence_streak = 0
    MAX_SILENCE_BEFORE_SKIP = 50  # ~1.5 сек тишины подряд = скипаем обработку
    
    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "ChatGPT-style STT server ready (v2.1 anti-hallucination)",
            "sample_rate": SAMPLE_RATE,
            "model": "whisper-small",
            "device": device,
            "features": [
                "adaptive_pause_detection",
                "streaming_partials",
                "speaker_identification",
                "hallucination_filter",
            ],
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
                    
                    # Проверяем энергию чанка для отслеживания тишины
                    chunk_energy = calculate_energy(audio_chunk)
                    if chunk_energy < VADConfig.ENERGY_THRESHOLD:
                        silence_streak += 1
                    else:
                        silence_streak = 0
                    
                    # Если слишком долго тишина - сбрасываем буферы для предотвращения галлюцинаций
                    if silence_streak > MAX_SILENCE_BEFORE_SKIP:
                        if session.state != SpeechState.SILENCE:
                            print(f"🔇 [{client_id}] Long silence detected, resetting buffers")
                            session.speech_buffer = []
                            session.state = SpeechState.SILENCE
                            session.speech_frames = 0
                            session.silence_frames = 0
                        pcm_buffer = np.array([], dtype=np.float32)
                        continue
                    
                    pcm_buffer = np.concatenate([pcm_buffer, audio_chunk])
                    
                    # Обрабатываем фреймы
                    while len(pcm_buffer) >= frame_samples:
                        frame = pcm_buffer[:frame_samples]
                        pcm_buffer = pcm_buffer[frame_samples:]
                        
                        result = await process_vad_frame(session, frame, websocket)
                        if result:
                            await websocket.send(json.dumps(result))
                
                elif msg_type == "finalize":
                    # Принудительная финализация
                    if session.state != SpeechState.SILENCE:
                        result = await finalize_segment(session)
                        if result:
                            await websocket.send(json.dumps(result))
                        else:
                            await websocket.send(json.dumps({
                                "type": "transcription",
                                "text": "",
                                "is_final": True,
                                "timestamp": datetime.now().isoformat(),
                            }))
                    else:
                        if len(pcm_buffer) > frame_samples:
                            session.speech_buffer = [pcm_buffer]
                            result = await finalize_segment(session)
                            pcm_buffer = np.array([], dtype=np.float32)
                            if result:
                                await websocket.send(json.dumps(result))
                            else:
                                await websocket.send(json.dumps({
                                    "type": "transcription",
                                    "text": "",
                                    "is_final": True,
                                    "timestamp": datetime.now().isoformat(),
                                }))
                        else:
                            await websocket.send(json.dumps({
                                "type": "transcription",
                                "text": "",
                                "is_final": True,
                                "timestamp": datetime.now().isoformat(),
                            }))
                
                elif msg_type == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat(),
                    }))
                
                elif msg_type == "reset":
                    session.conversation_context = []
                    session.last_transcript = ""
                    await websocket.send(json.dumps({
                        "type": "reset_ack",
                        "timestamp": datetime.now().isoformat(),
                    }))
                
                elif msg_type == "config":
                    # Динамическая настройка параметров
                    if "pause_ms" in data:
                        VADConfig.DEFAULT_PAUSE_MS = int(data["pause_ms"])
                    if "energy_threshold" in data:
                        VADConfig.ENERGY_THRESHOLD = float(data["energy_threshold"])
                    await websocket.send(json.dumps({
                        "type": "config_ack",
                        "config": {
                            "pause_ms": VADConfig.DEFAULT_PAUSE_MS,
                            "energy_threshold": VADConfig.ENERGY_THRESHOLD,
                        },
                    }))
            
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON",
                }))
            except Exception as e:
                print(f"❌ Ошибка обработки: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))
    
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Клиент отключился: {client_id}")
    except Exception as e:
        print(f"❌ Ошибка в handle_client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"👋 Сессия завершена: {client_id}")
        print(f"   📊 Всего сегментов: {session.total_segments}")
        print(f"   ⏱️  Общее время речи: {session.total_speech_ms/1000:.1f}с")
        print(f"   🎭 Спикеров: {session.speaker_counter}")
        
        with sessions_lock:
            if client_id in sessions:
                del sessions[client_id]


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
