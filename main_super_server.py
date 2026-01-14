"""
WebSocket STT сервер v2.6 - Wake word "Optimus"
OpenAI Whisper Medium на GPU + Picovoice Porcupine Wake Word

Улучшения v2.6:
- Замена wake word "Kiko" на "Optimus" во всём коде
- Обновлены CORRECTION_DICT, PHONETIC_VARIANTS, HOTWORDS
- Обновлены функции fuzzy_match, check_first_word, clean_duplicate
"""
import warnings
warnings.filterwarnings("ignore")

import asyncio
import websockets
import websockets.exceptions
import json
import whisper
import torch
import numpy as np
from datetime import datetime
import base64
import time
from collections import defaultdict
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import gc
from concurrent.futures import ThreadPoolExecutor
import signal
import sys as _sys

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

# Загружаем Whisper Medium - более точная модель
print(f"📦 Загружаем Whisper Medium ({device.upper()})...")
start_time = time.time()
whisper_model = whisper.load_model("medium", device=device)
load_time = time.time() - start_time
print(f"✅ Whisper загружен за {load_time:.2f}с\n")

print("=" * 80)
print(f"🌐 WebSocket сервер готов на ws://0.0.0.0:8765")
print(f"📊 Режим: {device.upper()} | ChatGPT-style диалог")
print("=" * 80)
print()

# ThreadPoolExecutor для блокирующих операций (Whisper)
# Используем 1 воркер - модель НЕ thread-safe!
whisper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper_")

# Lock для сериализации доступа к Whisper модели (CUDA не thread-safe)
whisper_lock = threading.Lock()

# Таймаут для транскрибации (секунды)
TRANSCRIBE_TIMEOUT = 30.0

# Интервал очистки GPU памяти (секунды)
GPU_CLEANUP_INTERVAL = 60.0
last_gpu_cleanup = time.time()

# ===============================
# НАСТРОЙКИ - ОПТИМИЗИРОВАННЫЕ КАК У OPENAI AUDIO
# ===============================
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16

# === WHISPER ADVANCED SETTINGS - МАКСИМАЛЬНОЕ КАЧЕСТВО v2.2 ===
class WhisperConfig:
    # Beam search - УВЕЛИЧЕНО для лучшего качества
    BEAM_SIZE = 7      # 7 beams - выше качество для коротких фраз
    BEST_OF = 7        # Выбор лучшего из 7 кандидатов
    
    # Temperature - НИЗКАЯ для стабильности, агрессивный fallback
    TEMPERATURE = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)  # Больше шагов для fallback
    
    # Compression ratio - МЯГЧЕ для сохранения коротких фраз
    COMPRESSION_RATIO_THRESHOLD = 2.8
    
    # Log probability - ЕЩЁ МЯГЧЕ для улучшения распознавания
    LOGPROB_THRESHOLD = -1.5  # Мягче - принимаем менее уверенные слова
    
    # No speech threshold - МЯГЧЕ для тихой речи
    NO_SPEECH_THRESHOLD = 0.55  # Ниже - меньше пропусков тихой речи
    
    # Condition on previous - отключено для независимости
    CONDITION_ON_PREVIOUS = False
    
    # Word timestamps - ВКЛЮЧЕНО для точности и word-level confidence
    WORD_TIMESTAMPS = True  # Улучшает точность + даёт уверенность по словам
    
    # Punctuations - расширенный набор
    PREPEND_PUNCTUATIONS = "\"'¿([{-«"
    APPEND_PUNCTUATIONS = "\"'.;:?!,،、。」』】〗》）»\n"


# === VAD настройки - МАКСИМАЛЬНАЯ ЧУВСТВИТЕЛЬНОСТЬ v2.2 ===
class VADConfig:
    # Порог энергии для определения речи - ULTRA МЯГКИЙ
    ENERGY_THRESHOLD = 0.002  # Очень низкий для захвата шёпота
    
    # Минимальная энергия для транскрибации - ULTRA МЯГКИЙ
    MIN_AUDIO_ENERGY = 0.003  # Очень низкий для тихого микрофона
    
    # Адаптивные паузы - БЫСТРЕЕ для отзывчивости
    MIN_PAUSE_MS = 650        # 650мс минимум - быстрее реакция
    DEFAULT_PAUSE_MS = 1000   # 1000мс стандарт - быстрее
    MAX_PAUSE_MS = 1500       # 1500мс макс для длинных предложений
    QUESTION_PAUSE_MS = 750   # 750мс для вопросов - быстрее
    
    # Минимальная длительность речи - оптимальная для Whisper
    MIN_SPEECH_MS = 500       # 500мс - минимум для коротких слов
    
    # Максимальная длительность сегмента
    MAX_SEGMENT_MS = 30000    # 30 секунд
    
    # Порог для мягкой финализации
    SOFT_SEGMENT_MS = 20000   # 20 сек
    
    # Частота partial - баланс скорости/качества
    PARTIAL_INTERVAL_MS = 300  # 300мс - реже для лучшего качества
    
    # Размер VAD фрейма
    FRAME_MS = 20             # 20мс фреймы
    
    # Количество фреймов для начала речи - баланс скорости и стабильности
    SPEECH_START_FRAMES = 2   # 2 фрейма = 40мс - стабильнее
    
    # ДЕДУПЛИКАЦИЯ
    DEDUP_WINDOW_MS = 2500    # 2.5 секунды


# === Hotwords для boosting ===
HOTWORDS = ["Optimus", "optimus", "OPTIMUS", "оптимус", "Оптимус"]

# Количество preroll фреймов - МАКСИМУМ для захвата начала слов
PREROLL_FRAMES = 30  # 30 фреймов = 600мс preroll для идеального захвата начала

# Словарь для post-correction - РАСШИРЕННЫЙ v2.2
# Включает все фонетические варианты Optimus и частые ошибки Whisper
CORRECTION_DICT = {
    # Прямые фонетические варианты Optimus
    "optimus": "Optimus", "optimas": "Optimus", "optimis": "Optimus", "optimes": "Optimus",
    "optimus'": "Optimus", "optimus's": "Optimus", "optimous": "Optimus", "optimis": "Optimus",
    "optimus,": "Optimus,", "optimus.": "Optimus.", "optimus?": "Optimus?", "optimus!": "Optimus!",
    "optimuss": "Optimus", "optimuz": "Optimus", "optimuse": "Optimus", "optimust": "Optimus",
    "optumus": "Optimus", "optames": "Optimus", "optemos": "Optimus", "optimis": "Optimus",
    # Двухсловные варианты
    "hey optimus": "Optimus", "ok optimus": "Optimus", "okay optimus": "Optimus",
    "hi optimus": "Optimus", "oh optimus": "Optimus", "yo optimus": "Optimus",
    # Русские варианты
    "оптимус": "Optimus", "оптімус": "Optimus", "оптимас": "Optimus", "оптимос": "Optimus",
    "оптимус.": "Optimus", "оптимус,": "Optimus", "оптімус.": "Optimus",
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
    
    # Время создания сессии
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
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
sessions_lock = asyncio.Lock()  # asyncio Lock вместо threading Lock для async контекста

# Максимальное количество одновременных сессий (защита от перегрузки)
MAX_CONCURRENT_SESSIONS = 50

# Таймаут неактивной сессии (секунды) - сессии без активности удаляются
SESSION_IDLE_TIMEOUT = 120.0  # 2 минуты


async def cleanup_stale_sessions():
    """Удаляет зависшие сессии которые не отправляли данные долгое время"""
    current_time = time.time()
    stale_ids = []
    
    async with sessions_lock:
        for client_id, session in sessions.items():
            # Используем last_activity для отслеживания активности
            idle_time = current_time - session.last_activity
            
            if idle_time > SESSION_IDLE_TIMEOUT:
                stale_ids.append(client_id)
        
        # Удаляем зависшие сессии
        for client_id in stale_ids:
            print(f"🗑️ [{client_id}] Removing stale session (idle > {SESSION_IDLE_TIMEOUT}s)")
            del sessions[client_id]
    
    if stale_ids:
        print(f"🧹 Cleaned up {len(stale_ids)} stale sessions")


async def cleanup_gpu_memory(force: bool = False):
    """Периодическая очистка GPU памяти для предотвращения утечек"""
    global last_gpu_cleanup
    current_time = time.time()
    
    # Принудительная очистка или по таймеру
    if force or current_time - last_gpu_cleanup > GPU_CLEANUP_INTERVAL:
        try:
            # Сначала очищаем зависшие сессии
            await cleanup_stale_sessions()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            last_gpu_cleanup = current_time
            async with sessions_lock:
                active_count = len(sessions)
            print(f"🧹 GPU memory cleanup performed")
            print(f"📊 Активных сессий: {active_count}")
        except Exception as e:
            print(f"⚠️ GPU cleanup error: {e}")


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
    
    # 2. Явно завершённые предложения (точка, !, ?)
    if re.search(r'[.!?]$', text_lower):
        return VADConfig.MIN_PAUSE_MS + 50  # Немного больше для уверенности
    
    # 3. Короткие ответы - быстро
    for pattern in SHORT_RESPONSE_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return VADConfig.MIN_PAUSE_MS
    
    # 4. Команды - быстро
    for pattern in COMMAND_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return VADConfig.MIN_PAUSE_MS + 50
    
    # 5. Вопросы - нужна пауза побольше чтобы человек договорил
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return VADConfig.DEFAULT_PAUSE_MS  # 1000ms для вопросов
    
    # 6. Незавершённые предложения - ждём дольше
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


# Фонетические варианты Optimus - ТОЛЬКО явные варианты, без обычных слов
OPTIMUS_PHONETIC_VARIANTS = [
    # Основные варианты - звучат как "optimus"
    "optimus", "optimas", "optimis", "optimes", "optimous", "optimuz",
    "optumus", "optemos", "optimuss", "optimuse", "optimust",
    # Русские варианты
    "оптимус", "оптімус", "оптимас", "оптимос", "оптімас",
]


def fuzzy_match_optimus(word: str) -> bool:
    """Проверяет, похоже ли слово на 'Optimus' - СТРОГАЯ версия"""
    clean = re.sub(r'[^\w]', '', word).lower()
    
    # Пустое слово или слишком короткое/длинное
    if not clean or len(clean) < 6 or len(clean) > 10:
        return False
    
    # Прямое совпадение в словаре
    if clean in CORRECTION_DICT:
        return True
    
    # Фонетические варианты (строгий список)
    if clean in OPTIMUS_PHONETIC_VARIANTS:
        return True
    
    # Строгий паттерн: opt + im/em/am + us/is/os
    if re.match(r'^opt[iea]m[uio]s+[est]?$', clean):
        return True
    
    # Русский паттерн: опт + им/ім + ус/ас
    if re.match(r'^опт[иіе]м[уао]с$', clean):
        return True
    
    return False


def apply_post_correction(text: str) -> str:
    """
    Применяем пост-коррекцию текста - СТРОГАЯ версия.
    Заменяем только явные варианты Optimus, не трогаем обычные слова.
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
        
        # ТОЛЬКО прямое совпадение в словаре - никакого fuzzy matching!
        if clean_word in CORRECTION_DICT:
            corrected = CORRECTION_DICT[clean_word]
        # Строгий fuzzy match только для явных вариантов Optimus
        elif fuzzy_match_optimus(clean_word):
            corrected = "Optimus"
        
        if corrected:
            final_word = punctuation_before + corrected + punctuation_after
            corrected_words.append(final_word)
        else:
            corrected_words.append(word)
    
    result = ' '.join(corrected_words)
    
    # Убираем дубликаты Optimus рядом: "Optimus Optimus включи" -> "Optimus, включи"
    result = re.sub(r'\bOptimus\s+Optimus\b', 'Optimus,', result, flags=re.IGNORECASE)
    
    return result


def check_first_word_is_optimus(text: str) -> str:
    """
    Проверяет первое слово на похожесть с Optimus и исправляет если нужно.
    СТРОГАЯ версия - только явные варианты Optimus, не трогаем обычные слова.
    """
    if not text or len(text) < 2:
        return text
    
    words = text.split()
    if not words:
        return text
    
    first_word = words[0].lower().strip('.,!?')
    
    # РАСШИРЕННЫЙ список фонетических вариантов Optimus
    optimus_like_starts = [
        # Прямые варианты звучащие как "optimus"
        "optimus", "optimas", "optimis", "optimes", "optimous", "optimuz",
        "optumus", "optemos", "optimuss", "optimuse", "optimust",
        # Русские варианты
        "оптимус", "оптімус", "оптимас", "оптимос", "оптимус,", "оптимус.",
    ]
    
    # Проверяем первое слово
    if first_word in optimus_like_starts:
        words[0] = "Optimus"
        return ' '.join(words)
    
    # Проверяем ТОЛЬКО явные двухсловные комбинации с optimus
    if len(words) >= 2:
        two_words = f"{words[0]} {words[1]}".lower()
        optimus_like_two_words = [
            "hey optimus", "ok optimus", "okay optimus",
            "hi optimus", "oh optimus", "yo optimus",
            "эй оптимус", "хей оптимус", "о оптимус", "привет оптимус",
        ]
        if two_words in optimus_like_two_words:
            # Заменяем первые два слова на Optimus
            return "Optimus " + ' '.join(words[2:]) if len(words) > 2 else "Optimus"
    
    return text


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
    r'^optimus[\s,\.]*optimus[\s,\.]*optimus',  # Повторяющееся Optimus 3+ раз
    r'^(optimus[\s,\.]*){4,}',  # Optimus 4+ раз подряд
    r'^оптимус[\s,\.]*оптимус[\s,\.]*оптимус',  # То же на русском
    r'voice assistant',  # ГЛАВНЫЙ источник галлюцинаций!
    r'optimus assistant',   # Частая галлюцинация
    r'optimus is a',        # Галлюцинация из промпта
    r'assistant optimus',   # Ещё вариант
    r'common phrases',   # Из промпта  
    r'having a conversation',  # Из промпта
    r'^\s*\.+\s*$',     # Только точки
    r'^\s*,+\s*$',      # Только запятые
    r'thank you for watching',  # Типичная галлюцинация YouTube
    r'thanks for watching',
    r'subscribe',
    r'like and subscribe',
    r'please subscribe',
    # УБРАНО: r'^optimus\.?$' - это валидный wake word!
    r'\bthe\s+optimus\b',  # "the Optimus" - неестественно
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
    
    # Фильтр мусорных звуков: Grrrr, hmmm, aaah, etc.
    t_lower = t.lower().rstrip('!')
    # Повторяющиеся буквы: grrrr, hmmm, aaaa
    if re.match(r'^([a-z])\1{2,}$', t_lower):
        return True
    # Короткие междометия
    if t_lower in ['hmm', 'hm', 'uh', 'um', 'ah', 'oh', 'eh', 'mm', 'mhm', 'ugh', 'grr', 'grrr', 'grrrr', 'aah', 'ooh']:
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
    
    # Проверка на повторяющиеся слова ("optimus optimus optimus" или "the the the")
    words = t.split()
    if len(words) >= 3:
        # Если одно слово повторяется 3+ раза подряд
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
    
    # УБРАНО: одиночное "Optimus" - это валидный wake word!
    # Теперь НЕ фильтруем одиночное Optimus - это нормальная активация ассистента
    # non_optimus_words = [w for w in words if w.lower() != 'optimus']
    # if len(words) > 0 and len(non_optimus_words) == 0:
    #     return True
    
    return False


def clean_duplicate_optimus(text: str) -> str:
    """Удаляет повторяющиеся 'optimus', оставляя только первый.
    Также убирает пунктуацию рядом с удалёнными optimus.
    """
    if not text:
        return text
    
    # Считаем сколько optimus в тексте (включая с пунктуацией рядом)
    optimus_matches = list(re.finditer(r'\boptimus\b', text, re.IGNORECASE))
    if len(optimus_matches) <= 1:
        return text
    
    # Удаляем все optimus кроме первого, вместе с окружающей пунктуацией
    result = text
    # Идём с конца чтобы индексы не сбивались
    for match in reversed(optimus_matches[1:]):
        start, end = match.start(), match.end()
        
        # Расширяем диапазон удаления на пунктуацию и пробелы вокруг
        while start > 0 and result[start-1] in ' ,.:;!?':
            start -= 1
        while end < len(result) and result[end] in ' ,.:;!?':
            end += 1
            
        result = result[:start] + ' ' + result[end:]
    
    # Убираем двойные пробелы и лишнюю пунктуацию
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s*,\s*,+', ',', result)  # ,, -> ,
    result = re.sub(r'\s*\.\s*\.+', '.', result)  # .. -> .
    result = re.sub(r',\s*\.', '.', result)  # ,. -> .
    result = re.sub(r'\.\s*,', '.', result)  # ., -> .
    
    return result.strip()


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

async def transcribe_audio(audio: np.ndarray, session: ClientSession, is_partial: bool = False) -> Tuple[str, dict]:
    """Транскрибация аудио с метриками и защитой от галлюцинаций.
    Использует OpenAI-style параметры для лучшего качества.
    is_partial=True подавляет verbose логирование.
    """
    audio_duration = len(audio) / SAMPLE_RATE
    
    # ЗАЩИТА ОТ ГАЛЛЮЦИНАЦИЙ: проверяем энергию аудио перед транскрибацией
    if not has_sufficient_audio_energy(audio):
        if not is_partial:  # Не спамим для partial
            print(f"⚠️ [{session.client_id}] Audio energy too low, skipping transcription")
        return "", {"transcription_time_ms": 0, "audio_duration_s": round(audio_duration, 3), 
                   "realtime_factor": 0, "samples": len(audio), "skipped": "low_energy"}
    
    # === УЛУЧШЕННАЯ ПРЕДОБРАБОТКА АУДИО v2.3 ===
    
    # ВАЖНО: Whisper требует float32, убеждаемся в правильном типе
    audio = audio.astype(np.float32)
    
    # 1. Убираем DC offset (постоянную составляющую)
    audio = audio - np.mean(audio, dtype=np.float32)
    
    # 2. Мягкий high-pass фильтр для удаления низкочастотного гула (< 80 Hz)
    # Простой single-pole filter: y[n] = x[n] - x[n-1] + 0.97 * y[n-1]
    alpha = np.float32(0.97)
    filtered = np.zeros_like(audio, dtype=np.float32)
    for i in range(1, len(audio)):
        filtered[i] = audio[i] - audio[i-1] + alpha * filtered[i-1]
    audio = filtered
    
    # 3. Улучшенная нормализация громкости (peak + RMS hybrid)
    max_val = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    if max_val > 0.01:
        # Нормализуем по пику, но учитываем RMS для контроля динамики
        target_rms = np.float32(0.15)  # Целевой RMS уровень
        if rms > 0.001:
            # Ограничиваем усиление чтобы не поднять шум
            gain = np.float32(min(target_rms / rms, 0.95 / max_val, 3.0))
            audio = audio * gain
        else:
            audio = audio / max_val * np.float32(0.95)
    
    # 4. Мягкое ограничение пиков (soft clipping) для предотвращения клиппинга
    audio = np.tanh(audio * np.float32(1.2)) / np.float32(np.tanh(1.2))
    
    # Финальная проверка типа - ОБЯЗАТЕЛЬНО float32 для Whisper!
    audio = audio.astype(np.float32)
    
    # ПРОМПТ ОТКЛЮЧЁН - вызывал галлюцинации и ухудшал распознавание
    # context_prompt = None
    
    start_time = time.perf_counter()
    
    # Синхронная функция для выполнения в executor - МАКСИМАЛЬНОЕ КАЧЕСТВО
    # ВАЖНО: используем lock для сериализации доступа к GPU модели
    def _transcribe_sync():
        # Защита от слишком короткого аудио (< 0.6 сек) - оптимизировано
        if len(audio) < SAMPLE_RATE * 0.6:
            return {"text": "", "segments": []}
        
        # Lock для предотвращения конкурентного доступа к модели
        # Это решает ошибки "Key and Value must have the same sequence length"
        with whisper_lock:
            try:
                return whisper_model.transcribe(
                    audio,
                    language="en",  # Фиксированный язык для стабильности
                    task="transcribe",
                    # initial_prompt ОТКЛЮЧЁН
                    fp16=True,
                    
                    # Beam search для качества
                    beam_size=WhisperConfig.BEAM_SIZE,
                    best_of=WhisperConfig.BEST_OF,
                    
                    # Temperature - низкая для стабильности
                    temperature=WhisperConfig.TEMPERATURE,
                    
                    # Фильтры качества
                    compression_ratio_threshold=WhisperConfig.COMPRESSION_RATIO_THRESHOLD,
                    logprob_threshold=WhisperConfig.LOGPROB_THRESHOLD,
                    no_speech_threshold=WhisperConfig.NO_SPEECH_THRESHOLD,
                    
                    # Независимые сегменты
                    condition_on_previous_text=WhisperConfig.CONDITION_ON_PREVIOUS,
                    
                    # Word timestamps для точности
                    word_timestamps=WhisperConfig.WORD_TIMESTAMPS,
                    
                    # Пунктуация
                    prepend_punctuations=WhisperConfig.PREPEND_PUNCTUATIONS,
                    append_punctuations=WhisperConfig.APPEND_PUNCTUATIONS,
                )
            except RuntimeError as e:
                # Ловим CUDA/PyTorch ошибки и возвращаем пустой результат
                error_msg = str(e)
                if "sequence length" in error_msg or "size" in error_msg or "shape" in error_msg:
                    print(f"⚠️ [{session.client_id}] CUDA tensor error (recovering): {error_msg[:80]}")
                    # Очищаем GPU кэш при ошибке
                    torch.cuda.empty_cache()
                    return {"text": "", "segments": []}
                raise  # Пробрасываем другие ошибки
    
    try:
        # Выполняем блокирующую транскрибацию в отдельном потоке с таймаутом
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(whisper_executor, _transcribe_sync),
            timeout=TRANSCRIBE_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"⚠️ [{session.client_id}] Transcription timeout after {TRANSCRIBE_TIMEOUT}s")
        return "", {"transcription_time_ms": TRANSCRIBE_TIMEOUT * 1000, 
                   "audio_duration_s": round(audio_duration, 3),
                   "error": "timeout"}
    except Exception as e:
        print(f"❌ [{session.client_id}] Transcription error: {e}")
        return "", {"transcription_time_ms": 0, 
                   "audio_duration_s": round(audio_duration, 3),
                   "error": str(e)}
    
    # Защита от None результата (может случиться при CUDA ошибках)
    if result is None:
        print(f"⚠️ [{session.client_id}] Transcription returned None")
        return "", {"transcription_time_ms": 0, 
                   "audio_duration_s": round(audio_duration, 3),
                   "error": "null_result"}
    
    text = result.get("text", "").strip() if isinstance(result, dict) else ""
    text = apply_post_correction(text)
    
    # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: если первое слово похоже на Optimus - исправляем
    original_first_word = text
    text = check_first_word_is_optimus(text)
    if text != original_first_word and not is_partial:
        print(f"🔧 [{session.client_id}] Fixed first word to Optimus: {original_first_word!r} -> {text!r}")
    
    # Очищаем дублирующиеся "optimus" (оставляем только первый)
    original_text = text
    text = clean_duplicate_optimus(text)
    if text != original_text and not is_partial:
        print(f"🔧 [{session.client_id}] Cleaned duplicate optimus: {original_text!r} -> {text!r}")
    
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
                # УВЕЛИЧЕННЫЙ Preroll для захвата начала фразы с Optimus
                session.speech_buffer = list(session.audio_buffer[-PREROLL_FRAMES:])
                print(f"🎤 [{session.client_id}] Speech started (preroll: {len(session.speech_buffer)} frames)")
        
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
                    # 400мс минимум для качественного partial
                    if len(audio) > SAMPLE_RATE * 0.4:
                        text, _ = await transcribe_audio(audio, session, is_partial=True)
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
    
    # Сохраняем фрейм для preroll - УВЕЛИЧЕННЫЙ буфер
    session.audio_buffer.append(frame)
    if len(session.audio_buffer) > PREROLL_FRAMES + 5:  # +5 запас
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
    
    # Проверка лимита сессий для защиты от перегрузки GPU
    async with sessions_lock:
        current_count = len(sessions)
        if current_count >= MAX_CONCURRENT_SESSIONS:
            print(f"⚠️ [{client_id}] Rejected: too many sessions ({current_count}/{MAX_CONCURRENT_SESSIONS})")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Server overloaded. Max {MAX_CONCURRENT_SESSIONS} concurrent sessions.",
                "code": "max_sessions_reached"
            }))
            await websocket.close()
            return
    
    # Создаём чистую сессию (защита от проблем при перезагрузке)
    session = ClientSession(client_id=client_id)
    async with sessions_lock:
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
    
    # Счётчик для периодической очистки
    message_counter = 0
    
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
                    
                    # Обновляем время последней активности сессии
                    session.last_activity = time.time()
                    
                    # Периодическая очистка GPU памяти (каждые 500 сообщений на сессию)
                    message_counter += 1
                    if message_counter % 500 == 0:
                        await cleanup_gpu_memory(force=True)
                    
                    try:
                        audio_chunk = np.frombuffer(
                            base64.b64decode(audio_b64),
                            dtype=np.int16
                        ).astype(np.float32) / 32768.0
                    except Exception as e:
                        print(f"⚠️ [{client_id}] Audio decode error: {e}")
                        continue
                    
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
                            try:
                                await websocket.send(json.dumps(result))
                            except websockets.exceptions.ConnectionClosed:
                                return  # Клиент отключился
                
                elif msg_type == "finalize":
                    # Принудительная финализация
                    if session.state != SpeechState.SILENCE:
                        result = await finalize_segment(session)
                        try:
                            if result:
                                await websocket.send(json.dumps(result))
                            else:
                                await websocket.send(json.dumps({
                                    "type": "transcription",
                                    "text": "",
                                    "is_final": True,
                                    "timestamp": datetime.now().isoformat(),
                                }))
                        except websockets.exceptions.ConnectionClosed:
                            return  # Клиент отключился
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
        
        # Очищаем буферы сессии
        session.audio_buffer.clear()
        session.speech_buffer.clear()
        session.conversation_context.clear()
        
        async with sessions_lock:
            if client_id in sessions:
                del sessions[client_id]
        
        # Очищаем GPU память после отключения клиента
        await cleanup_gpu_memory()


async def main():
    """Запуск WebSocket сервера с graceful shutdown"""
    
    # Обработка сигналов для graceful shutdown
    stop_event = asyncio.Event()
    
    def signal_handler():
        print("\n🛑 Получен сигнал завершения...")
        stop_event.set()
    
    # Регистрируем обработчики сигналов
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows не поддерживает add_signal_handler
            pass
    
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8765,
        ping_interval=20,
        ping_timeout=20,
        max_size=10 * 1024 * 1024,  # 10MB max message size
        close_timeout=10,  # Таймаут закрытия соединения
    )
    
    print("🎧 Ожидаю подключений...")
    
    # Фоновая задача для периодической очистки памяти
    async def periodic_cleanup():
        while not stop_event.is_set():
            await asyncio.sleep(GPU_CLEANUP_INTERVAL)
            await cleanup_gpu_memory()
            # Логируем состояние сессий
            async with sessions_lock:
                if sessions:
                    print(f"📊 Активных сессий: {len(sessions)}")
    
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        print("🔄 Завершаю сервер...")
        cleanup_task.cancel()
        server.close()
        await server.wait_closed()
        
        # Закрываем executor
        whisper_executor.shutdown(wait=True, cancel_futures=True)
        
        # Финальная очистка GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Сервер остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Сервер остановлен по Ctrl+C")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Финальная очистка
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Ресурсы освобождены")
