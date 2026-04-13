#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/listener.py - РАСПОЗНАВАНИЕ РЕЧИ (VOSK)                              ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Модуль для распознавания речи. Слушает микрофон, ждёт wake word «Гралл»,   ║
║   и передаёт распознанный текст агенту.                                      ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ:                                                            ║
║                                                                              ║
║   1. Vosk слушает микрофон и распознаёт речь ОФЛАЙН (без интернета).         ║
║   2. Если громкость выше порога (SOUND_THRESHOLD) — начинается запись.       ║
║   3. Когда наступает тишина — распознанный текст отправляется в колбэк.      ║
║                                                                              ║
║ РЕЖИМЫ РАБОТЫ:                                                               ║
║                                                                              ║
║   • ОБЫЧНЫЙ РЕЖИМ:                                                           ║
║     - Ждёт wake word («Гралл», «Робот», «Эй»).                               ║
║     - После активации слушает команды 15 секунд.                             ║
║     - Команда «стоп» работает БЕЗ wake word (аварийная остановка).           ║
║                                                                              ║
║   • ИНТЕРАКТИВНЫЙ РЕЖИМ:                                                     ║
║     - Слушает ВСЁ, что говорят вокруг.                                       ║
║     - Не требует wake word.                                                  ║
║     - Робот сам решает, когда ответить (через LLM).                          ║
║                                                                              ║
║ ПОЧЕМУ VOSK, А НЕ ОБЛАЧНЫЕ СЕРВИСЫ:                                          ║
║   1. Работает БЕЗ ИНТЕРНЕТА — робот автономен.                               ║
║   2. Мгновенный отклик (нет задержки на отправку в облако).                  ║
║   3. Приватность — голос не покидает робота.                                 ║
║   4. Wake word настраивается в коде.                                         ║
║                                                                              ║
║ ТРЕБОВАНИЯ:                                                                  ║
║   - Модель Vosk (скачать с https://alphacephei.com/vosk/models)             ║
║   - Для Termux: pkg install vosk-api python-vosk                              ║
║   - pyaudio для захвата звука                                                ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Wake words:                                                             ║
║      wake_words=["гралл", "робот", "эй"]  # добавьте свои                    ║
║                                                                              ║
║   2. Таймаут активности:                                                     ║
║      active_timeout=15.0  # сколько секунд слушать после активации           ║
║                                                                              ║
║   3. Порог громкости:                                                        ║
║      threshold=500  # ниже — не реагирует (меньше ложных срабатываний)       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import queue
import threading
import json
import numpy as np
import time
import logging
from typing import Optional, Callable

# Vosk для распознавания речи
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    Model = None
    KaldiRecognizer = None

# PyAudio для захвата микрофона
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class VoiceListener:
    """
    РАСПОЗНАВАНИЕ РЕЧИ С ПОДДЕРЖКОЙ WAKE WORD
    
    Слушает микрофон, ждёт активации, передаёт текст агенту.
    Работает полностью офлайн.
    """
    
    def __init__(self, 
                 model_path: str,
                 threshold: int = 500,
                 wake_words: list = None,
                 active_timeout: float = 15.0):
        """
        Args:
            model_path: путь к модели Vosk (папка с файлами)
            threshold: порог громкости для начала записи (0-32767)
            wake_words: список слов-активаторов
            active_timeout: сколько секунд слушать после активации
        """
        if not VOSK_AVAILABLE:
            raise RuntimeError("Vosk не установлен. pip install vosk")
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("PyAudio не установлен. pip install pyaudio")
        
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.threshold = threshold
        self.wake_words = wake_words or ["гралл", "робот", "эй"]
        self.active_timeout = active_timeout
        
        # Очередь для аудио-данных
        self.audio_queue = queue.Queue()
        
        # Состояние
        self.is_listening = False
        self.is_active = False
        self.active_until = 0
        self.interactive_mode = False
        
        # Колбэк для передачи распознанного текста
        self.callback: Optional[Callable] = None
        
        logger.info(f"✅ VoiceListener инициализирован (wake words: {self.wake_words})")
    
    def set_callback(self, callback: Callable):
        """
        УСТАНАВЛИВАЕТ ФУНКЦИЮ ДЛЯ ОБРАБОТКИ РАСПОЗНАННОГО ТЕКСТА
        
        Колбэк будет вызван с параметрами:
            text: str — распознанный текст
            wake: bool — была ли активация через wake word
            emergency_stop: bool — это аварийная остановка?
            interactive: bool — это интерактивный режим?
        """
        self.callback = callback
    
    def set_interactive_mode(self, enabled: bool):
        """
        ВКЛЮЧАЕТ/ВЫКЛЮЧАЕТ ИНТЕРАКТИВНЫЙ РЕЖИМ
        
        В интерактивном режиме робот слушает ВСЁ, без wake word.
        """
        self.interactive_mode = enabled
        if enabled:
            logger.info("🎙️ Интерактивный режим ВКЛЮЧЁН")
            self.activate()  # сразу активируем
        else:
            logger.info("🎙️ Интерактивный режим ВЫКЛЮЧЁН")
            self.deactivate()
    
    def activate(self, duration: float = None):
        """
        АКТИВИРУЕТ РОБОТА НА УКАЗАННОЕ ВРЕМЯ
        
        В активном состоянии робот слушает команды без wake word.
        """
        if duration is None:
            duration = self.active_timeout
        
        self.is_active = True
        self.active_until = time.time() + duration
        logger.debug(f"🎙️ Активирован на {duration:.0f}с")
    
    def deactivate(self):
        """ДЕАКТИВИРУЕТ РОБОТА"""
        self.is_active = False
        self.active_until = 0
        logger.debug("🎙️ Деактивирован")
    
    def is_emergency_stop(self, text: str) -> bool:
        """
        ПРОВЕРЯЕТ, ЯВЛЯЕТСЯ ЛИ ТЕКСТ АВАРИЙНОЙ ОСТАНОВКОЙ
        
        Работает ВСЕГДА, даже без активации.
        """
        stop_words = ["стоп", "стой", "stop", "halt", "тормози"]
        text_lower = text.lower()
        return any(word in text_lower for word in stop_words)
    
    def contains_wake_word(self, text: str) -> bool:
        """Проверяет, содержит ли текст wake word"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.wake_words)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        ВЫЗЫВАЕТСЯ PyAudio ПРИ ПОЛУЧЕНИИ НОВОГО АУДИО-БУФЕРА
        
        Проверяет громкость и, если она выше порога, кладёт данные в очередь.
        """
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        volume = np.max(np.abs(audio_array))
        
        if volume > self.threshold:
            self.audio_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    async def start_listening(self):
        """
        ЗАПУСКАЕТ ПРОСЛУШИВАНИЕ МИКРОФОНА
        
        Запускает отдельный поток для захвата аудио.
        """
        self.is_listening = True
        loop = asyncio.get_event_loop()
        threading.Thread(target=self._listen_thread, args=(loop,), daemon=True).start()
        logger.info("🎙️ VoiceListener запущен")
    
    def _listen_thread(self, loop):
        """
        ПОТОК ЗАХВАТА АУДИО
        
        Работает в фоне, постоянно слушает микрофон.
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4000,
            stream_callback=self.audio_callback
        )
        stream.start_stream()
        
        frames = []
        silence_frames = 0
        max_silence = 20      # кадров тишины для окончания фразы
        min_voice_frames = 10  # минимум кадров для распознавания
        
        while self.is_listening:
            try:
                # Ждём аудио из очереди
                data = self.audio_queue.get(timeout=0.1)
                frames.append(data)
                silence_frames = 0
            except queue.Empty:
                if frames:
                    silence_frames += 1
                    
                    # Достаточно тишины — заканчиваем фразу
                    if silence_frames > max_silence:
                        if len(frames) >= min_voice_frames:
                            audio_data = b''.join(frames)
                            if self.recognizer.AcceptWaveform(audio_data):
                                result = json.loads(self.recognizer.Result())
                                text = result.get('text', '').strip()
                                if text:
                                    self._process_text(text, loop)
                        frames = []
                        silence_frames = 0
                
                # Проверяем таймаут активности
                if self.is_active and not self.interactive_mode:
                    if time.time() > self.active_until:
                        self.is_active = False
                        logger.debug("🎙️ Таймаут активности")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    def _process_text(self, text: str, loop):
        """
        ОБРАБАТЫВАЕТ РАСПОЗНАННЫЙ ТЕКСТ
        
        Определяет тип события (аварийный стоп / wake word / обычная команда)
        и вызывает колбэк.
        """
        logger.info(f"🎙️ Распознано: {text}")
        
        # 1. АВАРИЙНАЯ ОСТАНОВКА — всегда, без активации
        if self.is_emergency_stop(text):
            logger.warning(f"🚨 АВАРИЙНАЯ ОСТАНОВКА: {text}")
            asyncio.run_coroutine_threadsafe(
                self.callback(text, emergency_stop=True),
                loop
            )
            return
        
        # 2. ИНТЕРАКТИВНЫЙ РЕЖИМ — слушаем всё
        if self.interactive_mode:
            if not self.is_active:
                self.activate()
            asyncio.run_coroutine_threadsafe(
                self.callback(text, wake=False, interactive=True),
                loop
            )
            return
        
        # 3. ОБЫЧНЫЙ РЕЖИМ — нужен wake word
        if self.contains_wake_word(text):
            self.activate()
            clean_text = self._remove_wake_word(text)
            asyncio.run_coroutine_threadsafe(
                self.callback(clean_text, wake=True),
                loop
            )
        elif self.is_active:
            asyncio.run_coroutine_threadsafe(
                self.callback(text, wake=False),
                loop
            )
        else:
            logger.debug(f"🎙️ Игнорирую (нет wake word): {text}")
    
    def _remove_wake_word(self, text: str) -> str:
        """Удаляет wake word из текста"""
        text_lower = text.lower()
        for word in self.wake_words:
            if word in text_lower:
                text = text_lower.replace(word, "").strip()
                break
        return text
    
    def stop(self):
        """ОСТАНАВЛИВАЕТ ПРОСЛУШИВАНИЕ"""
        self.is_listening = False
        logger.info("🎙️ VoiceListener остановлен")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    """
    ДЕМОНСТРАЦИЯ РАСПОЗНАВАНИЯ РЕЧИ
    
    Запустите и говорите в микрофон.
    Нажмите Ctrl+C для выхода.
    """
    import os
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ VOICE LISTENER")
    print("="*60 + "\n")
    
    # Путь к модели Vosk (можно указать свой)
    model_path = os.path.expanduser("~/vosk-model-small-ru-0.22")
    
    if not os.path.exists(model_path):
        print(f"❌ Модель Vosk не найдена: {model_path}")
        print("   Скачайте с: https://alphacephei.com/vosk/models")
        print("   Например: vosk-model-small-ru-0.22.zip")
        exit(1)
    
    # Колбэк для вывода распознанного текста
    async def on_speech(text: str, wake: bool = False, 
                        emergency_stop: bool = False, interactive: bool = False):
        if emergency_stop:
            print(f"🚨 АВАРИЙНЫЙ СТОП: {text}")
        elif wake:
            print(f"🔊 WAKE: {text}")
        elif interactive:
            print(f"💬 ИНТЕРАКТИВ: {text}")
        else:
            print(f"🎤 КОМАНДА: {text}")
    
    async def demo():
        listener = VoiceListener(
            model_path=model_path,
            threshold=500,
            wake_words=["гралл", "робот", "эй"],
            active_timeout=15.0
        )
        listener.set_callback(on_speech)
        
        await listener.start_listening()
        
        print("🎙️ Слушаю...")
        print("   Скажите 'Гралл' чтобы активировать")
        print("   Скажите 'стоп' для аварийной остановки")
        print("   Нажмите Ctrl+C для выхода\n")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Завершение...")
        finally:
            listener.stop()
    
    asyncio.run(demo())
