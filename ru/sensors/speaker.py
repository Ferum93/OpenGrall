#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/speaker.py - СИНТЕЗАТОР РЕЧИ (ГОЛОС РОБОТА)                          ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Модуль для озвучивания текста. Создаёт синтетический «роботизированный»    ║
║   голос, который звучит как робот из фантастических фильмов 80-х.            ║
║                                                                              ║
║ ПОЧЕМУ ТАКОЙ ГОЛОС:                                                          ║
║   1. Харизма — робот звучит как робот, а не притворяется человеком.          ║
║   2. Ошибки маленьких LLM становятся «диалектом робота», а не багом.         ║
║   3. Отстройка от Алисы/Маруси — у Гралла свой узнаваемый тембр.             ║
║                                                                              ║
║ ПОДДЕРЖИВАЕМЫЕ ДВИЖКИ (выбирается автоматически):                            ║
║   • RHVoice — голос Aleksandr (советский диктор-робот, лучший русский)       ║
║   • eSpeak — fallback, работает везде                                        ║
║                                                                              ║
║ ПРЕДУСТАНОВЛЕННЫЕ ПРОФИЛИ:                                                   ║
║   RU_ROBOT — русский, монотонный, с металлическим оттенком.                  ║
║   EN_ROBOT — британский мужской, роботизированный (mb-en1).                  ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║   from sensors.speaker import TTSEngine, RU_ROBOT                            ║
║   tts = TTSEngine(**RU_ROBOT)                                                ║
║   await tts.speak("Привет, я робот Гралл")                                   ║
║                                                                              ║
║ УСТАНОВКА:                                                                   ║
║   Termux: pkg install rhvoice   (или pkg install espeak для fallback)        ║
║   Linux:  sudo apt install rhvoice (или espeak)                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import subprocess
import logging
import shutil
import os

logger = logging.getLogger(__name__)

# Предустановленные профили голосов
RU_ROBOT = {
    "voice": "ru",
    "speed": 110,      # слов в минуту
    "pitch": 45,       # высота (0-99, 45 = металлический оттенок)
    "amplitude": 100   # громкость (0-200)
}

EN_ROBOT = {
    "voice": "mb-en1",  # британский мужской, встроенный роботизированный тембр
    "speed": 120,
    "pitch": 50,
    "amplitude": 100
}


class TTSEngine:
    """
    СИНТЕЗАТОР РЕЧИ С АВТОВЫБОРОМ ДВИЖКА
    
    Пробует RHVoice (лучший русский), fallback на eSpeak.
    """
    
    def __init__(self, 
                 voice: str = "ru",
                 speed: int = 110,
                 pitch: int = 45,
                 amplitude: int = 100):
        
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.amplitude = amplitude
        
        self.engine = self._detect_engine()
        
        logger.info(f"✅ TTSEngine готов (движок: {self.engine})")
    
    def _detect_engine(self) -> str:
        """Определяет, какой TTS доступен"""
        if shutil.which("RHVoice-client") or shutil.which("rhvoice-client"):
            return "rhvoice"
        
        if shutil.which("espeak"):
            return "espeak"
        
        logger.error("❌ Нет ни RHVoice, ни eSpeak!")
        logger.error("   Установите: pkg install rhvoice (Termux) или apt install rhvoice (Linux)")
        logger.error("   Или fallback: pkg install espeak")
        raise RuntimeError("TTS движок не найден")
    
    async def speak(self, text: str):
        """Произносит текст через доступный движок"""
        if not text:
            return
        
        log_text = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"🗣️ Робот говорит: {log_text}")
        
        if self.engine == "rhvoice":
            await self._speak_rhvoice(text)
        else:
            await self._speak_espeak(text)
    
    async def _speak_rhvoice(self, text: str):
        """Использует RHVoice (Александр)"""
        rh_rate = max(0.5, min(2.0, self.speed / 120))
        rh_pitch = max(-3, min(3, int((self.pitch - 50) / 10)))
        
        tmp_file = "/tmp/tts_text.txt"
        with open(tmp_file, "w") as f:
            f.write(text)
        
        # Пробуем разные варианты имени клиента
        client_cmd = None
        if shutil.which("RHVoice-client"):
            client_cmd = ["RHVoice-client", "-s", "Aleksandr"]
        elif shutil.which("rhvoice-client"):
            client_cmd = ["rhvoice-client", "-s", "Aleksandr"]
        
        if client_cmd:
            cmd = client_cmd + [
                "--rate", str(rh_rate),
                "--pitch", str(rh_pitch),
                "--volume", str(self.amplitude / 100),
                "-i", tmp_file
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                await process.wait()
            except Exception as e:
                logger.error(f"Ошибка RHVoice: {e}, fallback на eSpeak")
                await self._speak_espeak(text)
        
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
    
    async def _speak_espeak(self, text: str):
        """Fallback: eSpeak"""
        voice = self.voice if self.voice in ["ru", "en", "mb-en1"] else "ru"
        
        cmd = [
            "espeak",
            "-v", voice,
            "-s", str(self.speed),
            "-p", str(self.pitch),
            "-a", str(self.amplitude),
            text
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await process.wait()
        except Exception as e:
            logger.error(f"Ошибка eSpeak: {e}")
    
    def speak_sync(self, text: str):
        """Синхронная версия (для использования в потоках)"""
        if not text:
            return
        
        if self.engine == "espeak":
            cmd = [
                "espeak",
                "-v", self.voice if self.voice in ["ru", "en", "mb-en1"] else "ru",
                "-s", str(self.speed),
                "-p", str(self.pitch),
                "-a", str(self.amplitude),
                text
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error(f"Ошибка eSpeak (sync): {e}")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ TTS")
    print("="*60 + "\n")
    
    async def demo():
        print("🇷🇺 Русский робот (RU_ROBOT):")
        tts_ru = TTSEngine(**RU_ROBOT)
        await tts_ru.speak("Привет. Я робот Гралл. Моя речь синтетическая.")
        
        print("\n🇬🇧 Английский робот (EN_ROBOT):")
        tts_en = TTSEngine(**EN_ROBOT)
        await tts_en.speak("Hello. I am Grall. I am a robot.")
        
        print("\n✅ Демонстрация завершена.")
    
    asyncio.run(demo())
