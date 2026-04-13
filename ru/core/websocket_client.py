#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/websocket_client.py - АСИНХРОННЫЙ КЛИЕНТ ДЛЯ СВЯЗИ С СЕРВЕРОМ           ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «нервная система» робота. Через WebSocket идут ВСЕ команды движения,   ║
║   данные сенсоров, уведомления о рефлексах.                                  ║
║                                                                              ║
║ ГЛАВНАЯ ФИШКА — ПОЛНАЯ АСИНХРОННОСТЬ:                                        ║
║                                                                              ║
║   В классических системах (ROS) клиент ЖДЁТ ответа на каждый запрос.         ║
║   Это убивает отзывчивость.                                                  ║
║                                                                              ║
║   В OpenGrall:                                                               ║
║     • Отправка и приём НЕ БЛОКИРУЮТ основной цикл агента.                    ║
║     • Сообщения приходят когда угодно — колбэк on_message обрабатывает их    ║
║       в тот же момент, не прерывая работу LLM.                               ║
║     • Если связь потеряна — клиент САМ переподключается в фоне.              ║
║                                                                              ║
║   Результат: робот не «зависает» в ожидании ответа от сервера.               ║
║                                                                              ║
║ МОДУЛЬНОСТЬ — КАК ДОБАВИТЬ НОВОЕ УСТРОЙСТВО:                                 ║
║                                                                              ║
║   OpenGrall спроектирован так, чтобы новые сенсоры и контроллеры             ║
║   добавлялись БЕЗ изменения ядра.                                            ║
║                                                                              ║
║   Пример: вы хотите добавить ESP32-CAM для заднего вида.                     ║
║                                                                              ║
║   1. Прошиваете ESP32-CAM кодом, который:                                    ║
║      - подключается к тому же WebSocket серверу                              ║
║      - при подключении отправляет:                                           ║
║        {"type": "register", "role": "camera", "capabilities": ["video_back"]}║
║                                                                              ║
║   2. Сервер автоматически добавляет устройство в реестр.                     ║
║                                                                              ║
║   3. Агент может получать данные с новой камеры через тот же протокол.       ║
║                                                                              ║
║   4. В коде агента НИЧЕГО менять не нужно — SensorMemory автоматически       ║
║      примет новый источник данных с типом "camera_back".                     ║
║                                                                              ║
║   ТО ЖЕ САМОЕ ДЛЯ ДОПОЛНИТЕЛЬНЫХ ДАТЧИКОВ:                                   ║
║   - Отдельная ESP32 с датчиком температуры/влажности                         ║
║   - Подключается к серверу с role="sensor"                                   ║
║   - Шлёт данные в формате протокола                                          ║
║   - Агент видит их в SensorMemory с типом "temperature"                      ║
║                                                                              ║
║   ЭТО И ЕСТЬ НАСТОЯЩАЯ МОДУЛЬНОСТЬ.                                          ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from core.websocket_client import WebSocketClient                          ║
║                                                                              ║
║   client = WebSocketClient("ws://192.168.43.1:5002", on_message)             ║
║   asyncio.create_task(client.connect())  # запускаем в фоне                   ║
║                                                                              ║
║   # Отправить команду (не блокирует)                                         ║
║   await client.send({"type": "motor", "left": 300, "right": 300})            ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. URL сервера — в config.py (WS_URL)                                      ║
║   2. Интервал переподключения — изменить RECONNECT_DELAY                     ║
║   3. Добавить обработку новых типов сообщений — в колбэке on_message         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import websockets
import json
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 3  # секунд между попытками переподключения


class WebSocketClient:
    """
    АСИНХРОННЫЙ WEBSOCKET КЛИЕНТ С АВТОПЕРЕПОДКЛЮЧЕНИЕМ
    
    Отправляет и принимает сообщения, не блокируя основной поток.
    При разрыве соединения автоматически переподключается.
    
    МОДУЛЬНОСТЬ:
        Это НЕ жёстко привязанный к агенту клиент. Любое устройство
        может использовать ТОТ ЖЕ САМЫЙ протокол для регистрации.
    """
    
    def __init__(self, url: str, on_message: Callable):
        self.url = url
        self.on_message = on_message
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        
        logger.info(f"✅ WebSocketClient создан (url={url})")
    
    async def connect(self):
        """Подключается к серверу и запускает бесконечный цикл приёма"""
        while True:
            try:
                logger.info(f"🔌 Подключение к {self.url}...")
                
                self.ws = await websockets.connect(self.url)
                self.connected = True
                logger.info(f"✅ WebSocket подключен к {self.url}")
                
                # Регистрируемся как агент
                await self.send({
                    "type": "register",
                    "role": "agent"
                })
                
                # Слушаем сообщения
                async for message in self.ws:
                    try:
                        data = json.loads(message)
                        asyncio.create_task(self._handle_message(data))
                    except json.JSONDecodeError:
                        logger.error(f"❌ Невалидный JSON: {message[:100]}")
                    except Exception as e:
                        logger.error(f"❌ Ошибка обработки сообщения: {e}")
                        
            except websockets.ConnectionClosed:
                logger.warning("⚠️ Соединение закрыто сервером")
            except Exception as e:
                logger.error(f"❌ Ошибка WebSocket: {e}")
            
            self.connected = False
            logger.info(f"🔄 Переподключение через {RECONNECT_DELAY} сек...")
            await asyncio.sleep(RECONNECT_DELAY)
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Внутренний обработчик — вызывает колбэк асинхронно"""
        try:
            await self.on_message(data)
        except Exception as e:
            logger.error(f"❌ Ошибка в колбэке on_message: {e}")
    
    async def send(self, data: Dict[str, Any]):
        """Отправляет сообщение. НЕ БЛОКИРУЕТ."""
        if self.ws and self.connected:
            try:
                await self.ws.send(json.dumps(data))
                logger.debug(f"📤 Отправлено: {data.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"❌ Ошибка отправки: {e}")
                self.connected = False
        else:
            logger.warning(f"⚠️ Нет соединения, сообщение не отправлено")
    
    def is_connected(self) -> bool:
        return self.connected and self.ws is not None
    
    async def close(self):
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info("🛑 WebSocket клиент остановлен")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ WEBSOCKET КЛИЕНТА")
    print("="*60 + "\n")
    
    async def on_message(data: Dict):
        msg_type = data.get('type', 'unknown')
        print(f"📥 Получено: {msg_type}")
        if msg_type == 'registered':
            print("   ✅ Агент зарегистрирован на сервере!")
    
    async def demo():
        client = WebSocketClient("ws://127.0.0.1:5002", on_message)
        task = asyncio.create_task(client.connect())
        
        await asyncio.sleep(2)
        
        if client.is_connected():
            print("\n✅ Клиент подключён!")
            await client.send({"type": "ping"})
        else:
            print("\n❌ Не удалось подключиться. Сервер запущен?")
        
        await asyncio.sleep(5)
        await client.close()
        task.cancel()
    
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n👋 Выход")
