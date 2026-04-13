#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/vlm_scanner.py - ФОНОВЫЙ СКАНЕР VLM          ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «глаза» робота. VLM (Visual Language Model) анализирует кадры с камеры ║
║   и превращает их в структурированное описание сцены.                        ║
║                                                                              ║
║   В отличие от лидара, VLM ПОНИМАЕТ СЕМАНТИКУ:                               ║
║   - Не просто «препятствие на 1.5м», а «стул у стены».                       ║
║   - Не просто «движущийся объект», а «человек идёт вправо».                  ║
║   - Не просто «свободное пространство», а «коридор, ведущий в комнату».      ║
║                                                                              ║                              ║
║                                                                              ║
║   Если у пользователя нет лидара — VLM становится ГЛАВНЫМ источником         ║
║   навигационных данных. Он даёт:                                             ║
║     - Оценку расстояния до объектов (в метрах)                               ║
║     - Направление (слева, справа, впереди)                                   ║
║     - Свободен ли путь                                                       ║
║     - Тип помещения (для навигации между комнатами)                          ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ:                                                            ║
║                                                                              ║
║   1. Фоновый цикл запрашивает кадр с камеры (через WebSocket)                ║
║   2. Кадр отправляется в VLM с чётким промптом                               ║
║   3. VLM возвращает JSON с описанием сцены                                   ║
║   4. Данные проходят через WeightCalculator (обнаружение аномалий)           ║
║   5. Сохраняются в SensorMemory с весом и возрастом                          ║
║   6. Агент получает готовую сводку через ContextBuilder                      ║
║                                                                              ║
║ ФОРМАТ ОТВЕТА VLM (JSON):                                                    ║
║                                                                              ║
║   {                                                                          ║
║     "scene": "коридор",                                                      ║
║     "room_type": "corridor",                                                 ║
║     "objects": [                                                             ║
║       {"name": "человек", "distance": 2.5, "position": "right", "action": "стоит"}║
║     ],                                                                       ║
║     "path_status": "free",  // free / occupied / unknown                     ║
║     "free_space": {"front": 5.0, "left": 2.0, "right": 1.5}                  ║
║   }                                                                          ║
║                                                                              ║
║ ЧТО ДЕЛАТЬ, ЕСЛИ НЕТ VLM:                                                    ║
║                                                                              ║
║   В config.py установите VLM_MODEL = None.                                   ║
║   Сканер будет возвращать заглушку: {"scene": "unknown", "objects": []}.     ║
║   Робот сможет работать на одних только лидарах (если они есть).             ║
║                                                                              ║
║ ТРЕБОВАНИЯ:                                                                  ║
║   - VLM модель (локально через Ollama или облако)                            ║
║   - Камера, доступная через WebSocket сервер                                 ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Промпт для VLM — изменить в _analyze_frame()                            ║
║   2. Интервал сканирования — VLM_SCAN_INTERVAL в config.py                   ║
║   3. Порог бездействия — VLM_IDLE_THRESHOLD в config.py                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import time
import logging
import json
import re
from typing import Optional, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

# Промпт для VLM — запрашиваем СТРОГО JSON
VLM_PROMPT = """Ты — визуальный ассистент робота. Проанализируй изображение и верни ТОЛЬКО JSON.

Опиши:
- scene: что это за место (коридор, комната, кухня, улица, офис)
- objects: массив объектов. Для каждого: name (человек, стул, стол, дверь, кот, ...), distance (в метрах), position (left, right, center, front), action (стоит, идёт, сидит)
- path_status: free (можно ехать), occupied (путь заблокирован), unknown (непонятно)
- free_space: расстояния до препятствий вперёд, влево, вправо (в метрах)

Если не уверен — пиши "unknown".

Формат ответа СТРОГО:
{"scene": "...", "objects": [...], "path_status": "...", "free_space": {"front": X, "left": Y, "right": Z}}

Не пиши ничего кроме JSON."""


class VLMScanner:
    """
    ФОНОВЫЙ СКАНЕР VLM — ГЛАЗА РОБОТА
    
    Работает асинхронно, не блокирует основной цикл.
    Может работать как с реальной VLM, так и без неё (заглушка).
    """
    
    def __init__(self,
                 vlm_client,           # VLM клиент (может быть None)
                 ws_client,            # WebSocket для получения кадров
                 scan_interval: float = 0.5,
                 idle_threshold: float = 10.0,
                 frame_timeout: float = 2.0):
        
        self.vlm = vlm_client
        self.ws = ws_client
        self.scan_interval = scan_interval
        self.idle_threshold = idle_threshold
        self.frame_timeout = frame_timeout
        
        # Хранилище результатов
        self.latest_result: Optional[Dict[str, Any]] = None
        self.result_history = deque(maxlen=10)
        self.last_scan_time: float = 0
        self.last_frame_time: float = 0
        
        # Состояние
        self.is_running = False
        self.is_active = False
        self.task: Optional[asyncio.Task] = None
        self.last_activity_time: float = 0
        
        # Интеграция с фреймворком
        self.weight_calculator = None
        self.sensor_memory = None
        
        # Статистика
        self.stats = {
            "scans_completed": 0,
            "scans_failed": 0,
            "avg_latency": 0.0,
            "vlm_available": vlm_client is not None,
            "frames_captured": 0,
            "frames_failed": 0
        }
        
        logger.info(f"✅ VLMScanner инициализирован (VLM: {self.stats['vlm_available']})")
    
    def set_weight_calculator(self, weight_calc):
        """Подключает WeightCalculator для обнаружения аномалий"""
        self.weight_calculator = weight_calc
    
    def set_sensor_memory(self, memory):
        """Подключает SensorMemory для хранения данных"""
        self.sensor_memory = memory
    
    async def start(self):
        """Запускает фоновое сканирование"""
        if self.task and not self.task.done():
            logger.warning("VLMScanner уже запущен")
            return
        
        self.is_running = True
        self.task = asyncio.create_task(self._scan_loop())
        logger.info("🔄 VLMScanner запущен")
    
    async def stop(self):
        """Останавливает сканирование"""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 VLMScanner остановлен")
    
    def on_movement(self):
        """Вызывается при начале движения — активирует сканер"""
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
            logger.debug("VLMScanner активирован (движение)")
    
    def on_command(self):
        """Вызывается при получении команды — активирует сканер"""
        self.last_activity_time = time.time()
        if not self.is_active:
            self.is_active = True
            logger.debug("VLMScanner активирован (команда)")
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        ВОЗВРАЩАЕТ САМЫЙ СВЕЖИЙ РЕЗУЛЬТАТ СКАНИРОВАНИЯ
        
        Используется агентом для получения данных о сцене.
        """
        if not self.stats["vlm_available"]:
            return self._get_stub_result()
        
        if self.latest_result:
            age = time.time() - self.last_scan_time
            self.latest_result["age"] = age
        
        return self.latest_result
    
    def _get_stub_result(self) -> Dict[str, Any]:
        """Заглушка для работы без VLM"""
        return {
            "timestamp": time.time(),
            "source_type": "vlm",
            "capability": "vision.scene_analysis",
            "data": {
                "scene": "неизвестно",
                "room_type": "unknown",
                "objects": [],
                "path_status": "unknown",
                "free_space": {"front": 10.0, "left": 10.0, "right": 10.0}
            },
            "weight": 0.3,
            "vlm_available": False
        }
    
    async def _scan_loop(self):
        """Бесконечный цикл сканирования"""
        while self.is_running:
            # Проверяем, нужно ли деактивировать
            if self.is_active:
                idle = time.time() - self.last_activity_time
                if idle > self.idle_threshold:
                    self.is_active = False
                    logger.debug(f"VLMScanner деактивирован (бездействие {idle:.1f}с)")
            
            # Сканируем только если активно
            if self.is_active:
                await self._perform_scan()
                await asyncio.sleep(self.scan_interval)
            else:
                await asyncio.sleep(1.0)
    
    async def _perform_scan(self):
        """Выполняет один цикл сканирования"""
        try:
            start_time = time.time()
            
            # 1. Захватываем кадр
            frame = await self._capture_frame()
            if frame is None:
                self.stats["frames_failed"] += 1
                return
            
            self.stats["frames_captured"] += 1
            
            # 2. Анализируем кадр
            if self.vlm:
                result = await self._analyze_frame(frame)
            else:
                result = self._get_stub_result()
            
            if result:
                # 3. Обогащаем весами
                result = self._enrich_with_weights(result)
                
                # 4. Сохраняем в SensorMemory
                self._save_to_memory(result)
                
                # 5. Сохраняем локально
                self.latest_result = result
                self.result_history.append({
                    "timestamp": time.time(),
                    "result": result
                })
                self.last_scan_time = time.time()
                self.stats["scans_completed"] += 1
                
                # 6. Обновляем статистику
                latency = time.time() - start_time
                self.stats["avg_latency"] = (
                    self.stats["avg_latency"] * 0.9 + latency * 0.1
                )
                
                logger.debug(f"📸 VLM скан #{self.stats['scans_completed']} за {latency:.2f}с")
            else:
                self.stats["scans_failed"] += 1
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Ошибка VLM сканирования: {e}")
            self.stats["scans_failed"] += 1
    
    async def _capture_frame(self) -> Optional[bytes]:
        """Запрашивает кадр через WebSocket"""
        try:
            frame_future = asyncio.Future()
            
            if hasattr(self.ws, 'pending_frame'):
                self.ws.pending_frame = frame_future
            else:
                self.ws._temp_frame_future = frame_future
            
            await self.ws.send({"type": "capture_frame"})
            
            frame = await asyncio.wait_for(frame_future, timeout=self.frame_timeout)
            self.last_frame_time = time.time()
            return frame
            
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут получения кадра ({self.frame_timeout}с)")
            return None
        except Exception as e:
            logger.error(f"Ошибка захвата кадра: {e}")
            return None
        finally:
            if hasattr(self.ws, '_temp_frame_future'):
                delattr(self.ws, '_temp_frame_future')
            if hasattr(self.ws, 'pending_frame'):
                self.ws.pending_frame = None
    
    async def _analyze_frame(self, frame: bytes) -> Optional[Dict]:
        """
        ОТПРАВЛЯЕТ КАДР В VLM И ПАРСИТ JSON-ОТВЕТ
        
        ВАЖНО: VLM запускается в отдельном потоке, чтобы не блокировать event loop.
        """
        import base64
        import cv2
        import numpy as np
        
        try:
            # Декодируем изображение
            image_bytes = base64.b64decode(frame)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Не удалось декодировать изображение")
                return None
            
            # Запускаем VLM в отдельном потоке (НЕ БЛОКИРУЕТ)
            result_text = await asyncio.to_thread(
                self.vlm.analyze_scene,
                img,
                VLM_PROMPT
            )
            
            # Парсим JSON из ответа
            parsed = self._parse_vlm_response(result_text)
            
            return {
                "timestamp": time.time(),
                "source_type": "vlm",
                "capability": "vision.scene_analysis",
                "data": parsed,
                "scan_latency": 0  # будет заполнено в _perform_scan
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа VLM: {e}")
            return None
    
    def _parse_vlm_response(self, text: str) -> Dict:
        """
        ПАРСИТ JSON ИЗ ОТВЕТА VLM
        
        VLM должна вернуть строгий JSON. Если не получилось — возвращаем заглушку.
        """
        # Ищем JSON в ответе
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                # Проверяем обязательные поля
                return {
                    "scene": data.get("scene", "неизвестно"),
                    "room_type": data.get("room_type", data.get("scene", "unknown")),
                    "objects": data.get("objects", []),
                    "path_status": data.get("path_status", "unknown"),
                    "free_space": data.get("free_space", {"front": 10.0, "left": 10.0, "right": 10.0})
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback — заглушка
        logger.warning(f"VLM вернула не-JSON: {text[:100]}...")
        return {
            "scene": "неизвестно",
            "room_type": "unknown",
            "objects": [],
            "path_status": "unknown",
            "free_space": {"front": 10.0, "left": 10.0, "right": 10.0}
        }
    
    def _enrich_with_weights(self, result: Dict) -> Dict:
        """Обогащает результат через WeightCalculator"""
        if not self.weight_calculator:
            result["weight"] = 0.7
            return result
        
        enriched = self.weight_calculator.process_with_meta(
            source_type="vlm",
            data=result["data"],
            timestamp=result["timestamp"],
            priority=4,
            confidence=0.7,
            latency=result.get("scan_latency", None)
        )
        
        result["data"] = enriched
        result["weight"] = enriched.get("_meta", {}).get("weight", 0.7)
        
        return result
    
    def _save_to_memory(self, result: Dict):
        """Сохраняет результат в SensorMemory"""
        if not self.sensor_memory:
            return
        
        weight = result.get("weight", 0.7)
        meta = {
            "scan_timestamp": result.get("timestamp"),
            "scan_latency": result.get("scan_latency"),
            "source": "vlm_scanner"
        }
        
        self.sensor_memory.update(
            source="vlm",
            data=result.get("data", {}),
            weight=weight,
            meta=meta
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сканера"""
        return {
            **self.stats,
            "is_active": self.is_active,
            "last_scan_age": time.time() - self.last_scan_time if self.last_scan_time else None,
            "history_count": len(self.result_history)
        }
    
    def activate_now(self):
        """Принудительно активирует сканер"""
        self.is_active = True
        self.last_activity_time = time.time()
    
    def deactivate_now(self):
        """Принудительно деактивирует сканер"""
        self.is_active = False


# ================================================================
# ДЕМОНСТРАЦИЯ (С ЗАГЛУШКОЙ)
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ VLM SCANNER")
    print("="*60 + "\n")
    
    print("⚠️ Для полной демонстрации нужен WebSocket сервер и камера.")
    print("   Здесь показана работа с заглушкой.\n")
    
    # Мокаем WebSocket клиент
    class MockWS:
        async def send(self, data):
            pass
    
    # Создаём сканер без VLM (заглушка)
    scanner = VLMScanner(
        vlm_client=None,
        ws_client=MockWS(),
        scan_interval=0.5
    )
    
    async def demo():
        await scanner.start()
        scanner.activate_now()
        
        print("📸 Сканер активирован. Жду 3 секунды...")
        await asyncio.sleep(3)
        
        latest = scanner.get_latest()
        print(f"\n📋 Последний результат:")
        print(f"   Сцена: {latest['data']['scene']}")
        print(f"   Объекты: {latest['data']['objects']}")
        print(f"   Путь: {latest['data']['path_status']}")
        print(f"   Вес: {latest.get('weight', 'N/A')}")
        
        print(f"\n📊 Статистика:")
        stats = scanner.get_stats()
        print(f"   Сканов: {stats['scans_completed']}")
        print(f"   Ошибок: {stats['scans_failed']}")
        
        await scanner.stop()
        print("\n✅ Демонстрация завершена.")
    
    asyncio.run(demo())
