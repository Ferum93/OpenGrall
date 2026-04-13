#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/protocol_v5.py - ЕДИНЫЙ ПРОТОКОЛ ОБМЕНА OpenGrall v5.0                  ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это "общий язык" для всех компонентов системы.                             ║
║   Агент, TinyML, сервер, оператор, лидар, VLM — все говорят на этом языке.  ║
║                                                                              ║
║ ПОЧЕМУ JSON, А НЕ БИНАРНЫЙ ФОРМАТ:                                           ║
║   1. Нативный язык LLM — модель сама генерирует и читает JSON.               ║
║   2. Человекочитаемость — логи можно читать без спец. инструментов.          ║
║   3. Лёгкость расширения — добавить поле за секунды.                         ║
║   4. Для слабых каналов (BLE) есть адаптеры сжатия в 2-4 байта.              ║
║                                                                              ║
║ ЭТО НАШ АНАЛОГ DDS ИЗ ROS, НО ПРОЩЕ И ДРУЖЕЛЮБНЕЕ К LLM.                     ║
║                                                                              ║
║ ФОРМАТ СООБЩЕНИЯ:                                                            ║
║   {                                                                          ║
║     "version": "5.0",              // версия протокола                        ║
║     "message_id": "uuid",          // уникальный ID                           ║
║     "timestamp": 1712345678.123,   // когда отправлено                        ║
║     "source": "agent",             // кто отправил                            ║
║     "source_type": "agent",        // тип источника (для весов)               ║
║     "target": "esp",               // кому (или "all")                        ║
║     "type": "command",             // command / data / response / reflex      ║
║     "capability": "motor.move",    // ЧТО делает сообщение                    ║
║     "data": {...},                 // полезная нагрузка                       ║
║     "weight": 0.85,                // базовый вес источника                   ║
║     "priority": 5                  // 1-10 (1 = наивысший)                    ║
║   }                                                                          ║
║                                                                              ║
║ БАЗОВЫЕ ВЕСА ИСТОЧНИКОВ:                                                     ║
║   tinyml: 1.0 (право вето), lidar: 0.95, agent: 0.85, operator: 0.80,        ║
║   vlm: 0.70, llm: 0.50, unknown: 0.50                                        ║
║                                                                              ║
║ КОМАНДЫ ДВИЖЕНИЯ (ПОДДЕРЖИВАЮТ ДВА РЕЖИМА):                                  ║
║   - duration: ехать N секунд (для простых платформ)                          ║
║   - distance/angle: проехать N метров / повернуть на N° (требует IMU)        ║
║   Исполнение по distance/angle — НА СТОРОНЕ TINYML.                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import uuid
import struct
import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum

logger = logging.getLogger(__name__)

# ================================================================
# КОНСТАНТЫ
# ================================================================

VERSION = "5.0"
DECAY_FACTOR = 0.9
MIN_WEIGHT = 0.1

BASE_WEIGHTS = {
    "tinyml": 1.0,      # исполнитель, право вето
    "lidar": 0.95,      # самый точный сенсор
    "agent": 0.85,      # стратегические решения
    "operator": 0.80,   # человек
    "vision_memory": 0.75,
    "vlm": 0.70,        # может ошибаться
    "esp12": 0.70,
    "llm": 0.50,        # самая ненадёжная в real-time
    "unknown": 0.50
}


class SourceType(str, Enum):
    TINYML = "tinyml"
    LIDAR = "lidar"
    VLM = "vlm"
    LLM = "llm"
    OPERATOR = "operator"
    ESP12 = "esp12"
    AGENT = "agent"
    VISION_MEMORY = "vision_memory"
    UNKNOWN = "unknown"


class MessageType(str, Enum):
    COMMAND = "command"
    DATA = "data"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    REGISTER = "register"
    FEEDBACK = "feedback"
    REFLEX = "reflex"


class RobotProtocolV5:
    """
    ЕДИНЫЙ ПРОТОКОЛ ОБМЕНА OpenGrall v5.0
    
    Все компоненты общаются на JSON через этот протокол.
    Для слабых каналов (BLE) есть адаптеры сжатия.
    """
    
    VERSION = VERSION
    DECAY_FACTOR = DECAY_FACTOR
    BASE_WEIGHTS = BASE_WEIGHTS
    
    @classmethod
    def create_message(cls,
                       source: str,
                       capability: str,
                       data: Dict[str, Any] = None,
                       target: str = "all",
                       msg_type: str = "command",
                       source_type: str = None,
                       priority: int = 5,
                       metadata: Dict = None) -> Dict[str, Any]:
        """Создаёт сообщение в формате протокола v5.0"""
        if not source_type:
            source_type = source.split("_")[0]
            if source_type not in BASE_WEIGHTS:
                source_type = "unknown"
        
        now = time.time()
        timestamp_us = time.time_ns() // 1000
        message_id = str(uuid.uuid4())
        
        return {
            "version": cls.VERSION,
            "message_id": message_id,
            "timestamp": now,
            "timestamp_us": timestamp_us,
            "source": source,
            "source_type": source_type,
            "target": target,
            "type": msg_type,
            "capability": capability,
            "data": data or {},
            "weight": BASE_WEIGHTS.get(source_type, 0.5),
            "priority": priority,
            "metadata": {
                "ttl": 5.0,
                "channel": "wifi",
                **(metadata or {})
            }
        }
    
    @classmethod
    def create_motor_command(cls,
                             left: int,
                             right: int,
                             source: str = "agent",
                             duration: float = None,
                             distance: float = None,
                             angle: float = None) -> Dict:
        """
        Создаёт команду движения.
        
        Режимы:
        - duration: ехать N секунд (без IMU)
        - distance: проехать N метров (с одометрией)
        - angle: повернуть на N градусов (с IMU)
        
        Исполнение по distance/angle — на стороне TinyML.
        """
        data = {"left": left, "right": right}
        if duration is not None:
            data["duration"] = duration
        if distance is not None:
            data["distance"] = distance
        if angle is not None:
            data["angle"] = angle
        
        return cls.create_message(
            source=source,
            source_type=source.split("_")[0],
            capability="motor.move",
            data=data
        )
    
    @classmethod
    def create_reflex_notification(cls,
                                   reflex_type: str,
                                   distance_cm: float,
                                   action_taken: str,
                                   pattern: Dict = None) -> Dict:
        """
        Уведомление о рефлексе. TinyML УЖЕ выполнил действие.
        Агент получает это как информацию, а не как команду.
        """
        return cls.create_message(
            source="tinyml",
            source_type="tinyml",
            capability="reflex",
            msg_type="reflex",
            priority=1,
            data={
                "reflex": {
                    "triggered": True,
                    "type": reflex_type,
                    "distance_cm": distance_cm,
                    "action_taken": action_taken,
                    "pattern": pattern or {}
                }
            }
        )
    
    @classmethod
    def calculate_weight(cls,
                         source_type: str,
                         timestamp: float,
                         base_weight: float = None,
                         confidence: float = 1.0) -> float:
        """Рассчитывает вес сообщения с учётом возраста и достоверности"""
        base = base_weight or BASE_WEIGHTS.get(source_type, 0.5)
        age = time.time() - timestamp
        decay = DECAY_FACTOR ** age
        weight = base * decay * confidence
        return max(MIN_WEIGHT, min(1.0, weight))
    
    @classmethod
    def parse_message(cls, raw_msg: Union[str, bytes, Dict]) -> Optional[Dict]:
        """Разбирает входящее сообщение и добавляет age и weight"""
        try:
            if isinstance(raw_msg, (str, bytes)):
                msg = json.loads(raw_msg)
            else:
                msg = raw_msg
            
            required = ["version", "message_id", "timestamp", "source", "capability"]
            if not all(f in msg for f in required):
                logger.warning(f"Сообщение не соответствует протоколу")
                return None
            
            if "timestamp" in msg:
                msg["age"] = time.time() - msg["timestamp"]
                msg["weight"] = cls.calculate_weight(
                    msg.get("source_type", "unknown"),
                    msg["timestamp"],
                    confidence=msg.get("data", {}).get("confidence", 1.0)
                )
            
            return msg
        except Exception as e:
            logger.error(f"Ошибка парсинга: {e}")
            return None
    
    @classmethod
    def create_response(cls,
                        original_msg: Dict[str, Any],
                        status: str = "success",
                        data: Dict[str, Any] = None,
                        reason: str = "") -> Dict:
        """Создаёт ответ на сообщение"""
        return cls.create_message(
            source=original_msg.get("target", "system"),
            source_type=original_msg.get("source_type", "unknown"),
            capability=original_msg.get("capability", "response"),
            msg_type="response",
            data=data or {},
            metadata={
                "in_response_to": original_msg.get("message_id"),
                "status": status,
                "reason": reason
            }
        )


class ProtocolAdapters:
    """
    АДАПТЕРЫ ДЛЯ СЖАТИЯ СООБЩЕНИЙ ПОД РАЗНЫЕ КАНАЛЫ
    
    - BLE: сжатие JSON в 2-4 байта (экономия трафика)
    - LiDAR: бинарная упаковка кластеров
    - WiFi: полный JSON
    """
    
    BLE_COMMANDS = {
        "motor.move": 0x01,
        "light.set": 0x02,
        "sensor.tof.read": 0x03,
        "servo.set": 0x04,
        "system.ping": 0x05,
        "reflex": 0x06
    }
    
    @staticmethod
    def to_ble(message: Dict) -> bytes:
        """Сжимает JSON в BLE-пакет (2-4 байта)"""
        cap = message.get("capability", "")
        data = message.get("data", {})
        
        if cap == "motor.move":
            left = data.get("left", 0)
            right = data.get("right", 0)
            return bytes([
                ProtocolAdapters.BLE_COMMANDS[cap],
                (left >> 4) & 0xFF,
                ((left & 0x0F) << 4) | ((right >> 8) & 0x0F),
                right & 0xFF
            ])
        
        elif cap == "light.set":
            state = 1 if data.get("state") else 0
            brightness = data.get("brightness", 255)
            return bytes([0x02, (state << 7) | (brightness & 0x7F)])
        
        elif cap == "reflex":
            reflex = data.get("reflex", {})
            reflex_type = reflex.get("type", "unknown")
            distance = reflex.get("distance_cm", 0)
            
            type_map = {
                "obstacle_left": 0,
                "obstacle_right": 1,
                "obstacle_front": 2,
                "obstacle_all": 3
            }
            type_code = type_map.get(reflex_type, 0)
            
            # Добавляем сжатый timestamp (доли секунды, 0-255 = 0-2.55с)
            timestamp = message.get("timestamp", 0)
            time_byte = int((timestamp % 1) * 255)  # дробная часть секунды
            
            return bytes([0x06, (type_code << 4) | min(distance // 10, 15), time_byte])
        
        return b''
    
    @staticmethod
    def from_ble(ble_data: bytes) -> Dict:
        """Восстанавливает JSON из BLE-пакета"""
        if not ble_data:
            return {}
        
        cmd = ble_data[0]
        
        if cmd == 0x01 and len(ble_data) >= 4:
            left = (ble_data[1] << 4) | ((ble_data[2] >> 4) & 0x0F)
            right = ((ble_data[2] & 0x0F) << 8) | ble_data[3]
            return {
                "capability": "motor.move",
                "data": {"left": left, "right": right}
            }
        
        elif cmd == 0x06 and len(ble_data) >= 3:
            type_code = (ble_data[1] >> 4) & 0x0F
            distance = (ble_data[1] & 0x0F) * 10
            time_byte = ble_data[2] if len(ble_data) > 2 else 0
            
            type_map = {
                0: "obstacle_left",
                1: "obstacle_right",
                2: "obstacle_front",
                3: "obstacle_all"
            }
            
            # Восстанавливаем timestamp (приблизительно)
            now = time.time()
            fraction = time_byte / 255.0
            timestamp = int(now) + fraction
            
            return {
                "capability": "reflex",
                "timestamp": timestamp,
                "data": {
                    "reflex": {
                        "triggered": True,
                        "type": type_map.get(type_code, "unknown"),
                        "distance_cm": distance
                    }
                }
            }
        
        return {}
    
    @staticmethod
    def pack_lidar_data(clusters: List[Dict]) -> bytes:
        """Бинарная упаковка данных лидара"""
        if len(clusters) > 50:
            binary = bytearray()
            for c in clusters:
                binary.extend(struct.pack('fff', 
                    c.get('x', 0), 
                    c.get('y', 0), 
                    c.get('velocity', 0)
                ))
            return bytes(binary)
        return b''


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("OpenGrall Protocol v5.0 — ДЕМОНСТРАЦИЯ")
    print("="*60 + "\n")
    
    # 1. Команда движения (по времени)
    cmd_time = RobotProtocolV5.create_motor_command(
        left=300, right=300, duration=2.0
    )
    print("📤 Команда движения (по времени):")
    print(json.dumps(cmd_time, indent=2))
    
    # 2. Команда движения (по расстоянию — требует IMU)
    cmd_dist = RobotProtocolV5.create_motor_command(
        left=300, right=300, distance=1.0
    )
    print("\n📤 Команда движения (по расстоянию, с IMU):")
    print(json.dumps(cmd_dist, indent=2))
    
    # 3. Команда поворота (по углу — требует IMU)
    cmd_turn = RobotProtocolV5.create_motor_command(
        left=-400, right=400, angle=90
    )
    print("\n📤 Команда поворота (на 90° по IMU):")
    print(json.dumps(cmd_turn, indent=2))
    
    # 4. Уведомление о рефлексе
    reflex = RobotProtocolV5.create_reflex_notification(
        reflex_type="obstacle_front",
        distance_cm=25,
        action_taken="emergency_stop"
    )
    print("\n🚨 Уведомление о рефлексе (TinyML уже остановился):")
    print(json.dumps(reflex, indent=2))
    
    # 5. BLE адаптер
    ble = ProtocolAdapters.to_ble(cmd_time)
    print(f"\n🔵 BLE пакет (4 байта вместо JSON): {ble.hex()}")
    restored = ProtocolAdapters.from_ble(ble)
    print(f"   Восстановлено: {restored}")
    
    # 6. Расчёт веса
    weight = RobotProtocolV5.calculate_weight("lidar", time.time() - 0.5)
    print(f"\n📊 Вес лидара (возраст 0.5с): {weight:.2f}")
    
    print("\n" + "="*60)
    print("✅ Протокол v5.0 готов.")
    print("="*60 + "\n")
