#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/context_builder.py - КОМПРЕССОР ДАННЫХ ДЛЯ LLM                          ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это фильтр, который берёт ОГРОМНЫЙ поток сенсорных данных и сжимает его   ║
║   до МИНИМАЛЬНОГО, но ИНФОРМАТИВНОГО промпта.                                ║
║                                                                              ║
║ ГЛАВНАЯ ЦЕЛЬ — ЭКОНОМИЯ ТОКЕНОВ БЕЗ ПОТЕРИ СМЫСЛА:                           ║
║                                                                              ║
║   Лидар выдаёт 500 точек 50 раз в секунду → это тысячи чисел.               ║
║   VLM возвращает развёрнутое описание на 200 слов.                           ║
║   Одометрия шлёт скорость 100 раз в секунду.                                 ║
║                                                                              ║
║   Если отправить всё это в LLM:                                              ║
║       - Контекст переполнится за 2 секунды                                   ║
║       - LLM запутается в противоречивых данных                               ║
║       - Каждый вызов будет стоить сотни тысяч токенов                        ║
║                                                                              ║
║   ContextBuilder делает вместо этого:                                         ║
║                                                                              ║
║       Лидар (500 точек) → "front=0.4м, left=2.0м, right=2.5м |               ║
║                            ближе 50см: 15°;45° ~40см, v=1.2м/с →"            ║
║       VLM (200 слов) → "сцена: коридор, человек 2.0м, путь: free"            ║
║       Одометрия → "vл=0.3, vп=0.3, курс(global, 0°=nord)=0°"                 ║
║                                                                              ║
║   В результате:                                                              ║
║       - Токенов в 50-100 раз меньше                                          ║
║       - LLM видит ТОЛЬКО СУТЬ, без шума                                       ║
║       - Каждые данные имеют ВЕС и ВОЗРАСТ (чтобы LLM знала, чему верить)    ║
║                                                                              ║
║ ЧТО НЕ ПОПАДАЕТ В ПРОМПТ (И ПОЧЕМУ):                                         ║
║                                                                              ║
║   ✗ Рефлексы TinyML — они УЖЕ отработали. LLM получает обновлённую картину  ║
║                        мира и продолжает планирование.                        ║
║   ✗ Гироскоп — TinyML использует его для стабилизации, LLM не нужен.        ║
║   ✗ Данные с весом < 0.3 — считаются недостоверными и отбрасываются.         ║
║   ✗ Сырые облака точек — LLM не понимает их, нужна семантическая сводка.    ║
║   ✗ Характеристики робота — они зашиты в системный промпт (экономия токенов)║
║                                                                              ║
║ ПРИМЕР ТОГО, ЧТО ПОЛУЧАЕТ LLM (ВСЕГО ~120 ТОКЕНОВ):                          ║
║                                                                              ║
║   ТЕКУЩАЯ СИТУАЦИЯ:                                                          ║
║   Время: 1712345720.45                                                       ║
║   Текущее намерение: движение вперёд                                         ║
║                                                                              ║
║   ДАННЫЕ СЕНСОРОВ (по важности):                                             ║
║     • lidar: front=0.4м, left=2.0м, right=2.5м | ближе 50см: 15°;45° ~40см,  ║
║              0.6м, v=1.2м/с → [вес: 0.95, возраст: 0.02с]                    ║
║     • vlm: сцена: коридор, человек 2.0м, путь: free [вес: 0.65, возраст: 0.5с]║
║     • odometry: vл=0.3, vп=0.3, курс(global, 0°=nord)=0° [вес: 0.68, возраст: 0.04s]║
║                                                                              ║
║   Этого достаточно, чтобы LLM приняла решение:                               ║
║   "Впереди движущийся объект (вероятно, человек, которого VLM видел слева).  ║
║    Нужно остановиться и подождать."                                          ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Если у вас слабая LLM (например, <1B параметров):                       ║
║      - Установите LIDAR_MODE = "4" в config.py (4 сектора вместо 8)          ║
║      - Уменьшите MAX_SENSORS_IN_PROMPT до 3                                  ║
║                                                                              ║
║   2. Если хотите добавить новый сенсор:                                      ║
║      - Добавьте его в _make_summary()                                        ║
║      - Формат должен быть МАКСИМАЛЬНО КОМПАКТНЫМ                             ║
║                                                                              ║
║   3. Характеристики робота меняются в agent_v5.py (_setup_llm_model)         ║
║      и НЕ передаются в каждом промпте (экономия токенов).                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional

# Импортируем конфигурацию
try:
    from config import (
        DANGER_DISTANCE_CM, LIDAR_MODE, MAX_SENSORS_IN_PROMPT
    )
except ImportError:
    # Значения по умолчанию
    DANGER_DISTANCE_CM = 50
    LIDAR_MODE = "8"  # "8" или "4"
    MAX_SENSORS_IN_PROMPT = 5

from core.weight_calculator import WeightCalculator
from core.sensor_memory import SensorMemory

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    КОМПРЕССОР ДАННЫХ ДЛЯ LLM
    
    Берёт сырые данные от сенсоров, диалог, намерение — и сжимает
    до минимального, но информативного промпта.
    
    ПРИНЦИПЫ:
        1. Только данные с весом > 0.3
        2. Только самые свежие версии каждого сенсора
        3. Максимально компактный формат (без JSON, без лишних слов)
        4. Явные метаданные: [вес: X.XX, возраст: Y.YYс]
        5. Сортировка по важности (самые надёжные сенсоры — сверху)
        6. Рефлексы TinyML НЕ передаются (они уже отработали)
        7. Характеристики робота НЕ передаются (они в системном промпте)
    """
    
    def __init__(self, weight_calculator: WeightCalculator, sensor_memory: SensorMemory = None):
        self.weight_calc = weight_calculator
        self.sensor_memory = sensor_memory
        self.previous_decision = None
        self.previous_decision_timestamp = 0
        self.previous_decision_weight = 0.5
        
        # Кэш последних сенсоров (для оценки действий)
        self._last_sensors: List[Dict] = []
        
        logger.info(f"✅ ContextBuilder инициализирован (LIDAR_MODE={LIDAR_MODE})")
    
    # ================================================================
    # ГЛАВНЫЙ МЕТОД — ПОСТРОЕНИЕ КОНТЕКСТА
    # ================================================================
    
    def build_context(self, 
                     dialog_context: Dict,
                     sensor_data: List[Dict],
                     active_reflexes: List[Dict],
                     current_intent: str) -> Dict[str, Any]:
        """
        СТРОИТ ВЗВЕШЕННЫЙ КОНТЕКСТ ДЛЯ LLM
        
        Args:
            dialog_context: история диалога из DialogContext
            sensor_data: сырые данные сенсоров (после WeightCalculator)
            active_reflexes: активные рефлексы (НЕ включаются в контекст!)
            current_intent: текущее намерение из диалога
        
        Returns:
            Словарь контекста, готовый для format_for_llm()
            
        ВАЖНО:
            active_reflexes НЕ включаются в контекст!
            TinyML уже отработал, LLM получает обновлённую картину мира
            и просто продолжает планирование.
        """
        now = time.time()
        
        # 1. Если есть SensorMemory — берём данные из неё (предпочтительно)
        if self.sensor_memory:
            sensor_summaries = self.sensor_memory.get_summaries(min_weight=0.3)
            
            weighted_sensors = []
            for source, info in sensor_summaries.items():
                weighted_sensors.append({
                    "source_type": source,
                    "summary": info["summary"],
                    "effective_weight": info["weight"],
                    "age": info["age"],
                    "timestamp": info["timestamp"]
                })
            
            # Сортируем по весу (самые важные — сверху)
            weighted_sensors.sort(key=lambda x: x['effective_weight'], reverse=True)
        
        else:
            # 2. Fallback: обрабатываем переданные данные
            weighted_sensors = []
            for sensor in sensor_data:
                effective_weight = self.weight_calc.calculate(
                    source_type=sensor.get('source_type', 'unknown'),
                    timestamp=sensor.get('timestamp', now),
                    priority=sensor.get('priority', 5),
                    confidence=sensor.get('confidence', 1.0)
                )
                
                if effective_weight > 0.3:
                    weighted_sensors.append({
                        **sensor,
                        'summary': self._make_summary(sensor),
                        'effective_weight': effective_weight,
                        'age': now - sensor.get('timestamp', now)
                    })
            
            weighted_sensors.sort(key=lambda x: x['effective_weight'], reverse=True)
        
        # 3. Ограничиваем количество сенсоров (экономия токенов)
        weighted_sensors = weighted_sensors[:MAX_SENSORS_IN_PROMPT]
        
        # 4. Сохраняем для последующей оценки действий
        self._last_sensors = weighted_sensors.copy()
        
        # 5. Вес предыдущего решения (затухает со временем)
        prev_weight = 0.0
        if self.previous_decision:
            age = now - self.previous_decision_timestamp
            prev_weight = self.previous_decision_weight * (0.9 ** age)
            prev_weight = max(0.1, min(0.5, prev_weight))
        
        # 6. ФОРМИРУЕМ КОНТЕКСТ (БЕЗ РЕФЛЕКСОВ!)
        context = {
            "timestamp": now,
            "dialog": dialog_context,
            "current_intent": current_intent,
            "sensors": weighted_sensors,
            "previous_decision": {
                "decision": self.previous_decision,
                "weight": prev_weight,
                "age": now - self.previous_decision_timestamp if self.previous_decision else None
            } if self.previous_decision else None,
            "system_status": {
                "tinyml_in_control": True,           # TinyML управляет движением
                "agent_role": "strategic_planning"   # Агент только планирует
            }
        }
        
        return context
    
    # ================================================================
    # ФОРМАТИРОВАНИЕ ДЛЯ LLM (ГЛАВНЫЙ МЕТОД ЭКОНОМИИ ТОКЕНОВ)
    # ================================================================
    
    def format_for_llm(self, context: Dict[str, Any]) -> str:
        """
        ПРЕВРАЩАЕТ КОНТЕКСТ В КОМПАКТНЫЙ ПРОМПТ
        
        Это сердце экономии токенов. Каждая строчка здесь выверена,
        чтобы передать МАКСИМУМ информации МИНИМАЛЬНЫМ количеством слов.
        
        ВАЖНО: Характеристики робота ЗДЕСЬ НЕ ПЕРЕДАЮТСЯ.
        Они зашиты в системный промпт (agent_v5.py) для экономии токенов.
        """
        lines = []
        
        # Заголовок
        lines.append("ТЕКУЩАЯ СИТУАЦИЯ:")
        lines.append(f"Время: {context['timestamp']:.2f}")
        lines.append(f"Текущее намерение: {context['current_intent']}")
        
        # Диалог (последние 3 реплики)
        if context['dialog'].get('turns'):
            lines.append("\nДИАЛОГ:")
            for turn in context['dialog']['turns'][-3:]:
                if turn.get('human'):
                    lines.append(f"Человек: {turn['human']}")
                if turn.get('agent'):
                    lines.append(f"Робот: {turn['agent']}")
        
        # Сенсоры (по важности) — КОМПАКТНО, С ВЕСАМИ И ВОЗРАСТОМ
        if context['sensors']:
            lines.append("\nДАННЫЕ СЕНСОРОВ (по важности):")
            for s in context['sensors']:
                lines.append(
                    f"  • {s['source_type']}: {s.get('summary', '')} "
                    f"[вес: {s['effective_weight']:.2f}, возраст: {s['age']:.2f}с]"
                )
        
        # Предыдущее решение (если есть и имеет вес)
        if context['previous_decision'] and context['previous_decision']['weight'] > 0.2:
            pd = context['previous_decision']
            decision_str = self._format_decision(pd['decision'])
            lines.append(f"\nПРЕДЫДУЩЕЕ РЕШЕНИЕ: {decision_str} "
                        f"[вес: {pd['weight']:.2f}, возраст: {pd['age']:.2f}с]")
        
        return "\n".join(lines)
    
    # ================================================================
    # ФОРМАТИРОВАНИЕ ОТДЕЛЬНЫХ СЕНСОРОВ
    # ================================================================
    
    def _make_summary(self, sensor: Dict) -> str:
        """
        ПРЕВРАЩАЕТ СЫРЫЕ ДАННЫЕ СЕНСОРА В КОМПАКТНУЮ СТРОКУ
        
        Это главный метод компрессии. Каждый тип сенсора имеет свой
        оптимизированный формат вывода.
        
        ВАЖНО: формат ответа VLM настраивается в VLMScanner (prompt).
        Здесь мы только форматируем то, что уже пришло.
        """
        cap = sensor.get('capability', '')
        data = sensor.get('data', {})
        source_type = sensor.get('source_type', 'unknown')
        
        # ==================== LIDAR ====================
        if cap == 'sensor.lidar.scan':
            sectors = data.get('sectors', {})
            clusters = data.get('clusters', [])
            
            if sectors:
                # Режим 4 сектора (для слабых LLM)
                if LIDAR_MODE == "4":
                    distances = data.get('distances', {})
                    return (f"перед={distances.get('front', 0):.1f}м, "
                           f"лев={distances.get('left', 0):.1f}м, "
                           f"прав={distances.get('right', 0):.1f}м")
                
                # Режим 8 секторов (по умолчанию)
                sector_str = ", ".join([f"{s}={d:.1f}м" for s, d in sectors.items()])
                
                # Ближние объекты (опасная дистанция из конфига)
                danger_m = DANGER_DISTANCE_CM / 100.0
                close_clusters = [c for c in clusters if c.get('min_distance', 1.0) < danger_m]
                
                if close_clusters:
                    cluster_strs = []
                    for c in close_clusters[:5]:  # максимум 5 ближних
                        angle_start = c.get('angle_start', 0)
                        angle_end = c.get('angle_end', 0)
                        dist = c.get('min_distance', 0) * 100  # в см
                        size = max(c.get('size', {}).get('w', 0), c.get('size', {}).get('d', 0))
                        speed = c.get('speed', 0)
                        
                        # Стрелка направления (↑ ↓ ← → ↖ ↗ ↙ ↘)
                        arrow = ""
                        if speed > 0.05:
                            vel = c.get('velocity', {})
                            vel_x = vel.get('x', 0)
                            vel_y = vel.get('y', 0)
                            vel_angle = (np.degrees(np.arctan2(vel_y, vel_x)) + 360) % 360
                            
                            if 337.5 <= vel_angle or vel_angle < 22.5:
                                arrow = "↑"
                            elif 22.5 <= vel_angle < 67.5:
                                arrow = "↗"
                            elif 67.5 <= vel_angle < 112.5:
                                arrow = "→"
                            elif 112.5 <= vel_angle < 157.5:
                                arrow = "↘"
                            elif 157.5 <= vel_angle < 202.5:
                                arrow = "↓"
                            elif 202.5 <= vel_angle < 247.5:
                                arrow = "↙"
                            elif 247.5 <= vel_angle < 292.5:
                                arrow = "←"
                            elif 292.5 <= vel_angle < 337.5:
                                arrow = "↖"
                        
                        arrow_str = f" {arrow}" if arrow else ""
                        cluster_strs.append(
                            f"{angle_start:.0f}°;{angle_end:.0f}° ~{dist:.0f}см, "
                            f"{size:.1f}м, v={speed:.1f}м/с{arrow_str}"
                        )
                    
                    return f"{sector_str} | ближе {DANGER_DISTANCE_CM}см: {', '.join(cluster_strs)}"
                return sector_str
            
            # Fallback на 4 направления
            distances = data.get('distances', {})
            return (f"перед={distances.get('front', 0):.1f}м, "
                   f"лев={distances.get('left', 0):.1f}м, "
                   f"прав={distances.get('right', 0):.1f}м")
        
        # ==================== VLM ====================
        # ВАЖНО: формат ответа VLM настраивается в VLMScanner (промпт).
        # Здесь мы только форматируем то, что уже пришло.
        elif cap == 'vision.scene_analysis':
            objects = data.get('objects', [])
            scene = data.get('scene', 'неизвестно')
            path_status = data.get('path_status', '?')
            
            if objects:
                obj_str = ', '.join([f"{o.get('name', '?')} {o.get('distance', 0):.1f}м" 
                                     for o in objects[:3]])
                return f"сцена: {scene}, объекты: {obj_str}, путь: {path_status}"
            return f"сцена: {scene}, путь: {path_status}"
        
        # ==================== ОДОМЕТРИЯ ====================
        # ВАЖНО: курс ГЛОБАЛЬНЫЙ (0° = север), в отличие от лидара (0° = нос)
        elif cap == 'sensor.odometry':
            speed_left = data.get('speed_left', 0)
            speed_right = data.get('speed_right', 0)
            heading = data.get('heading', 0)
            return f"vл={speed_left:.1f}, vп={speed_right:.1f}, курс(global, 0°=nord)={heading:.0f}°"
        
        # ==================== БАТАРЕЯ ====================
        elif cap == 'system.battery':
            level = data.get('level', 0)
            charging = data.get('charging', False)
            return f"{level}% {'(зарядка)' if charging else ''}"
        
        # ==================== ПО УМОЛЧАНИЮ ====================
        else:
            data_str = str(data)[:50]
            return data_str if data_str else source_type
    
    def _format_decision(self, decision: Dict) -> str:
        """Форматирует предыдущее решение LLM в компактный вид"""
        if not decision:
            return "нет"
        
        action = decision.get('action', 'unknown')
        params = decision.get('parameters', decision.get('params', {}))
        
        if action == 'move_forward':
            speed = params.get('speed', 300)
            return f"move_forward(speed={speed})"
        elif action == 'move_backward':
            speed = params.get('speed', 300)
            return f"move_backward(speed={speed})"
        elif action == 'turn_left':
            speed = params.get('speed', 512)
 
