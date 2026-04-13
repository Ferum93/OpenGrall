#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/sensor_memory.py - ЕДИНОЕ ХРАНИЛИЩЕ ВСЕХ СЕНСОРНЫХ ДАННЫХ               ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это "общая доска", куда все сенсоры складывают свои показания.             ║
║   Лидар, VLM, одометрия, батарея — всё в одном месте.                       ║
║                                                                              ║
║ ПОЧЕМУ ЭТО ВАЖНО:                                                            ║
║   В классической робототехнике каждый сенсор живёт своей жизнью.            ║
║   Агент должен сам решать, кому верить и какие данные свежее.               ║
║   SensorMemory решает это автоматически:                                     ║
║                                                                              ║
║     1. Все данные приходят АСИНХРОННО (кто когда успел)                      ║
║     2. Каждые данные проходят через WeightCalculator (вес + возраст)         ║
║     3. Агент забирает САМЫЕ СВЕЖИЕ версии с НАИВЫСШИМ ВЕСОМ                  ║
║     4. Никто никого не ждёт — робот не фризит                                ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ (ПРОСТЫМИ СЛОВАМИ):                                         ║
║                                                                              ║
║   Лидар обновляется 50 раз в секунду → кладёт данные с весом 0.95            ║
║   VLM обновляется 1 раз в секунду → кладёт данные с весом 0.70               ║
║   Когда агенту нужно принять решение, он спрашивает:                         ║
║       "Дай мне все данные с весом выше 0.3"                                  ║
║   SensorMemory возвращает:                                                   ║
║       - лидар (вес 0.95, возраст 0.02с)                                     ║
║       - одометрию (вес 0.68, возраст 0.04с)                                 ║
║       - VLM (вес 0.65, возраст 0.70с)                                       ║
║                                                                              ║
║   Агент НЕ ЖДЁТ VLM — если VLM устарел, его вес снижается,                   ║
║   и агент принимает решение на основе лидара, который всегда свежий.        ║
║                                                                              ║
║ ВАЖНО:                                                                       ║
║   - Данные НЕ удаляются, пока не устареют (max_age=10 секунд)                ║
║   - У каждого источника хранится история (последние 100 значений)            ║
║   - Вес динамический: старые данные весят меньше                             ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from core.sensor_memory import SensorMemory                                ║
║                                                                              ║
║   memory = SensorMemory(max_age=10.0)                                        ║
║                                                                              ║
║   # Лидар кладёт данные (50 раз в секунду)                                   ║
║   memory.update("lidar", {"sectors": {...}}, weight=0.95)                    ║
║                                                                              ║
║   # VLM кладёт данные (1 раз в секунду)                                      ║
║   memory.update("vlm", {"scene": "коридор", "objects": [...]}, weight=0.70)  ║
║                                                                              ║
║   # Агент забирает самые свежие данные                                       ║
║   summaries = memory.get_summaries(min_weight=0.3)                           ║
║   for source, info in summaries.items():                                     ║
║       print(f"{source}: {info['summary']} [вес: {info['weight']:.2f}]")      ║
║                                                                              ║
║ КАСТОМИЗАЦИЯ ПОД ВАШУ ПЛАТФОРМУ:                                             ║
║                                                                              ║
║   1. Если у вас есть новый тип сенсора (например, ультразвуковой датчик):   ║
║      - Добавьте его в _make_summary()                                        ║
║      - Определите базовый вес в config.py (INITIAL_WEIGHTS)                  ║
║                                                                              ║
║   2. Если хотите изменить время жизни данных:                                ║
║      memory = SensorMemory(max_age=5.0)  # 5 секунд вместо 10                ║
║                                                                              ║
║   3. Если хотите сохранять больше истории:                                   ║
║      измените MAX_HISTORY_PER_SOURCE в начале файла                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import json
import hashlib
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)

# ================================================================
# КОНСТАНТЫ (можно менять под свою задачу)
# ================================================================

DECAY_FACTOR = 0.9          # скорость затухания веса (0.9 = каждую секунду вес умножается на 0.9)
MIN_WEIGHT = 0.1            # минимальный вес (данные с весом ниже этого не отдаются)
MAX_HISTORY_PER_SOURCE = 100  # сколько последних значений хранить для каждого источника
DEFAULT_MAX_AGE = 10.0      # через сколько секунд данные считаются устаревшими


@dataclass
class SensorSnapshot:
    """
    ОДИН СНИМОК ДАННЫХ ОТ ОДНОГО СЕНСОРА
    
    Это "фотография" состояния сенсора в момент времени.
    Хранит:
        - source: кто прислал (lidar, vlm, odometry)
        - data: сами данные (словарь)
        - timestamp: когда прислали
        - weight: вес (на момент получения)
        - meta: дополнительные метаданные (ETA, аномалии и т.д.)
    
    ВАЖНО: вес меняется со временем! Чем старше данные, тем меньше вес.
    """
    source: str
    data: Dict[str, Any]
    timestamp: float
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_weight(self, current_time: float = None) -> float:
        """
        ВОЗВРАЩАЕТ ТЕКУЩИЙ ВЕС С УЧЁТОМ ВОЗРАСТА
        
        Формула: текущий_вес = начальный_вес * (DECAY_FACTOR ** возраст)
        
        Пример:
            Данные пришли с весом 0.95, возраст 0.5 секунды
            DECAY_FACTOR = 0.9
            текущий_вес = 0.95 * (0.9 ** 0.5) = 0.95 * 0.95 = 0.90
            
        Через 5 секунд:
            текущий_вес = 0.95 * (0.9 ** 5) = 0.95 * 0.59 = 0.56
            
        Через 10 секунд:
            текущий_вес = 0.95 * (0.9 ** 10) = 0.95 * 0.35 = 0.33
        """
        if current_time is None:
            current_time = time.time()
        age = current_time - self.timestamp
        # Экспоненциальное затухание: чем старше, тем меньше вес
        return self.weight * (DECAY_FACTOR ** age)
    
    def is_valid(self, min_weight: float = MIN_WEIGHT) -> bool:
        """Проверяет, не устарели ли данные (вес выше порога)"""
        return self.get_current_weight() > min_weight
    
    def get_age(self, current_time: float = None) -> float:
        """Возвращает возраст данных в секундах"""
        if current_time is None:
            current_time = time.time()
        return current_time - self.timestamp
    
    def to_dict(self) -> Dict:
        """Превращает снимок в словарь (для сохранения в файл)"""
        return {
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
            "weight": self.weight,
            "meta": self.meta,
            "current_weight": self.get_current_weight(),
            "age": self.get_age()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SensorSnapshot':
        """Восстанавливает снимок из словаря (из файла)"""
        return cls(
            source=data["source"],
            data=data["data"],
            timestamp=data["timestamp"],
            weight=data.get("weight", 1.0),
            meta=data.get("meta", {})
        )


class SensorMemory:
    """
    ЕДИНОЕ ХРАНИЛИЩЕ ВСЕХ СЕНСОРНЫХ ДАННЫХ
    
    Это главный "мозговой центр" для всех сенсоров.
    Все данные стекаются сюда, и только отсюда агент их забирает.
    
    ПРИНЦИП РАБОТЫ:
    
        1. Лидар обновляется 50 раз в секунду → вызывает memory.update("lidar", data, weight=0.95)
        2. VLM обновляется 1 раз в секунду → вызывает memory.update("vlm", data, weight=0.70)
        3. Агент вызывает memory.get_summaries() → получает всё, что свежее порога
        
    ПОЧЕМУ ЭТО ЛУЧШЕ, ЧЕМ КЛАССИЧЕСКИЙ ПОДХОД:
    
        В классике (ROS): сенсоры синхронизируются по самому медленному.
        Если VLM тормозит — робот ждёт и фризит.
        
        В OpenGrall: сенсоры работают асинхронно.
        Агент берёт то, что есть, и не ждёт медленные сенсоры.
        VLM может обновляться когда угодно — это не блокирует движение.
        
    КАК АГЕНТ ЗАБИРАЕТ ДАННЫЕ:
    
        # В agent_v5.py, метод collect_sensor_data():
        summaries = self.sensor_memory.get_summaries(min_weight=0.3)
        
        # Возвращает:
        {
            "lidar": {
                "summary": "front=0.4м, front_left=0.5м, ... | ближе 80см: -5°;15° ~40см",
                "weight": 0.95,
                "age": 0.02,
                "timestamp": 1712345720.45
            },
            "odometry": {
                "summary": "vл=0.3, vп=0.3, курс=0°",
                "weight": 0.68,
                "age": 0.04,
                "timestamp": 1712345720.43
            },
            "vlm": {
                "summary": "сцена: коридор, объекты: человек 1.2м, путь: free",
                "weight": 0.65,
                "age": 0.70,
                "timestamp": 1712345719.75
            }
        }
        
        Агент видит, что лидар самый свежий (вес 0.95) и доверяет ему больше.
        VLM устарел (вес 0.65) — агент учитывает, но не ждёт обновления.
    """
    
    def __init__(self, max_age: float = DEFAULT_MAX_AGE):
        """
        СОЗДАЁТ ПУСТОЕ ХРАНИЛИЩЕ
        
        Args:
            max_age: максимальный возраст данных в секундах
                     данные старше этого возраста удаляются автоматически
        """
        self.max_age = max_age
        
        # Текущие снимки (последние от каждого источника)
        # Пример: {"lidar": SensorSnapshot(...), "vlm": SensorSnapshot(...)}
        self._snapshots: Dict[str, SensorSnapshot] = {}
        
        # История по источникам (для отладки и анализа)
        # Пример: {"lidar": [snapshot1, snapshot2, ...]}
        self._history: Dict[str, List[SensorSnapshot]] = {}
        
        # Индекс весов (для быстрого поиска самых важных данных)
        self._weight_index: List[Tuple[str, float]] = []
        
        # Блокировка для многопоточности (чтобы данные не перемешались)
        self.lock = threading.Lock()
        
        # Статистика
        self.stats = {
            "updates": 0,        # сколько раз обновили данные
            "cleaned": 0,        # сколько устаревших записей удалили
            "last_cleanup": time.time()
        }
        
        logger.info(f"✅ SensorMemory инициализирована (max_age={max_age:.1f}с)")
    
    # ================================================================
    # ОСНОВНЫЕ МЕТОДЫ (ТО, ЧТО ИСПОЛЬЗУЕТ АГЕНТ)
    # ================================================================
    
    def update(self, source: str, data: Dict[str, Any], 
               weight: float = 1.0, meta: Dict = None) -> str:
        """
        ОБНОВИТЬ ДАННЫЕ ОТ ИСТОЧНИКА
        
        ЭТОТ МЕТОД ВЫЗЫВАЮТ СЕНСОРЫ, КОГДА У НИХ ЕСТЬ НОВЫЕ ДАННЫЕ.
        
        Args:
            source: кто прислал ("lidar", "vlm", "odometry", "battery")
            data: сами данные (словарь)
            weight: начальный вес (чем точнее сенсор, тем выше вес)
            meta: дополнительные метаданные (ETA, аномалии и т.д.)
        
        Returns:
            str: идентификатор обновления (для отладки)
        
        Пример вызова (из лидара):
        
            memory.update("lidar", {
                "sectors": {"front": 0.4, "left": 2.0, "right": 2.8},
                "clusters": [{"angle": 15, "distance": 0.4, "speed": 1.2}]
            }, weight=0.95)
        """
        with self.lock:
            timestamp = time.time()
            
            # Создаём снимок
            snapshot = SensorSnapshot(
                source=source,
                data=data,
                timestamp=timestamp,
                weight=weight,
                meta=meta or {}
            )
            
            # Сохраняем как текущий снимок (старый затирается)
            old_snapshot = self._snapshots.get(source)
            self._snapshots[source] = snapshot
            
            # Сохраняем в историю (для анализа)
            if source not in self._history:
                self._history[source] = []
            self._history[source].append(snapshot)
            
            # Ограничиваем историю (чтобы не разрасталась)
            if len(self._history[source]) > MAX_HISTORY_PER_SOURCE:
                self._history[source] = self._history[source][-MAX_HISTORY_PER_SOURCE:]
            
            # Обновляем индекс весов (для быстрого поиска)
            self._update_weight_index()
            
            # Иногда чистим старые данные (каждые 10 обновлений)
            if self.stats["updates"] % 10 == 0:
                self._cleanup_old()
            
            self.stats["updates"] += 1
            
            logger.debug(f"📝 SensorMemory обновлена: {source} (вес={weight:.2f}, "
                        f"возраст старого={old_snapshot.get_age() if old_snapshot else 0:.2f}с)")
            return f"{source}_{timestamp}"
    
    def get(self, source: str, min_weight: float = MIN_WEIGHT) -> Optional[SensorSnapshot]:
        """
        ПОЛУЧИТЬ ДАННЫЕ ОТ КОНКРЕТНОГО ИСТОЧНИКА
        
        Args:
            source: имя источника ("lidar", "vlm")
            min_weight: минимальный вес (данные с весом ниже не возвращаются)
        
        Returns:
            SensorSnapshot или None, если данных нет или они устарели
        
        Пример:
            lidar_data = memory.get("lidar")
            if lidar_data:
                print(f"Расстояние вперёд: {lidar_data.data['sectors']['front']}м")
        """
        with self.lock:
            snap = self._snapshots.get(source)
            if snap and snap.get_current_weight() > min_weight:
                return snap
            return None
    
    def get_all(self, min_weight: float = MIN_WEIGHT) -> List[SensorSnapshot]:
        """
        ПОЛУЧИТЬ ВСЕ АКТУАЛЬНЫЕ ДАННЫЕ ОТ ВСЕХ ИСТОЧНИКОВ
        
        Это метод, который использует агент для сбора контекста.
        Возвращает список всех снимков, отсортированный по весу (самые важные первые).
        
        Args:
            min_weight: минимальный вес (данные с весом ниже не возвращаются)
        
        Returns:
            List[SensorSnapshot] — список всех актуальных снимков
        
        Пример:
            all_data = memory.get_all(min_weight=0.3)
            for snap in all_data:
                print(f"{snap.source}: вес={snap.get_current_weight():.2f}")
        """
        with self.lock:
            result = []
            for snap in self._snapshots.values():
                if snap.get_current_weight() > min_weight:
                    result.append(snap)
            # Сортируем по весу (сначала самые важные)
            result.sort(key=lambda x: x.get_current_weight(), reverse=True)
            return result
    
    def get_summaries(self, min_weight: float = MIN_WEIGHT) -> Dict[str, Dict]:
        """
        ПОЛУЧИТЬ КРАТКИЕ ОПИСАНИЯ ДАННЫХ ДЛЯ ПРОМПТА
        
        ЭТО ГЛАВНЫЙ МЕТОД, КОТОРЫЙ ИСПОЛЬЗУЕТ АГЕНТ.
        Он возвращает не сырые данные, а уже отформатированные строки,
        готовые для вставки в промпт LLM.
        
        Args:
            min_weight: минимальный вес (данные с весом ниже не возвращаются)
        
        Returns:
            Словарь вида:
            {
                "lidar": {
                    "summary": "front=0.4м, front_left=0.5м, ... | ближе 80см: ...",
                    "weight": 0.95,
                    "age": 0.02,
                    "timestamp": 1712345720.45
                },
                "vlm": {...}
            }
        
        Пример использования в agent_v5.py:
        
            summaries = self.sensor_memory.get_summaries(min_weight=0.3)
            for source, info in summaries.items():
                prompt_line = f"{source}: {info['summary']} [вес: {info['weight']:.2f}, возраст: {info['age']:.2f}с]"
        """
        summaries = {}
        for snap in self.get_all(min_weight):
            summaries[snap.source] = {
                "summary": self._make_summary(snap.source, snap.data),
                "weight": snap.get_current_weight(),
                "age": snap.get_age(),
                "timestamp": snap.timestamp,
                "eta": snap.meta.get("eta") if snap.meta else None
            }
        return summaries
    
    # ================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    def _make_summary(self, source: str, data: Dict) -> str:
        """
        ПРЕВРАЩАЕТ СЫРЫЕ ДАННЫЕ В ЧЕЛОВЕКОЧИТАЕМУЮ СТРОКУ
        
        Это сердце форматирования для LLM.
        Каждый тип сенсора имеет свой формат вывода.
        
        ЧТО ВОЗВРАЩАЕТ:
        
            Лидар:    "front=0.4м, front_left=0.5м, left=2.0м, ... | ближе 80см: -5°;15° ~40см, 0.6м, v=1.2м/с ↓"
            VLM:      "сцена: коридор, объекты: человек 1.2м, путь: free"
            Одометрия: "vл=0.3, vп=0.3, курс=0°"
            Батарея:   "75% (зарядка)"
        """
        # ЛИДАР (8 секторов + ближние объекты)
        if source == "lidar":
            sectors = data.get("sectors", {})
            clusters = data.get("clusters", [])
            
            if sectors:
                # Все 8 секторов
                sector_str = ", ".join([f"{s}={d:.1f}м" for s, d in sectors.items()])
                
                # Ближние объекты (80см)
                close_clusters = [c for c in clusters if c.get('min_distance', 1.0) < 0.8]
                
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
                            f"{angle_start:.0f}°;{angle_end:.0f}° ~{dist:.0f}см, {size:.1f}м, v={speed:.1f}м/с{arrow_str}"
                        )
                    
                    return f"{sector_str} | ближе 80см: {', '.join(cluster_strs)}"
                return sector_str
            
            # Если нет секторов — 4 направления
            distances = data.get('distances', {})
            return f"перед={distances.get('front', 0):.1f}м, лев={distances.get('left', 0):.1f}м, прав={distances.get('right', 0):.1f}м"
        
        # VLM (визуальное понимание)
        elif source == "vlm":
            scene = data.get("scene", "?")
            objects = data.get("objects", [])
            path_status = data.get("path_status", "?")
            
            if objects:
                obj_str = ', '.join([f"{o.get('name', '?')} {o.get('distance', 0):.1f}м" for o in objects[:3]])
                return f"сцена: {scene}, объекты: {obj_str}, путь: {path_status}"
            return f"сцена: {scene}, путь: {path_status}"
        
        # ОДОМЕТРИЯ (скорость колёс и курс)
        elif source == "odometry":
            speed_left = data.get('speed_left', 0)
            speed_right = data.get('speed_right', 0)
            heading = data.get('heading', 0)
            return f"vл={speed_left:.1f}, vп={speed_right:.1f}, курс={heading:.0f}°"
        
        # БАТАРЕЯ
        elif source == "battery":
            level = data.get('level', 0)
            charging = data.get('charging', False)
            return f"{level}% {'(зарядка)' if charging else ''}"
        
        # ВСЁ ОСТАЛЬНОЕ
        return str(data)[:50]
    
    def _update_weight_index(self):
        """Обновляет индекс весов (для быстрого доступа к самым важным данным)"""
        self._weight_index = [
            (source, snap.get_current_weight())
            for source, snap in self._snapshots.items()
        ]
        self._weight_index.sort(key=lambda x: x[1], reverse=True)
    
    def _cleanup_old(self):
        """Удаляет данные, которые старше max_age"""
        now = time.time()
        to_remove = []
        for source, snap in self._snapshots.items():
            if now - snap.timestamp > self.max_age:
                to_remove.append(source)
        
        for source in to_remove:
            del self._snapshots[source]
            self.stats["cleaned"] += 1
        
        if to_remove:
            logger.debug(f"🧹 SensorMemory: удалено {len(to_remove)} источников (старше {self.max_age}с)")
    
    # ================================================================
    # МЕТОДЫ ДЛЯ ОТЛАДКИ И СОХРАНЕНИЯ
    # ================================================================
    
    def get_stats(self) -> Dict:
        """Возвращает статистику хранилища (для отладки)"""
        with self.lock:
            return {
                "active_sources": len(self._snapshots),
                "sources": list(self._snapshots.keys()),
                "weights": {s: snap.get_current_weight() for s, snap in self._snapshots.items()},
                "ages": {s: snap.get_age() for s, snap in self._snapshots.items()},
                "history_sizes": {s: len(h) for s, h in self._history.items()},
                "updates": self.stats["updates"],
                "cleaned": self.stats["cleaned"]
            }
    
    def save_to_file(self, filepath: str):
        """Сохраняет всё хранилище в файл (чтобы не потерять при перезагрузке)"""
        with self.lock:
            data = {
                "snapshots": {s: snap.to_dict() for s, snap in self._snapshots.items()},
                "history": {s: [h.to_dict() for h in hist[-10:]] for s, hist in self._history.items()},
                "stats": self.stats,
                "timestamp": time.time(),
                "version": "2.0"
            }
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 SensorMemory сохранена в {filepath} ({len(self._snapshots)} источников)")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения: {e}")
    
    def load_from_file(self, filepath: str):
        """Загружает хранилище из файла (после перезагрузки)"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                self._snapshots.clear()
                self._history.clear()
                
                for source, sdata in data.get("snapshots", {}).items():
                    self._snapshots[source] = SensorSnapshot.from_dict(sdata)
                
                for source, hist_data in data.get("history", {}).items():
                    history = [SensorSnapshot.from_dict(h) for h in hist_data]
                    self._history[source] = history[-MAX_HISTORY_PER_SOURCE:]
                
                self.stats = data.get("stats", self.stats)
                self._update_weight_index()
            
            logger.info(f"📂 SensorMemory загружена из {filepath} ({len(self._snapshots)} источников)")
        except FileNotFoundError:
            logger.info("🆕 Создана новая SensorMemory")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
    
    def clear(self):
        """Полностью очищает хранилище"""
        with self.lock:
            self._snapshots.clear()
            self._history.clear()
            self._weight_index.clear()
            self.stats = {
                "updates": 0,
                "cleaned": 0,
                "last_cleanup": time.time()
            }
        logger.info("🧹 SensorMemory очищена")


# ================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (КАК ЭТО РАБОТАЕТ В ЖИВОМ РОБОТЕ)
# ================================================================

if __name__ == "__main__":
    """
    ДЕМОНСТРАЦИЯ РАБОТЫ SENSOR MEMORY
    
    Запустите этот файл, чтобы увидеть, как:
        - Лидар и VLM асинхронно складывают данные
        - Агент забирает самые свежие
        - Веса затухают со временем
    """
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ SENSOR MEMORY")
    print("="*60 + "\n")
    
    # Создаём хранилище
    memory = SensorMemory(max_age=5.0)
    
    # Симулируем лидар (обновляется 10 раз в секунду)
    print("📡 Симуляция лидара: обновление каждые 0.1 секунды")
    for i in range(5):
        time.sleep(0.1)
        memory.update("lidar", {
            "sectors": {"front": 0.5, "left": 2.0, "right": 2.5},
            "clusters": []
        }, weight=0.95)
        print(f"   Лидар обновлён #{i+1}")
    
    # Симулируем VLM (обновляется 1 раз в секунду)
    print("\n👁️ Симуляция VLM: обновление каждую секунду")
    for i in range(2):
        time.sleep(1.0)
        memory.update("vlm", {
            "scene": "коридор",
            "objects": [{"name": "человек", "distance": 2.0}],
            "path_status": "free"
        }, weight=0.70)  # <-- ИСПРАВЛЕНО: теперь weight передаётся как keyword
        print(f"   VLM обновлён #{i+1}")
    
    # Агент забирает данные
    print("\n🤖 Агент забирает данные (через 0.5 секунды после последнего обновления):")
    time.sleep(0.5)
    
    summaries = memory.get_summaries(min_weight=0.3)
    for source, info in summaries.items():
        print(f"\n   📍 {source.upper()}:")
        print(f"      Данные: {info['summary']}")
        print(f"      Вес: {info['weight']:.2f}")
        print(f"      Возраст: {info['age']:.2f}с")
    
    # Показываем затухание веса
    print("\n⏳ Демонстрация затухания веса:")
    print("   (ждём 3 секунды, вес лидара должен упасть)\n")
    
    time.sleep(3)
    
    lidar = memory.get("lidar")
    if lidar:
        print(f"   Лидар после 3 секунд: вес = {lidar.get_current_weight():.2f} (было 0.95)")
    
    print("\n" + "="*60)
    print("Если вы видите эту строку — SensorMemory работает!")
    print("="*60 + "\n")
