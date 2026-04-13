#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/weight_calculator.py - РАЗРЕШЕНИЕ КОНФЛИКТОВ СЕНСОРОВ ВО ВРЕМЕНИ        ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это "судья", который решает, каким сенсорам верить прямо сейчас.           ║
║   Он не просто ставит оценку — он РАЗДЕЛЯЕТ ДАННЫЕ ПО ВРЕМЕНИ.               ║
║                                                                              ║
║ ГЛАВНАЯ ФУНКЦИЯ — РАЗРЕШЕНИЕ ВРЕМЕННЫХ КОНФЛИКТОВ:                           ║
║                                                                              ║
║   VLM сделал скан 0.5 секунды назад:                                         ║
║       "сцена: коридор, объекты: человек 2.0м слева, путь: free"              ║
║                                                                              ║
║   Лидар сделал скан 0.1 секунды назад:                                       ║
║       "перед=0.4м! движущийся объект 15°;45° ~40см, v=1.2м/с →"              ║
║                                                                              ║
║   WeightCalculator присваивает:                                               ║
║       - VLM: вес 0.65, возраст 0.50с                                         ║
║       - Лидар: вес 0.95, возраст 0.02с                                       ║
║                                                                              ║
║   LLM видит ОБА сенсора в промпте и ПОНИМАЕТ:                                 ║
║       "Человек, которого VLM видел слева 0.5с назад,                         ║
║        только что переместился вперёд и теперь представляет угрозу!"         ║
║                                                                              ║
║   БЕЗ ВРЕМЕННЫХ МЕТОК LLM увидела бы:                                         ║
║       "VLM: человек слева. Лидар: препятствие впереди."                       ║
║       И не поняла бы, что это ОДИН И ТОТ ЖЕ объект в движении.               ║
║                                                                              ║
║ ДОПОЛНИТЕЛЬНО — ОБНАРУЖЕНИЕ АНОМАЛИЙ:                                        ║
║                                                                              ║
║   Ситуация 1: Лидар забрызган грязью.                                        ║
║       → 30-50% секторов показывают почти одинаковое расстояние.              ║
║       → WeightCalculator СНИЖАЕТ ВЕС до 0.5-0.6.                              ║
║       → LLM видит пометку "АНОМАЛИЯ" и полагается на другие сенсоры.         ║
║                                                                              ║
║   Ситуация 2: VLM ослеплён солнцем.                                          ║
║       → VLM возвращает "неизвестно" или пустой список объектов.              ║
║       → WeightCalculator СНИЖАЕТ ВЕС в 2 раза.                                ║
║       → LLM игнорирует VLM и едет по лидару.                                  ║
║                                                                              ║
║   Ситуация 3: Сенсор передаёт данные по медленному BLE.                      ║
║       → Данные приходят с большой задержкой.                                  ║
║       → WeightCalculator ШТРАФУЕТ за латентность.                             ║
║       → LLM предпочитает более быстрые сенсоры.                               ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ (ФОРМУЛА):                                                  ║
║                                                                              ║
║   Итоговый вес = БазовыйВес × Возраст × Приоритет × Достоверность ×          ║
║                  ИсторическаяНадёжность × ШтрафЗаЛатентность × Аномалия      ║
║                                                                              ║
║   ВАЖНО:                                                                     ║
║   - Вес НИЖЕ 0.3 — агент НЕ ВИДИТ эти данные                                 ║
║   - Возраст передаётся в LLM явно: "[вес: 0.65, возраст: 0.50с]"             ║
║   - LLM сама решает, как интерпретировать временной конфликт                 ║
║                                                                              ║
║ ПРИМЕР ТОГО, ЧТО ВИДИТ LLM В ПРОМПТЕ:                                        ║
║                                                                              ║
║   ДАННЫЕ СЕНСОРОВ (по важности):                                             ║
║     • lidar: front=0.4м, left=2.0м, right=2.5м | ближе 80см: 15°;45° ~40см,  ║
║              0.6м, v=1.2м/с → [вес: 0.95, возраст: 0.02с]                    ║
║     • vlm: сцена: коридор, объекты: человек 2.0м, путь: free                 ║
║            [вес: 0.65, возраст: 0.50с]                                       ║
║     • odometry: vл=0.3, vп=0.3, курс=0° [вес: 0.68, возраст: 0.04с]         ║
║                                                                              ║
║   LLM видит: "Лидар свежий (0.02с) и показывает объект впереди.              ║
║               VLM устарел (0.50с) и показывал человека слева.                ║
║               Вероятно, человек переместился. Нужно остановиться!"           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Импортируем конфигурацию, если доступна
try:
    from config import (
        DECAY_FACTOR, MIN_WEIGHT, LIDAR_ANOMALY_THRESHOLD,
        LIDAR_ANOMALY_MIN_DIST, VLM_ANOMALY_EMPTY_FACTOR,
        LATENCY_ALPHA, LATENCY_DEV_FACTOR, INITIAL_WEIGHTS
    )
except ImportError:
    # Значения по умолчанию (если config.py не используется)
    DECAY_FACTOR = 0.9
    MIN_WEIGHT = 0.1
    LIDAR_ANOMALY_THRESHOLD = 0.05      # 5% разброс = аномалия
    LIDAR_ANOMALY_MIN_DIST = 0.5        # проверяем только если расстояние > 0.5м
    LIDAR_ANOMALY_SECTOR_RATIO = 0.3    # если 30% секторов аномальны → вес 0.6
    LIDAR_ANOMALY_HEAVY_RATIO = 0.5     # если 50% секторов аномальны → вес 0.5
    VLM_ANOMALY_EMPTY_FACTOR = 0.5      # коэффициент при пустом ответе VLM
    LATENCY_ALPHA = 0.1                 # скользящее среднее для латентности
    LATENCY_DEV_FACTOR = {1.0: 1.0, 2.0: 0.9, 3.0: 0.7}
    INITIAL_WEIGHTS = {
        "tinyml": 1.0, "lidar": 0.95, "vlm": 0.70, "llm": 0.50,
        "operator": 0.80, "esp12": 0.70, "agent": 0.85,
        "odometry": 0.65, "vision_memory": 0.75, "unknown": 0.50
    }

logger = logging.getLogger(__name__)


class WeightCalculator:
    """
    ДИНАМИЧЕСКИЙ РАСЧЁТ ВЕСОВ С УЧЁТОМ ВРЕМЕНИ, АНОМАЛИЙ И ЛАТЕНТНОСТИ
    
    Главная задача — дать LLM возможность видеть ВРЕМЕННУЮ ДИНАМИКУ сцены.
    Старые данные имеют меньший вес, но НЕ удаляются — LLM сама решает,
    как интерпретировать конфликт между "старым VLM" и "свежим лидаром".
    """
    
    def __init__(self, decay_factor: float = DECAY_FACTOR):
        self.decay_factor = decay_factor
        self.base_weights = INITIAL_WEIGHTS.copy()
        
        # Историческая достоверность источников (накапливается со временем)
        self.source_reliability = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "avg_latency": 0.0
        })
        
        # Статистика аномалий (как часто сенсор "врёт")
        self.anomaly_stats = defaultdict(lambda: {
            "anomalies": 0,
            "total": 0,
            "anomaly_rate": 0.0,
            "last_anomaly": 0
        })
        
        # Статистика задержек (скользящее среднее и std)
        self.latency_stats = defaultdict(lambda: {
            "avg": 0.5,
            "std": 0.1,
            "samples": 0,
            "last_latency": 0.5
        })
        
        logger.info("✅ WeightCalculator инициализирован")
    
    # ================================================================
    # ГЛАВНЫЙ МЕТОД — РАСЧЁТ ВЕСА
    # ================================================================
    
    def calculate(self, 
                 source_type: str,
                 timestamp: float,
                 priority: int = 5,
                 confidence: float = 1.0,
                 latency: Optional[float] = None,
                 data: Optional[Dict] = None) -> float:
        """
        РАССЧИТЫВАЕТ ИТОГОВЫЙ ВЕС С УЧЁТОМ ВСЕХ ФАКТОРОВ
        
        Args:
            source_type: тип источника ("lidar", "vlm", "odometry")
            timestamp: когда данные были получены (time.time())
            priority: приоритет (1-10, чем меньше — тем важнее)
            confidence: заявленная достоверность от сенсора (0-1)
            latency: задержка передачи данных (сек). None = не учитывать
            data: сырые данные для анализа аномалий
        
        Returns:
            float: вес от 0.1 до 1.5
            
        ПРИМЕР:
        
            # Лидар только что прислал данные
            weight = wc.calculate("lidar", timestamp=time.time())
            # → 0.95 (базовый вес)
            
            # VLM прислал данные с задержкой 1.5 сек (медленный BLE)
            weight = wc.calculate("vlm", timestamp=time.time(), latency=1.5)
            # → ~0.60 (снижен из-за латентности)
        """
        # 1. Базовый вес (из конфига)
        base = self.base_weights.get(source_type, 0.5)
        
        # 2. Возраст (чем старше, тем меньше вес)
        age = time.time() - timestamp
        age_factor = self.decay_factor ** age
        
        # 3. Приоритет (1-10 → 1.0-0.55)
        priority_factor = 1.0 - (priority - 1) * 0.05
        
        # 4. Историческая достоверность (как часто сенсор был прав)
        historical_reliability = self._get_historical_reliability(source_type)
        
        # 5. Штраф за латентность (если данные шли долго)
        latency_factor = 1.0
        if latency is not None:
            latency_factor = self._evaluate_latency(source_type, latency)
        
        # 6. Обнаружение аномалий (грязный лидар, ослеплённый VLM)
        anomaly_factor = 1.0
        if data is not None:
            anomaly_detected, anomaly_factor = self._detect_anomalies(source_type, data)
            if anomaly_detected:
                logger.debug(f"⚠️ Аномалия в {source_type}, коэффициент {anomaly_factor:.2f}")
                self._update_anomaly_stats(source_type, success=False)
            else:
                self._update_anomaly_stats(source_type, success=True)
        
        # 7. Итоговый вес
        weight = (base * age_factor * priority_factor * confidence * 
                  historical_reliability * latency_factor * anomaly_factor)
        
        return max(MIN_WEIGHT, min(1.5, weight))
    
    def process_with_meta(self,
                          source_type: str,
                          data: Dict,
                          timestamp: float,
                          priority: int = 5,
                          confidence: float = 1.0,
                          latency: Optional[float] = None) -> Dict:
        """
        ОБРАБАТЫВАЕТ ДАННЫЕ И ВОЗВРАЩАЕТ ОБОГАЩЁННУЮ ВЕРСИЮ С МЕТАДАННЫМИ
        
        Это основной метод для сенсорных данных. Он не только считает вес,
        но и добавляет в данные поля _meta с возрастом, ETA и флагами аномалий.
        
        Returns:
            Словарь data с добавленным полем _meta:
            {
                "_meta": {
                    "weight": 0.85,
                    "age": 0.12,
                    "eta": {"cluster_1": 2.5},  # только для лидара
                    "anomaly_factor": 1.0
                }
            }
        """
        weight = self.calculate(
            source_type=source_type,
            timestamp=timestamp,
            priority=priority,
            confidence=confidence,
            latency=latency,
            data=data
        )
        
        # Вычисляем ETA (время до столкновения) для движущихся объектов
        eta = None
        if source_type == "lidar":
            eta = self._calculate_eta(data)
        
        # Обогащаем данные
        enriched = {
            **data,
            "_meta": {
                "source": source_type,
                "original_timestamp": timestamp,
                "weight": weight,
                "age": time.time() - timestamp,
                "eta": eta,
                "anomaly_factor": self._get_anomaly_factor(source_type)
            }
        }
        
        # Обновляем статистику задержки
        if latency is not None:
            self._update_latency_stats(source_type, latency)
        
        return enriched
    
    # ================================================================
    # ОБНАРУЖЕНИЕ АНОМАЛИЙ
    # ================================================================
    
    def _detect_anomalies(self, source_type: str, data: Dict) -> Tuple[bool, float]:
        """
        ОПРЕДЕЛЯЕТ, ЕСТЬ ЛИ АНОМАЛИЯ В ДАННЫХ
        
        Returns:
            (обнаружена_ли_аномалия, коэффициент_снижения_веса)
        """
        if source_type == "lidar":
            return self._detect_lidar_anomaly(data)
        elif source_type == "vlm":
            return self._detect_vlm_anomaly(data)
        elif source_type == "odometry":
            return self._detect_odometry_anomaly(data)
        
        return False, 1.0
    
    def _detect_lidar_anomaly(self, data: Dict) -> Tuple[bool, float]:
        """
        ОБНАРУЖИВАЕТ ГРЯЗНЫЙ ЛИДАР
        
        Признак грязного лидара: несколько секторов показывают почти одинаковое
        расстояние (разброс < 5%). В норме сектора должны показывать РАЗНЫЕ
        расстояния (стена ближе, проход дальше).
        
        ЛОГИКА:
            - Если 30-50% секторов аномальны → вес 0.6
            - Если >50% секторов аномальны → вес 0.5
            - Иначе → нет аномалии (вес 1.0)
        """
        sectors = data.get("sectors", {})
        
        if not sectors or len(sectors) < 4:
            return False, 1.0
        
        # Собираем все расстояния по секторам
        distances = list(sectors.values())
        
        # Убираем очень дальние (>5м) — они не показательны для грязи
        valid_dist = [d for d in distances if d < 5.0 and d > LIDAR_ANOMALY_MIN_DIST]
        
        if len(valid_dist) < 3:
            return False, 1.0
        
        # Считаем, сколько секторов "аномально близки" друг к другу
        mean_dist = np.mean(valid_dist)
        std_dist = np.std(valid_dist)
        
        if mean_dist == 0:
            return False, 1.0
        
        # Коэффициент вариации (чем меньше, тем подозрительнее)
        cv = std_dist / mean_dist
        
        # Считаем долю секторов, которые попадают в узкий диапазон
        close_count = 0
        for d in valid_dist:
            if abs(d - mean_dist) / mean_dist < LIDAR_ANOMALY_THRESHOLD:
                close_count += 1
        
        anomaly_ratio = close_count / len(valid_dist)
        
        if anomaly_ratio >= LIDAR_ANOMALY_HEAVY_RATIO:
            logger.debug(f"Лидар: {anomaly_ratio:.0%} секторов аномальны (грязь). Вес снижен до 0.5")
            return True, 0.5
        elif anomaly_ratio >= LIDAR_ANOMALY_SECTOR_RATIO:
            logger.debug(f"Лидар: {anomaly_ratio:.0%} секторов аномальны. Вес снижен до 0.6")
            return True, 0.6
        
        return False, 1.0
    
    def _detect_vlm_anomaly(self, data: Dict) -> Tuple[bool, float]:
        """
        ОБНАРУЖИВАЕТ ОСЛЕПЛЁННЫЙ VLM
        
        Признаки:
            - Сцена "неизвестно" и нет объектов
            - Пустой список объектов при сцене "неизвестно"
        """
        scene = data.get("scene", "")
        objects = data.get("objects", [])
        
        if scene == "неизвестно" and not objects:
            logger.debug("VLM: пустой ответ (ослеплён?). Вес снижен вдвое")
            return True, VLM_ANOMALY_EMPTY_FACTOR
        
        return False, 1.0
    
    def _detect_odometry_anomaly(self, data: Dict) -> Tuple[bool, float]:
        """Обнаруживает аномалии в одометрии (пока базовая реализация)"""
        return False, 1.0
    
    # ================================================================
    # ЛАТЕНТНОСТЬ (ШТРАФ ЗА МЕДЛЕННУЮ ПЕРЕДАЧУ)
    # ================================================================
    
    def _evaluate_latency(self, source_type: str, latency: float) -> float:
        """
        ОЦЕНИВАЕТ, НАСКОЛЬКО ЗАДЕРЖКА ВЛИЯЕТ НА ДОСТОВЕРНОСТЬ
        
        Если задержка сильно выше средней для этого сенсора — вес снижается.
        Это важно для BLE-сенсоров, которые медленнее Wi-Fi.
        """
        stats = self.latency_stats[source_type]
        
        if stats["samples"] == 0:
            stats["avg"] = latency
            stats["std"] = latency * 0.1
        else:
            # Экспоненциальное скользящее среднее
            alpha = LATENCY_ALPHA
            stats["avg"] = alpha * latency + (1 - alpha) * stats["avg"]
            stats["std"] = alpha * abs(latency - stats["avg"]) + (1 - alpha) * stats["std"]
        
        stats["samples"] += 1
        stats["last_latency"] = latency
        
        if stats["std"] < 0.01:
            return 1.0
        
        deviation = abs(latency - stats["avg"]) / stats["std"]
        
        # Определяем коэффициент по отклонению
        if deviation < 1:
            return 1.0
        elif deviation < 2:
            return 0.9
        elif deviation < 3:
            return 0.7
        else:
            return 0.4
    
    def _update_latency_stats(self, source_type: str, latency: float):
        """Обновляет статистику задержки (вызывается автоматически)"""
        # Уже обновлено в _evaluate_latency
        pass
    
    # ================================================================
    # ETA — ВРЕМЯ ДО СТОЛКНОВЕНИЯ (ТОЛЬКО ДЛЯ ЛИДАРА)
    # ================================================================
    
    def _calculate_eta(self, data: Dict) -> Optional[Dict[str, float]]:
        """
        ВЫЧИСЛЯЕТ ВРЕМЯ ДО СТОЛКНОВЕНИЯ С ДВИЖУЩИМИСЯ ОБЪЕКТАМИ
        
        Используется только для лидара. VLM не предсказывает столкновения.
        
        Returns:
            {"cluster_1": 2.5} — через сколько секунд столкновение
            или None, если движущихся объектов нет
        """
        clusters = data.get("clusters", [])
        eta = {}
        
        for cluster in clusters:
            distance = cluster.get("min_distance", 0)
            vel = cluster.get("velocity", {})
            vel_x = vel.get("x", 0)
            speed = cluster.get("speed", 0)
            
            # Движется навстречу (отрицательная скорость по X)
            if vel_x < 0 and distance > 0 and speed > 0.05:
                ttc = distance / abs(vel_x)
                cluster_id = cluster.get('id', '?')
                eta[f"cluster_{cluster_id}"] = round(ttc, 2)
        
        return eta if eta else None
    
    # ================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    def _get_historical_reliability(self, source_type: str) -> float:
        """Возвращает историческую достоверность источника (0.6-1.0)"""
        stats = self.source_reliability.get(source_type, {})
        if stats.get("total", 0) == 0:
            return 1.0
        
        success_rate = stats["success"] / stats["total"]
        return 0.6 + 0.4 * success_rate
    
    def _get_anomaly_factor(self, source_type: str) -> float:
        """Возвращает текущий коэффициент аномальности (1.0 = нет аномалий)"""
        stats = self.anomaly_stats.get(source_type, {})
        if stats.get("total", 0) == 0:
            return 1.0
        return 1.0 - stats.get("anomaly_rate", 0)
    
    def _update_anomaly_stats(self, source_type: str, success: bool):
        """Обновляет статистику аномалий для источника"""
        stats = self.anomaly_stats[source_type]
        stats["total"] += 1
        if not success:
            stats["anomalies"] += 1
            stats["anomaly_rate"] = stats["anomalies"] / stats["total"]
            stats["last_anomaly"] = time.time()
    
    def update_reliability(self, source_type: str, success: bool, latency: float):
        """
        ОБНОВЛЯЕТ СТАТИСТИКУ ИСТОЧНИКА ПОСЛЕ ВЫПОЛНЕНИЯ ДЕЙСТВИЯ
        
        Вызывается агентом, когда становится известно, было ли действие
        успешным. Например, если робот проехал и не врезался — лидар был прав.
        """
        stats = self.source_reliability[source_type]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        stats["avg_latency"] = (stats["avg_latency"] * (stats["total"] - 1) + latency) / stats["total"]
    
    def get_source_stats(self, source_type: str) -> Dict:
        """Возвращает полную статистику по источнику (для отладки)"""
        reliability = self.source_reliability.get(source_type, {})
        latency = self.latency_stats.get(source_type, {})
        anomaly = self.anomaly_stats.get(source_type, {})
        
        return {
            "reliability": {
                "total": reliability.get("total", 0),
                "success": reliability.get("success", 0),
                "success_rate": reliability.get("success", 0) / max(1, reliability.get("total", 1))
            },
            "latency": {
                "avg": latency.get("avg", 0),
                "std": latency.get("std", 0),
                "samples": latency.get("samples", 0)
            },
            "anomaly": {
                "rate": anomaly.get("anomaly_rate", 0),
                "total": anomaly.get("total", 0),
                "last": anomaly.get("last_anomaly", 0)
            }
        }
    
    def adjust_base_weight(self, source_type: str, new_weight: float):
        """
        КОРРЕКТИРУЕТ БАЗОВЫЙ ВЕС ИСТОЧНИКА (ДЛЯ ДОЛГОСРОЧНОГО ОБУЧЕНИЯ)
        
        Если сенсор постоянно врёт, его базовый вес можно снизить.
        Если сенсор отличный — повысить.
        """
        if source_type in self.base_weights:
            old = self.base_weights[source_type]
            self.base_weights[source_type] = old * 0.9 + new_weight * 0.1
            self.base_weights[source_type] = max(0.3, min(1.5, self.base_weights[source_type]))
            logger.debug(f"⚖️ Базовый вес {source_type}: {old:.2f} → {self.base_weights[source_type]:.2f}")


# ================================================================
# ДЕМОНСТРАЦИЯ РАБОТЫ (КАК ЭТО ВЫГЛЯДИТ В ЖИВОМ РОБОТЕ)
# ================================================================

if __name__ == "__main__":
    """
    ДЕМОНСТРАЦИЯ РАБОТЫ WEIGHT CALCULATOR
    
    Показывает:
        - Как вес падает со временем
        - Как грязный лидар получает штраф
        - Как ослеплённый VLM получает штраф
        - Как медленный BLE снижает доверие
    """
    
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ WEIGHT CALCULATOR")
    print("="*60 + "\n")
    
    wc = WeightCalculator()
    
    # --------------------------------------------------------------
    # 1. НОРМАЛЬНАЯ РАБОТА: ЛИДАР ЧИСТЫЙ, VLM ВИДИТ
    # --------------------------------------------------------------
    print("📡 1. НОРМАЛЬНАЯ РАБОТА (все сенсоры в порядке)")
    print("-" * 40)
    
    now = time.time()
    
    # Чистый лидар (сектора показывают РАЗНЫЕ расстояния)
    clean_lidar_data = {
        "sectors": {
            "front": 0.4, "front_left": 2.0, "left": 3.5, "back_left": 5.0,
            "back": 6.0, "back_right": 5.5, "right": 2.8, "front_right": 1.5
        }
    }
    lidar_weight = wc.calculate("lidar", timestamp=now, data=clean_lidar_data)
    print(f"   Чистый лидар: вес = {lidar_weight:.2f}")
    
    # VLM видит сцену
    good_vlm_data = {
        "scene": "коридор",
        "objects": [{"name": "человек", "distance": 2.0}]
    }
    vlm_weight = wc.calculate("vlm", timestamp=now, data=good_vlm_data)
    print(f"   VLM видит: вес = {vlm_weight:.2f}")
    
    # --------------------------------------------------------------
    # 2. ГРЯЗНЫЙ ЛИДАР (30% секторов показывают почти одинаково)
    # --------------------------------------------------------------
    print("\n🧹 2. ГРЯЗНЫЙ ЛИДАР (30% секторов аномально близки)")
    print("-" * 40)
    
    dirty_lidar_data = {
        "sectors": {
            "front": 1.2, "front_left": 1.2, "left": 1.2,  # аномалия!
            "back_left": 5.0, "back": 6.0, "back_right": 5.5,
            "right": 2.8, "front_right": 1.5
        }
    }
    dirty_weight = wc.calculate("lidar", timestamp=now, data=dirty_lidar_data)
    print(f"   Грязный лидар (30%): вес = {dirty_weight:.2f} (было 0.95)")
    
    dirty_lidar_data_heavy = {
        "sectors": {
            "front": 1.2, "front_left": 1.2, "left": 1.2, "back_left": 1.2,  # 50%!
            "back": 6.0, "back_right": 5.5, "right": 2.8, "front_right": 1.5
        }
    }
    dirty_weight_heavy = wc.calculate("lidar", timestamp=now, data=dirty_lidar_data_heavy)
    print(f"   Грязный лидар (50%): вес = {dirty_weight_heavy:.2f}")
    
    # --------------------------------------------------------------
    # 3. ОСЛЕПЛЁННЫЙ VLM
    # --------------------------------------------------------------
    print("\n☀️ 3. ОСЛЕПЛЁННЫЙ VLM (ничего не видит)")
    print("-" * 40)
    
    blind_vlm_data = {
        "scene": "неизвестно",
        "objects": []
    }
    blind_weight = wc.calculate("vlm", timestamp=now, data=blind_vlm_data)
    print(f"   Ослеплённый VLM: вес = {blind_weight:.2f} (было 0.70)")
    
    # --------------------------------------------------------------
    # 4. ЗАТУХАНИЕ ВЕСА СО ВРЕМЕНЕМ
    # --------------------------------------------------------------
    print("\n⏳ 4. ЗАТУХАНИЕ ВЕСА СО ВРЕМЕНЕМ")
    print("-" * 40)
    
    old_timestamp = now - 2.0  # данные 2 секунды назад
    old_weight = wc.calculate("lidar", timestamp=old_timestamp, data=clean_lidar_data)
    print(f"   Лидар (возраст 2.0с): вес = {old_weight:.2f}")
    
    older_timestamp = now - 5.0  # данные 5 секунд назад
    older_weight = wc.calculate("lidar", timestamp=older_timestamp, data=clean_lidar_data)
    print(f"   Лидар (возраст 5.0с): вес = {older_weight:.2f}")
    
    # --------------------------------------------------------------
    # 5. МЕДЛЕННЫЙ BLE (ШТРАФ ЗА ЛАТЕНТНОСТЬ)
    # --------------------------------------------------------------
    print("\n🐌 5. МЕДЛЕННЫЙ BLE (штраф за задержку)")
    print("-" * 40)
    
    # Симулируем несколько быстрых передач (Wi-Fi)
    for _ in range(5):
        wc.calculate("lidar", timestamp=time.time(), latency=0.05, data=clean_lidar_data)
    
    # А теперь — медленная передача по BLE
    ble_weight = wc.calculate("lidar", timestamp=time.time(), latency=0.5, data=clean_lidar_data)
    print(f"   Лидар по BLE (задержка 0.5с): вес = {ble_weight:.2f}")
    
    very_slow_ble = wc.calculate("lidar", timestamp=time.time(), latency=1.5, data=clean_lidar_data)
    print(f"   Лидар по BLE (задержка 1.5с): вес = {very_slow_ble:.2f}")
    
    # --------------------------------------------------------------
    # 6. ЧТО ВИДИТ LLM В ПРОМПТЕ (пример форматирования)
    # --------------------------------------------------------------
    print("\n📋 6. ЧТО ВИДИТ LLM В ПРОМПТЕ")
    print("-" * 40)
    
    # Эмулируем формат, который ContextBuilder передаёт в LLM
    lidar_meta = wc.process_with_meta("lidar", clean_lidar_data, timestamp=now - 0.02)
    vlm_meta = wc.process_with_meta("vlm", good_vlm_data, timestamp=now - 0.50)
    
    print(f"""
    ДАННЫЕ СЕНСОРОВ (по важности):
      • lidar: front=0.4м, left=2.0м, right=2.5м | ближе 80см: 15°;45° ~40см, v=1.2м/с →
               [вес: {lidar_meta['_meta']['weight']:.2f}, возраст: {lidar_meta['_meta']['age']:.2f}с]
      • vlm: сцена: коридор, объекты: человек 2.0м, путь: free
             [вес: {vlm_meta['_meta']['weight']:.2f}, возраст: {vlm_meta['_meta']['age']:.2f}с]
    
    LLM видит: "VLM 0.5с назад показывал человека слева.
                Лидар сейчас показывает объект впереди.
                Вероятно, человек переместился. Нужно остановиться!"
    """)
    
    print("="*60)
    print("Демонстрация завершена.")
    print("="*60 + "\n")
