#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ sensors/lidar_processor.py - ПРОСТРАНСТВЕННОЕ ВОСПРИЯТИЕ (ПЕРИФЕРИЙНОЕ ЗРЕНИЕ)║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Сенсор геометрического восприятия. Превращает облако точек в структуру:    ║
║   8 секторов, кластеры объектов, скорости, направления движения.             ║
║                                                                              ║
║ ФИЛОСОФИЯ — ПЕРИФЕРИЙНОЕ ЗРЕНИЕ, А НЕ SLAM:                                  ║
║                                                                              ║
║   Лидар НЕ строит глобальную карту. Он работает как периферийное зрение      ║
║   человека: видит всё вокруг, но без деталей.                                ║
║                                                                              ║
║   • Замечает движение (что-то быстро приближается слева)                     ║
║   • Оценивает угрозу (время до столкновения)                                 ║
║   • Находит точки интереса — куда повернуть камеру (VLM) для фокусировки     ║
║                                                                              ║
║   VLM потом идентифицирует объект: «это человек», «это автомобиль».          ║
║   Такое разделение экономит ресурсы и работает быстрее классического SLAM.   ║
║                                                                              ║
║ БУДУЩЕЕ — ПОДКЛЮЧЕНИЕ SLAM (ОПЦИОНАЛЬНО):                                   ║
║                                                                              ║
║   Для сценариев, где нужна глобальная карта (склад, квартира,                ║
║   режим «исследовать локацию»), можно подключить SLAM как отдельный модуль.  ║
║                                                                              ║
║   Лидар при этом продолжает работать как периферийное зрение,                ║
║   а SLAM строит карту в фоне. Данные с лидара идут в ОБА модуля.             ║
║                                                                              ║
║   На улице, где окружение динамично, SLAM отключается — робот использует     ║
║   только периферийное зрение + GPS/одометрию.                                ║
║                                                                              ║
║ ЧТО ДАЁТ:                                                                    ║
║   • 8 секторов — минимальные расстояния по направлениям                      ║
║   • Кластеры объектов — угловой размер, дистанция, размер                    ║
║   • Скорость и направление движения (↑ ↓ ← → ↖ ↗ ↙ ↘)                        ║
║   • Время до столкновения (ETA) для движущихся объектов                      ║
║   • Оценка качества данных (для WeightCalculator)                            ║
║                                                                              ║
║ ФОРМАТ ДЛЯ LLM (через ContextBuilder):                                       ║
║   "front=0.4м, left=2.0м, right=2.5м | ближе 80см: 15°;45° ~40см, v=1.2м/с →"║
║                                                                              ║
║ ТРЕБОВАНИЯ:                                                                  ║
║   • numpy                                                                   ║
║   • scikit-learn (DBSCAN). Если нет — кластеризация отключается.             ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║   • cluster_threshold — порог кластеризации (метры)                          ║
║   • max_tracking_age — сколько секунд помнить объект                         ║
║   • DANGER_THRESHOLDS — опасные дистанции по секторам                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json

logger = logging.getLogger(__name__)

# ==================== ПРОВЕРКА НАЛИЧИЯ SKLEARN ====================
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("⚠️ sklearn не установлен, кластеризация лидара будет примитивной")
    
    class DBSCAN:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, X):
            return np.zeros(len(X))


# ==================== СЕКТОРЫ ЛИДАРА ====================

LIDAR_SECTORS = {
    "front": (-15, 15),
    "front_left": (15, 45),
    "left": (45, 90),
    "back_left": (90, 135),
    "back": (135, 225),
    "back_right": (225, 270),
    "right": (270, 315),
    "front_right": (315, 345)
}

DANGER_THRESHOLDS = {
    "front": 0.5,
    "front_left": 0.4,
    "front_right": 0.4,
    "left": 0.3,
    "right": 0.3,
    "back_left": 0.2,
    "back_right": 0.2,
    "back": 0.2
}

# Порог для отправки кластеров в LLM (метры)
CLUSTER_LLM_THRESHOLD = 0.8  # 80см


# ==================== СТРУКТУРЫ ДАННЫХ ====================

@dataclass
class Point3D:
    """Точка в облаке лидара"""
    x: float
    y: float
    z: float
    intensity: float = 0.0
    timestamp_us: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "i": self.intensity}


@dataclass
class ObjectCluster:
    """Кластер точек, представляющий объект"""
    id: int
    points: List[Point3D]
    center_x: float
    center_y: float
    center_z: float
    width: float
    depth: float
    height: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    speed: float = 0.0
    confidence: float = 1.0
    object_type: str = "unknown"
    first_seen: float = 0.0
    last_seen: float = 0.0
    min_distance: float = 0.0      # минимальная дистанция до кластера
    angle_start: float = 0.0       # начальный угол кластера (градусы)
    angle_end: float = 0.0         # конечный угол кластера (градусы)
    
    def __post_init__(self):
        """Вычислить min_distance и угловой диапазон после создания"""
        if self.points:
            # Минимальная дистанция
            distances = [np.sqrt(p.x**2 + p.y**2) for p in self.points]
            self.min_distance = min(distances)
            
            # Угловой диапазон
            angles = [np.degrees(np.arctan2(p.y, p.x)) for p in self.points]
            angles = [a if a >= 0 else a + 360 for a in angles]
            self.angle_start = min(angles)
            self.angle_end = max(angles)
    
    def _get_direction_arrow(self) -> str:
        """Определить стрелку направления движения (8 направлений)"""
        if self.speed <= 0.05:
            return ""
        
        # Угол движения объекта в градусах
        vel_angle = np.degrees(np.arctan2(self.velocity_y, self.velocity_x))
        vel_angle = (vel_angle + 360) % 360
        
        # 8 направлений: 0° = ↑ (прямо), 90° = → (вправо) и т.д.
        if 337.5 <= vel_angle or vel_angle < 22.5:
            return "↑"      # прямо
        elif 22.5 <= vel_angle < 67.5:
            return "↗"      # право-вперед
        elif 67.5 <= vel_angle < 112.5:
            return "→"      # вправо
        elif 112.5 <= vel_angle < 157.5:
            return "↘"      # право-назад
        elif 157.5 <= vel_angle < 202.5:
            return "↓"      # назад
        elif 202.5 <= vel_angle < 247.5:
            return "↙"      # лево-назад
        elif 247.5 <= vel_angle < 292.5:
            return "←"      # влево
        elif 292.5 <= vel_angle < 337.5:
            return "↖"      # лево-вперед
        
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "center": {"x": self.center_x, "y": self.center_y, "z": self.center_z},
            "size": {"w": self.width, "d": self.depth, "h": self.height},
            "velocity": {"x": self.velocity_x, "y": self.velocity_y},
            "speed": self.speed,
            "type": self.object_type,
            "confidence": self.confidence,
            "point_count": len(self.points),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "min_distance": self.min_distance,
            "angle_start": self.angle_start,
            "angle_end": self.angle_end
        }
    
    def to_llm_string(self) -> str:
        """Форматировать для вывода в промпт LLM (компактный формат)"""
        arrow = self._get_direction_arrow()
        arrow_str = f" {arrow}" if arrow else ""
        
        return (f"{self.angle_start:.0f}°;{self.angle_end:.0f}° "
                f"~{self.min_distance*100:.0f}см, "
                f"{max(self.width, self.depth):.1f}м, "
                f"v={self.speed:.1f}м/с{arrow_str}")


@dataclass
class ProcessedLidarData:
    """Обработанные данные лидара для протокола"""
    timestamp: float
    timestamp_us: int
    source_id: str
    scan_id: str
    
    front_distance: float
    left_distance: float
    right_distance: float
    back_distance: float
    
    sectors: Dict[str, float] = field(default_factory=dict)
    clusters: List[ObjectCluster] = field(default_factory=list)
    
    path_clear: bool
    obstacle_front: bool
    obstacle_left: bool
    obstacle_right: bool
    obstacle_back: bool
    
    moving_objects: List[Dict[str, Any]]
    points_count: int
    data_quality: float
    weight: float = 0.95
    age: float = 0.0
    
    def to_protocol_message(self) -> Dict[str, Any]:
        return {
            "version": "5.0",
            "message_id": self.scan_id,
            "timestamp": self.timestamp,
            "timestamp_us": self.timestamp_us,
            "source": self.source_id,
            "source_type": "lidar",
            "capability": "sensor.lidar.scan",
            "type": "data",
            "data": {
                "distances": {
                    "front": self.front_distance,
                    "left": self.left_distance,
                    "right": self.right_distance,
                    "back": self.back_distance
                },
                "sectors": self.sectors,
                "obstacles": {
                    "front": self.obstacle_front,
                    "left": self.obstacle_left,
                    "right": self.obstacle_right,
                    "back": self.obstacle_back
                },
                "path_clear": self.path_clear,
                "clusters": [c.to_dict() for c in self.clusters],
                "moving_objects": self.moving_objects,
                "quality": self.data_quality,
                "points_count": self.points_count
            },
            "weight": self.weight,
            "metadata": {
                "priority": 2,
                "ttl": 1.0,
                "compressed": False
            }
        }
    
    def format_for_llm(self) -> str:
        """Отформатировать для вставки в промпт LLM"""
        lines = []
        
        # Все 8 секторов
        sector_str = ", ".join([f"{s}={d:.1f}м" for s, d in self.sectors.items()])
        lines.append(f"сектора: {sector_str}")
        
        # Кластеры ближе 80см
        close_clusters = [c for c in self.clusters if c.min_distance < CLUSTER_LLM_THRESHOLD]
        if close_clusters:
            lines.append("ближе 80см:")
            for c in close_clusters:
                lines.append(f"  {c.to_llm_string()}")
        
        return "\n    ".join(lines)


# ==================== LIDAR ПРОЦЕССОР ====================

class LidarProcessor:
    def __init__(self, 
                 source_id: str = "lidar_front",
                 cluster_threshold: float = 0.3,
                 min_cluster_points: int = 5,
                 max_tracking_age: float = 2.0):
        
        self.source_id = source_id
        self.cluster_threshold = cluster_threshold
        self.min_cluster_points = min_cluster_points
        self.max_tracking_age = max_tracking_age
        
        self.tracked_objects: Dict[int, ObjectCluster] = {}
        self.next_object_id = 1
        self.scan_history = deque(maxlen=10)
        self.last_scan_time = 0
        
        logger.info(f"✅ LidarProcessor инициализирован (source: {source_id})")
    
    def process_raw_points(self, 
                          points: np.ndarray,
                          intensities: Optional[np.ndarray] = None,
                          timestamp: Optional[float] = None) -> Optional[ProcessedLidarData]:
        
        if timestamp is None:
            timestamp = time.time()
        
        if points is None or len(points) == 0:
            logger.warning("⚠️ Получен пустой скан лидара")
            return None
        
        timestamp_us = time.time_ns() // 1000
        scan_id = f"lidar_scan_{timestamp_us}"
        
        clusters = self._cluster_points(points, intensities, timestamp)
        self._update_tracking(clusters, timestamp)
        
        distances = self._compute_distances(points)
        sectors = self._compute_sectors(points)
        obstacles = self._detect_obstacles(clusters, distances)
        moving = self._find_moving_objects(timestamp)
        quality = self._assess_quality(points, timestamp)
        
        self.scan_history.append({
            "timestamp": timestamp,
            "points_count": len(points),
            "quality": quality
        })
        self.last_scan_time = timestamp
        
        return ProcessedLidarData(
            timestamp=timestamp,
            timestamp_us=timestamp_us,
            source_id=self.source_id,
            scan_id=scan_id,
            front_distance=distances["front"],
            left_distance=distances["left"],
            right_distance=distances["right"],
            back_distance=distances["back"],
            sectors=sectors,
            clusters=list(clusters.values()),
            path_clear=not any([obstacles["front"], obstacles["left"], obstacles["right"]]),
            obstacle_front=obstacles["front"],
            obstacle_left=obstacles["left"],
            obstacle_right=obstacles["right"],
            obstacle_back=obstacles["back"],
            moving_objects=moving,
            points_count=len(points),
            data_quality=quality,
            weight=0.95
        )
    
    def _cluster_points(self, 
                       points: np.ndarray,
                       intensities: Optional[np.ndarray],
                       timestamp: float) -> Dict[int, ObjectCluster]:
        
        if not SKLEARN_AVAILABLE:
            return {}
        
        if len(points) < self.min_cluster_points:
            return {}
        
        clustering = DBSCAN(eps=self.cluster_threshold, min_samples=3)
        labels = clustering.fit_predict(points[:, :2])
        
        clusters = {}
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            cluster_points = points[mask]
            
            if len(cluster_points) < self.min_cluster_points:
                continue
            
            center = cluster_points.mean(axis=0)
            min_pt = cluster_points.min(axis=0)
            max_pt = cluster_points.max(axis=0)
            
            obj_id = self._get_or_create_object_id(center, cluster_points)
            existing = self.tracked_objects.get(obj_id)
            
            clusters[obj_id] = ObjectCluster(
                id=obj_id,
                points=[Point3D(x=p[0], y=p[1], z=p[2]) for p in cluster_points],
                center_x=float(center[0]),
                center_y=float(center[1]),
                center_z=float(center[2]),
                width=float(max_pt[0] - min_pt[0]),
                depth=float(max_pt[1] - min_pt[1]),
                height=float(max_pt[2] - min_pt[2]),
                confidence=self._estimate_confidence(cluster_points, intensities, mask),
                object_type=self._classify_object(cluster_points, center),
                first_seen=existing.first_seen if existing else timestamp,
                last_seen=timestamp
            )
        
        return clusters
    
    def _compute_sectors(self, points: np.ndarray) -> Dict[str, float]:
        sectors_dist = {name: 10.0 for name in LIDAR_SECTORS.keys()}
        
        for point in points:
            x, y = point[0], point[1]
            distance = np.sqrt(x**2 + y**2)
            angle_deg = np.degrees(np.arctan2(y, x))
            
            if angle_deg < 0:
                angle_deg += 360
            
            for sector_name, (start, end) in LIDAR_SECTORS.items():
                if start < 0:
                    if angle_deg >= (360 + start) or angle_deg < end:
                        if distance < sectors_dist[sector_name]:
                            sectors_dist[sector_name] = distance
                        break
                else:
                    if start <= angle_deg < end:
                        if distance < sectors_dist[sector_name]:
                            sectors_dist[sector_name] = distance
                        break
        
        return sectors_dist
    
    def _compute_distances(self, points: np.ndarray) -> Dict[str, float]:
        distances = {"front": 10.0, "left": 10.0, "right": 10.0, "back": 10.0}
        
        for point in points:
            x, y = point[0], point[1]
            distance = np.sqrt(x**2 + y**2)
            
            if abs(y) < 0.3:
                if x > 0:
                    distances["front"] = min(distances["front"], distance)
                else:
                    distances["back"] = min(distances["back"], distance)
            elif x > 0:
                if y > 0:
                    distances["left"] = min(distances["left"], distance)
                else:
                    distances["right"] = min(distances["right"], distance)
        
        # Заменяем бесконечность на 10.0
        for k in distances:
            if distances[k] == float('inf'):
                distances[k] = 10.0
        
        return distances
    
    def _detect_obstacles(self, clusters: Dict[int, ObjectCluster],
                          distances: Dict[str, float]) -> Dict[str, bool]:
        DANGER_FRONT, DANGER_SIDE, DANGER_BACK = 0.5, 0.3, 0.3
        
        obstacles = {
            "front": distances["front"] < DANGER_FRONT,
            "left": distances["left"] < DANGER_SIDE,
            "right": distances["right"] < DANGER_SIDE,
            "back": distances["back"] < DANGER_BACK
        }
        
        for cluster in clusters.values():
            dist = np.sqrt(cluster.center_x**2 + cluster.center_y**2)
            
            if cluster.center_x > 0:
                if abs(cluster.center_y) < 0.5 and dist < DANGER_FRONT:
                    obstacles["front"] = True
                elif cluster.center_y > 0 and dist < DANGER_SIDE:
                    obstacles["left"] = True
                elif cluster.center_y < 0 and dist < DANGER_SIDE:
                    obstacles["right"] = True
            else:
                if dist < DANGER_BACK:
                    obstacles["back"] = True
        
        return obstacles
    
    def _find_moving_objects(self, current_time: float) -> List[Dict[str, Any]]:
        moving = []
        
        for obj_id, obj in self.tracked_objects.items():
            if obj.first_seen and obj.last_seen > obj.first_seen:
                age = obj.last_seen - obj.first_seen
                if age > 0.1 and obj.speed > 0.1:
                    moving.append({
                        "id": obj_id,
                        "position": {"x": obj.center_x, "y": obj.center_y},
                        "velocity": {"x": obj.velocity_x, "y": obj.velocity_y},
                        "speed": obj.speed,
                        "type": obj.object_type,
                        "confidence": obj.confidence,
                        "distance": np.sqrt(obj.center_x**2 + obj.center_y**2)
                    })
        
        return moving
    
    def _update_tracking(self, new_clusters: Dict[int, ObjectCluster], timestamp: float):
        expired = []
        for obj_id, obj in self.tracked_objects.items():
            if timestamp - obj.last_seen > self.max_tracking_age:
                expired.append(obj_id)
        
        for obj_id in expired:
            del self.tracked_objects[obj_id]
        
        for obj_id, cluster in new_clusters.items():
            if obj_id in self.tracked_objects:
                old = self.tracked_objects[obj_id]
                dt = timestamp - old.last_seen
                if dt > 0:
                    old.velocity_x = (cluster.center_x - old.center_x) / dt
                    old.velocity_y = (cluster.center_y - old.center_y) / dt
                    old.speed = np.sqrt(old.velocity_x**2 + old.velocity_y**2)
                
                old.center_x = cluster.center_x
                old.center_y = cluster.center_y
                old.center_z = cluster.center_z
                old.last_seen = timestamp
                old.points = cluster.points
                old.min_distance = cluster.min_distance
                old.angle_start = cluster.angle_start
                old.angle_end = cluster.angle_end
            else:
                cluster.first_seen = timestamp
                cluster.last_seen = timestamp
                self.tracked_objects[obj_id] = cluster
    
    def _get_or_create_object_id(self, center: np.ndarray, points: np.ndarray) -> int:
        for obj_id, obj in self.tracked_objects.items():
            dist = np.sqrt((center[0] - obj.center_x)**2 + (center[1] - obj.center_y)**2)
            if dist < self.cluster_threshold:
                return obj_id
        
        new_id = self.next_object_id
        self.next_object_id += 1
        return new_id
    
    def _estimate_confidence(self, points: np.ndarray, intensities: Optional[np.ndarray],
                             mask: np.ndarray) -> float:
        confidence = 1.0
        n_points = np.sum(mask)
        
        if n_points < self.min_cluster_points:
            confidence *= 0.5
        elif n_points < 10:
            confidence *= 0.8
        
        if intensities is not None:
            mean_intensity = np.mean(intensities[mask])
            if mean_intensity < 0.3:
                confidence *= 0.7
            elif mean_intensity < 0.6:
                confidence *= 0.9
        
        return min(1.0, max(0.3, confidence))
    
    def _classify_object(self, points: np.ndarray, center: np.ndarray) -> str:
        height = points[:, 2].max() - points[:, 2].min()
        width = points[:, 0].max() - points[:, 0].min()
        depth = points[:, 1].max() - points[:, 1].min()
        
        if 1.3 < height < 2.2 and width < 1.0 and depth < 1.0:
            return "human"
        if height < 0.1 and (width > 2.0 or depth > 2.0):
            return "wall"
        if 0.5 < height < 1.5 and 0.5 < width < 2.0:
            return "furniture"
        return "obstacle"
    
    def _assess_quality(self, points: np.ndarray, timestamp: float) -> float:
        quality = 1.0
        
        if len(points) < 100:
            quality *= 0.5
        elif len(points) < 300:
            quality *= 0.8
        
        if len(self.scan_history) > 5:
            prev_points = [s["points_count"] for s in self.scan_history]
            mean_points = np.mean(prev_points)
            std_points = np.std(prev_points)
            if std_points > mean_points * 0.3:
                quality *= 0.7
        
        if self.last_scan_time > 0:
            dt = timestamp - self.last_scan_time
            if dt > 0.2:
                quality *= 0.9
            if dt > 0.5:
                quality *= 0.7
        
        return min(1.0, max(0.3, quality))
