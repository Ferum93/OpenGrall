#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/vla_dataset_collector.py — СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ VLA                 ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Опциональный модуль для автоматического сбора датасета, пригодного для     ║
║   обучения VLA-моделей (Vision-Language-Action).                             ║
║                                                                              ║
║   ВКЛЮЧЕНИЕ:                                                                 ║
║      self.vla_collector = VLADatasetCollector(enabled=True)                  ║
║      self.tools.append(VLACollectorTool(self.vla_collector))                 ║
║                                                                              ║
║   ФОРМАТ ВЫХОДА:                                                             ║
║      data/vla_dataset/                                                        ║
║      ├── episodes.jsonl     # метаданные всех эпизодов                       ║
║      ├── frames/            # кадры с камер                                  ║
║      └── metadata.json      # информация о роботе и конфигурации             ║
║                                                                              ║
║   СОВМЕСТИМОСТЬ:                                                             ║
║      Формат совместим с LeRobot, Open X-Embodiment, RT-2, Octo, Green-VLA.   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import base64
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VLAFrame:
    """Один кадр эпизода"""
    step: int
    timestamp: float
    image_base64: str
    image_path: str
    camera_id: str
    context: Dict[str, Any]


@dataclass
class VLAAction:
    """Одно действие, выполненное роботом"""
    step: int
    action: str
    params: Dict[str, Any]
    reasoning: str
    timestamp: float


@dataclass
class VLAEpisode:
    """Один эпизод (полный цикл выполнения задачи)"""
    episode_id: str
    timestamp: float
    task_description: str
    frames: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "task_description": self.task_description,
            "frames": self.frames,
            "actions": self.actions,
            "success": self.success,
            "metadata": self.metadata
        }


class VLADatasetCollector:
    """
    КОЛЛЕКТОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ VLA
    
    Собирает кадры, контекст, действия и reasoning в формате,
    совместимом с LeRobot, Open X-Embodiment и другими VLA-фреймворками.
    """
    
    def __init__(self, 
                 enabled: bool = False,
                 base_path: str = None,
                 save_frames: bool = True,
                 compress_images: bool = True,
                 max_image_size: int = 640):
        
        self.enabled = enabled
        if not self.enabled:
            logger.info("⏸️ VLADatasetCollector отключён (enabled=False)")
            return
        
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "..", "data", "vla_dataset")
        
        self.base_path = os.path.abspath(base_path)
        self.frames_path = os.path.join(self.base_path, "frames")
        self.save_frames = save_frames
        self.compress_images = compress_images
        self.max_image_size = max_image_size
        
        # Текущий эпизод
        self.current_episode: Optional[VLAEpisode] = None
        self.step_counter = 0
        self.is_recording = False
        
        # Создаём директории
        os.makedirs(self.base_path, exist_ok=True)
        if self.save_frames:
            os.makedirs(self.frames_path, exist_ok=True)
        
        # Файл для метаданных всех эпизодов
        self.episodes_file = os.path.join(self.base_path, "episodes.jsonl")
        
        # Метаданные робота
        self.robot_metadata = {
            "collected_at": datetime.now().isoformat(),
            "collector_version": "1.0",
            "format": "VLA/OpenGrall-1.0"
        }
        
        self._save_metadata()
        
        logger.info(f"✅ VLADatasetCollector активирован (base_path={self.base_path})")
    
    def set_robot_info(self, **kwargs):
        """Сохраняет информацию о роботе (габариты, камеры, конфигурация)"""
        self.robot_metadata.update(kwargs)
        self._save_metadata()
    
    def start_episode(self, task_description: str) -> str:
        """Начинает новый эпизод"""
        if not self.enabled:
            return ""
        
        episode_id = f"ep_{int(time.time())}_{task_description[:20].replace(' ', '_')}"
        self.current_episode = VLAEpisode(
            episode_id=episode_id,
            timestamp=time.time(),
            task_description=task_description,
            metadata={"status": "recording"}
        )
        self.step_counter = 0
        self.is_recording = True
        
        logger.info(f"🎬 Начат эпизод: {episode_id} ({task_description[:50]}...)")
        return episode_id
    
    def record_frame(self, 
                     image_base64: str,
                     camera_id: str = "front",
                     context: Dict[str, Any] = None) -> Optional[str]:
        """
        Записывает один кадр с камеры.
        Вызывается перед выполнением действия или во время сканирования.
        """
        if not self.enabled or not self.is_recording:
            return None
        
        # Сохраняем кадр в файл
        image_path = ""
        if self.save_frames:
            episode_dir = os.path.join(self.frames_path, self.current_episode.episode_id)
            os.makedirs(episode_dir, exist_ok=True)
            image_path = os.path.join(episode_dir, f"step_{self.step_counter:04d}.jpg")
            
            try:
                image_bytes = base64.b64decode(image_base64)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                logger.error(f"Ошибка сохранения кадра: {e}")
                image_path = ""
        
        # Создаём запись кадра
        frame = VLAFrame(
            step=self.step_counter,
            timestamp=time.time(),
            image_base64=image_base64 if not self.save_frames else "",
            image_path=image_path,
            camera_id=camera_id,
            context=context or {}
        )
        
        self.current_episode.frames.append(asdict(frame))
        logger.debug(f"📸 Кадр {self.step_counter} записан ({camera_id})")
        
        return image_path
    
    def record_action(self, 
                      action: str,
                      params: Dict[str, Any],
                      reasoning: str = "") -> bool:
        """
        Записывает действие, выполненное роботом.
        Вызывается ПОСЛЕ того, как LLM приняла решение.
        """
        if not self.enabled or not self.is_recording:
            return False
        
        vla_action = VLAAction(
            step=self.step_counter,
            action=action,
            params=params,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        self.current_episode.actions.append(asdict(vla_action))
        self.step_counter += 1
        
        logger.debug(f"⚡ Действие {action} записано (шаг {self.step_counter-1})")
        return True
    
    def end_episode(self, success: bool = True) -> bool:
        """Завершает эпизод и сохраняет его в файл"""
        if not self.enabled or not self.is_recording:
            return False
        
        self.current_episode.success = success
        self.current_episode.metadata["status"] = "completed" if success else "failed"
        self.current_episode.metadata["total_steps"] = self.step_counter
        self.current_episode.metadata["duration"] = time.time() - self.current_episode.timestamp
        
        # Сохраняем в JSONL
        try:
            with open(self.episodes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.current_episode.to_dict(), ensure_ascii=False) + "\n")
            
            logger.info(f"✅ Эпизод {self.current_episode.episode_id} сохранён "
                       f"({'успех' if success else 'провал'}, {self.step_counter} шагов)")
        except Exception as e:
            logger.error(f"Ошибка сохранения эпизода: {e}")
            return False
        
        self.is_recording = False
        self.current_episode = None
        self.step_counter = 0
        
        return True
    
    def cancel_episode(self):
        """Отменяет текущий эпизод без сохранения"""
        if not self.enabled:
            return
        
        logger.warning(f"🛑 Эпизод {self.current_episode.episode_id if self.current_episode else '?'} отменён")
        self.is_recording = False
        self.current_episode = None
        self.step_counter = 0
    
    def _save_metadata(self):
        """Сохраняет метаданные робота"""
        metadata_file = os.path.join(self.base_path, "metadata.json")
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.robot_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения metadata.json: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сборщика"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            with open(self.episodes_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            total_episodes = len(lines)
            
            success_count = 0
            total_steps = 0
            for line in lines:
                ep = json.loads(line)
                if ep.get("success"):
                    success_count += 1
                total_steps += ep.get("metadata", {}).get("total_steps", 0)
            
            return {
                "enabled": True,
                "total_episodes": total_episodes,
                "success_rate": success_count / total_episodes if total_episodes > 0 else 0,
                "total_steps": total_steps,
                "is_recording": self.is_recording,
                "current_episode": self.current_episode.episode_id if self.current_episode else None,
                "base_path": self.base_path
            }
        except FileNotFoundError:
            return {
                "enabled": True,
                "total_episodes": 0,
                "success_rate": 0,
                "total_steps": 0,
                "is_recording": self.is_recording,
                "current_episode": self.current_episode.episode_id if self.current_episode else None,
                "base_path": self.base_path
            }


# ================================================================
# ИНСТРУМЕНТ ДЛЯ УПРАВЛЕНИЯ СБОРОМ ДАННЫХ
# ================================================================

class VLACollectorTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ УПРАВЛЕНИЯ СБОРОМ ДАННЫХ VLA
    
    Позволяет LLM включать/выключать запись эпизодов для обучения VLA.
    """
    name = "vla_record"
    description = "Управлять записью эпизодов для обучения VLA: start, stop, cancel, status"
    
    def __init__(self, collector: VLADatasetCollector):
        self.collector = collector
        self.latency = 0.05
    
    async def forward(self, command: str, task_description: str = "", success: bool = True) -> str:
        if not self.collector.enabled:
            return "Сбор данных VLA отключён"
        
        if command == "start":
            if not task_description:
                return "Укажите task_description (что делает робот)"
            episode_id = self.collector.start_episode(task_description)
            return f"Запись эпизода начата: {episode_id}"
        
        elif command == "stop":
            if not self.collector.is_recording:
                return "Нет активной записи"
            self.collector.end_episode(success)
            return f"Запись эпизода завершена (success={success})"
        
        elif command == "cancel":
            self.collector.cancel_episode()
            return "Запись эпизода отменена"
        
        elif command == "status":
            stats = self.collector.get_stats()
            if stats["is_recording"]:
                return f"Идёт запись эпизода {stats['current_episode']}. Всего эпизодов: {stats['total_episodes']}"
            else:
                return f"Запись не активна. Всего эпизодов: {stats['total_episodes']}, success rate: {stats['success_rate']:.1%}"
        
        else:
            return f"Неизвестная команда: {command}. Используйте start, stop, cancel, status"