#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/feedback_learner.py - ОБУЧЕНИЕ НА ОСНОВЕ ОБРАТНОЙ СВЯЗИ                  ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «дневник успехов и неудач» агента. Собирает статистику выполнения      ║
║   команд: что получилось, что нет, сколько времени заняло, в каком контексте.║
║                                                                              ║
║ ЗАЧЕМ ЭТО НУЖНО:                                                             ║
║   В отличие от StrategyLearner (который эволюционирует стратегии),           ║
║   FeedbackLearner собирает СЫРЫЕ ДАННЫЕ о каждом действии.                   ║
║                                                                              ║
║   На основе этих данных в будущем можно:                                     ║
║     • Предсказывать время выполнения команды                                 ║
║     • Выбирать оптимальные параметры (скорость, длительность)                ║
║     • Обнаруживать «проблемные» команды, которые часто проваливаются         ║
║     • Строить профиль надёжности разных инструментов                         ║
║                                                                              ║
║ КАК РАБОТАЕТ:                                                                ║
║                                                                              ║
║   Каждый раз, когда агент выполняет команду (через Tool), он вызывает:       ║
║   feedback_learner.add_feedback({                                            ║
║       "task_type": "command",                                                ║
║       "success": True/False,                                                 ║
║       "duration": 0.5,                                                       ║
║       "components_used": ["move_forward"],                                   ║
║       "context": {"intent": "explore", "sensors": {...}}                      ║
║   })                                                                         ║
║                                                                              ║
║   Данные сохраняются в JSON-файл и накапливаются между сессиями.             ║
║                                                                              ║
║ ФОРМАТ ДАННЫХ:                                                               ║
║                                                                              ║
║   {                                                                          ║
║     "feedback": [                                                            ║
║       {                                                                      ║
║         "timestamp": 1712345678.123,                                         ║
║         "task_type": "command",                                              ║
║         "success": true,                                                     ║
║         "duration": 0.52,                                                    ║
║         "components_used": [{"id": "move_forward", "type": "tool"}],         ║
║         "context": {"command": "move_forward", "cached": false}              ║
║       }                                                                      ║
║     ],                                                                       ║
║     "stats": {                                                               ║
║       "total": 42,                                                           ║
║       "success": 38,                                                         ║
║       "success_rate": 0.904                                                  ║
║     }                                                                        ║
║   }                                                                          ║
║                                                                              ║
║ ИСПОЛЬЗОВАНИЕ В АГЕНТЕ:                                                      ║
║                                                                              ║
║   Агент вызывает add_feedback() при:                                         ║
║     • Выполнении команды через Tool                                          ║
║     • Получении результата от TinyML                                         ║
║     • Завершении задачи (успех/провал)                                       ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Добавить новые поля в feedback — расширить словарь                      ║
║   2. Анализировать данные — написать метод get_insights()                    ║
║   3. Интегрировать с WeightCalculator — обновлять веса на основе статистики  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import os
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeedbackLearner:
    """
    ОБУЧЕНИЕ НА ОСНОВЕ ОБРАТНОЙ СВЯЗИ
    
    Собирает статистику выполнения команд и сохраняет её в JSON.
    В будущем эти данные можно использовать для оптимизации поведения.
    """
    
    def __init__(self, storage_path: str, max_history: int = 1000):
        """
        Args:
            storage_path: путь к JSON-файлу для сохранения истории
            max_history: максимальное количество записей (старые удаляются)
        """
        self.storage_path = storage_path
        self.max_history = max_history
        
        # История всех feedback'ов
        self.history: List[Dict[str, Any]] = []
        
        # Агрегированная статистика
        self.stats: Dict[str, Any] = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "by_task_type": defaultdict(lambda: {"total": 0, "success": 0}),
            "by_component": defaultdict(lambda: {"total": 0, "success": 0, "total_duration": 0.0})
        }
        
        # Загружаем сохранённые данные
        self.load()
        
        logger.info(f"✅ FeedbackLearner инициализирован (файл: {storage_path})")
    
    # ================================================================
    # ДОБАВЛЕНИЕ FEEDBACK
    # ================================================================
    
    def add_feedback(self, feedback: Dict[str, Any]):
        """
        ДОБАВЛЯЕТ ЗАПИСЬ О ВЫПОЛНЕНИИ ДЕЙСТВИЯ
        
        Вызывается агентом после каждой команды или задачи.
        
        Args:
            feedback: словарь с полями:
                - task_type: str (command, execution, navigation, ...)
                - success: bool
                - duration: float (опционально)
                - components_used: List[Dict] (опционально)
                - context: Dict (опционально)
        """
        # Добавляем временную метку
        feedback["timestamp"] = time.time()
        
        self.history.append(feedback)
        
        # Ограничиваем историю
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Обновляем статистику
        self._update_stats(feedback)
        
        # Периодическое сохранение (каждые 10 записей)
        if len(self.history) % 10 == 0:
            self.save()
        
        logger.debug(f"📊 Feedback добавлен: {feedback.get('task_type')} - {'✅' if feedback.get('success') else '❌'}")
    
    def _update_stats(self, feedback: Dict[str, Any]):
        """Внутренний метод: обновляет агрегированную статистику"""
        task_type = feedback.get("task_type", "unknown")
        success = feedback.get("success", False)
        duration = feedback.get("duration", 0.0)
        components = feedback.get("components_used", [])
        
        # Общая статистика
        self.stats["total"] += 1
        if success:
            self.stats["success"] += 1
        else:
            self.stats["failed"] += 1
        
        # По типу задачи
        self.stats["by_task_type"][task_type]["total"] += 1
        if success:
            self.stats["by_task_type"][task_type]["success"] += 1
        
        # По компонентам
        for comp in components:
            comp_id = comp.get("id", "unknown")
            comp_type = comp.get("type", "unknown")
            key = f"{comp_type}:{comp_id}"
            
            self.stats["by_component"][key]["total"] += 1
            if success:
                self.stats["by_component"][key]["success"] += 1
            self.stats["by_component"][key]["total_duration"] += duration
    
    # ================================================================
    # ПОЛУЧЕНИЕ СТАТИСТИКИ
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ВОЗВРАЩАЕТ ТЕКУЩУЮ СТАТИСТИКУ
        
        Используется агентом для отладки и может быть вызвано через инструмент.
        """
        total = self.stats["total"]
        success = self.stats["success"]
        
        stats_copy = {
            "total": total,
            "success": success,
            "failed": self.stats["failed"],
            "success_rate": success / total if total > 0 else 0.0,
            "by_task_type": {},
            "by_component": {}
        }
        
        # Копируем с вычислением success_rate
        for task_type, data in self.stats["by_task_type"].items():
            stats_copy["by_task_type"][task_type] = {
                "total": data["total"],
                "success": data["success"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0.0
            }
        
        for comp_key, data in self.stats["by_component"].items():
            stats_copy["by_component"][comp_key] = {
                "total": data["total"],
                "success": data["success"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0.0,
                "avg_duration": data["total_duration"] / data["total"] if data["total"] > 0 else 0.0
            }
        
        return stats_copy
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        ВОЗВРАЩАЕТ ПОСЛЕДНИЕ N ЗАПИСЕЙ
        
        Используется для отладки.
        """
        return self.history[-limit:]
    
    def get_success_rate(self, task_type: Optional[str] = None) -> float:
        """
        ВОЗВРАЩАЕТ ДОЛЮ УСПЕШНЫХ ВЫПОЛНЕНИЙ
        
        Args:
            task_type: если указан — только для этого типа задач
        """
        if task_type:
            data = self.stats["by_task_type"].get(task_type, {})
            total = data.get("total", 0)
            success = data.get("success", 0)
        else:
            total = self.stats["total"]
            success = self.stats["success"]
        
        return success / total if total > 0 else 0.0
    
    # ================================================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ================================================================
    
    def save(self):
        """Сохраняет историю и статистику в JSON-файл"""
        try:
            # Преобразуем defaultdict в обычные dict для сериализации
            stats_serializable = {
                "total": self.stats["total"],
                "success": self.stats["success"],
                "failed": self.stats["failed"],
                "by_task_type": dict(self.stats["by_task_type"]),
                "by_component": dict(self.stats["by_component"])
            }
            
            data = {
                "history": self.history[-self.max_history:],  # сохраняем только последние
                "stats": stats_serializable,
                "version": "1.0",
                "saved_at": time.time()
            }
            
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 FeedbackLearner сохранён ({len(self.history)} записей)")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения FeedbackLearner: {e}")
    
    def load(self):
        """Загружает историю и статистику из JSON-файла"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.history = data.get("history", [])
            
            # Восстанавливаем статистику
            saved_stats = data.get("stats", {})
            self.stats["total"] = saved_stats.get("total", 0)
            self.stats["success"] = saved_stats.get("success", 0)
            self.stats["failed"] = saved_stats.get("failed", 0)
            
            # Восстанавливаем defaultdict'ы
            for task_type, task_data in saved_stats.get("by_task_type", {}).items():
                self.stats["by_task_type"][task_type] = task_data
            
            for comp_key, comp_data in saved_stats.get("by_component", {}).items():
                self.stats["by_component"][comp_key] = comp_data
            
            logger.info(f"📂 FeedbackLearner загружен ({len(self.history)} записей)")
        except FileNotFoundError:
            logger.info("🆕 Создан новый FeedbackLearner")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки FeedbackLearner: {e}")
    
    def clear(self):
        """Очищает всю историю и статистику"""
        self.history.clear()
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "by_task_type": defaultdict(lambda: {"total": 0, "success": 0}),
            "by_component": defaultdict(lambda: {"total": 0, "success": 0, "total_duration": 0.0})
        }
        self.save()
        logger.info("🧹 FeedbackLearner очищен")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ FEEDBACK LEARNER")
    print("="*60 + "\n")
    
    # Создаём learner во временном файле
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        learner = FeedbackLearner(tmp.name)
    
    # Добавляем несколько feedback'ов
    print("📊 Добавляем feedback'ы...")
    
    learner.add_feedback({
        "task_type": "command",
        "success": True,
        "duration": 0.52,
        "components_used": [{"id": "move_forward", "type": "tool"}],
        "context": {"command": "move_forward", "cached": False}
    })
    
    learner.add_feedback({
        "task_type": "command",
        "success": True,
        "duration": 0.48,
        "components_used": [{"id": "turn_left", "type": "tool"}],
        "context": {"command": "turn_left", "cached": False}
    })
    
    learner.add_feedback({
        "task_type": "command",
        "success": False,
        "duration": 1.23,
        "components_used": [{"id": "move_forward", "type": "tool"}],
        "context": {"command": "move_forward", "obstacle": True}
    })
    
    learner.add_feedback({
        "task_type": "execution",
        "success": True,
        "duration": 0.15,
        "components_used": [{"id": "tinyml", "type": "executor"}],
        "context": {"reflex": "obstacle_front"}
    })
    
    learner.add_feedback({
        "task_type": "command",
        "success": True,
        "duration": 0.51,
        "components_used": [{"id": "move_forward", "type": "tool"}],
        "context": {"command": "move_forward", "cached": True}
    })
    
    # Сохраняем
    learner.save()
    
    # Показываем статистику
    print("\n📈 СТАТИСТИКА:")
    stats = learner.get_stats()
    print(f"   Всего действий: {stats['total']}")
    print(f"   Успешно: {stats['success']}")
    print(f"   Провалено: {stats['failed']}")
    print(f"   Общий success rate: {stats['success_rate']:.1%}")
    
    print("\n📊 ПО ТИПАМ ЗАДАЧ:")
    for task_type, data in stats['by_task_type'].items():
        print(f"   {task_type}: {data['success']}/{data['total']} ({data['success_rate']:.1%})")
    
    print("\n🔧 ПО КОМПОНЕНТАМ:")
    for comp, data in stats['by_component'].items():
        print(f"   {comp}: {data['success']}/{data['total']} ({data['success_rate']:.1%}), среднее время: {data['avg_duration']:.2f}с")
    
    # Последние записи
    print("\n📋 ПОСЛЕДНИЕ 3 ЗАПИСИ:")
    for fb in learner.get_recent_feedback(3):
        status = "✅" if fb['success'] else "❌"
        print(f"   {status} {fb['task_type']} ({fb.get('duration', 0):.2f}с)")
    
    print("\n" + "="*60)
    print("✅ Демонстрация завершена.")
    print("="*60 + "\n")
