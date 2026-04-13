#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/llm_decision_memory.py - КЭШ РЕШЕНИЙ LLM + УПРАВЛЕНИЕ ЗАДАЧАМИ          ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это "оперативная память" для LLM. Она решает две главные задачи:           ║
║                                                                              ║
║   1. КЭШИРОВАНИЕ РЕШЕНИЙ — чтобы не вызывать LLM для типовых ситуаций        ║
║   2. УПРАВЛЕНИЕ ЗАДАЧАМИ — хранение и пошаговое выполнение сложных планов    ║
║                                                                              ║
║ ЗАЧЕМ НУЖЕН КЭШ:                                                             ║
║                                                                              ║
║   LLM медленная и дорогая. Если робот едет по длинному коридору,             ║
║   ситуация повторяется десятки раз: "путь свободен, еду вперёд".             ║
║   Вместо 30 вызовов LLM мы делаем 1 вызов, а остальные берём из кэша.        ║
║                                                                              ║
║   КАК РАБОТАЕТ КЭШ:                                                          ║
║     1. Контекст (сенсоры + намерение) хешируется                             ║
║     2. Если такой хеш уже есть — берём решение из кэша                        ║
║     3. Решение имеет ВЕС, который ЗАТУХАЕТ со временем                        ║
║     4. Если вес упал ниже 0.3 — решение считается устаревшим                 ║
║                                                                              ║
║ ЗАЧЕМ НУЖНО УПРАВЛЕНИЕ ЗАДАЧАМИ:                                             ║
║                                                                              ║
║   Когда робот в интерактивном режиме сам придумывает задачу (см. пример      ║
║   выше про "поприветствовать человека"), эта задача состоит из НЕСКОЛЬКИХ    ║
║   шагов. LLMDecisionMemory хранит эту задачу, отслеживает текущий шаг,       ║
║   и выполняет шаги последовательно.                                          ║
║                                                                              ║
║   ПРИМЕР ЗАДАЧИ (которую LLM сгенерировала сама):                            ║
║   {                                                                          ║
║     "task_name": "поприветствовать человека",                                ║
║     "reasoning": "...",                                                      ║
║     "steps": [                                                               ║
║       "повернуться направо",                                                 ║
║       "сказать: 'Здравствуйте! Вам нужна помощь?'",                          ║
║       "выслушать ответ",                                                     ║
║       "если нужно — помочь, иначе — извиниться и уйти"                       ║
║     ],                                                                       ║
║     "expected_outcome": "..."                                                ║
║   }                                                                          ║
║                                                                              ║
║   LLMDecisionMemory сохраняет это как current_task и выполняет шаг за шагом.  ║
║   Каждый шаг отправляется в LLM (или берётся из кэша), и робот действует.    ║
║                                                                              ║
║ ИНТЕРАКТИВНЫЙ РЕЖИМ И АВТОНОМНЫЕ ЗАДАЧИ:                                     ║
║                                                                              ║
║   Когда робот долго бездействует (сейчас хардкод: 120 секунд), он            ║
║   входит в интерактивный режим и генерирует себе задачу через                ║
║   StrategyLearner.generate_self_task().                                      ║
║                                                                              ║
║   В БУДУЩЕМ (в дорожной карте):                                              ║
║   - Вместо хардкода 120 секунд — рандомайзер (робот "оживает" непредсказуемо)║
║   - Успешные задачи сохраняются в EpisodicMemory для повторения              ║
║   - Робот учится на своих планах и становится всё более "живым"              ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from core.llm_decision_memory import LLMDecisionMemory                      ║
║                                                                              ║
║   memory = LLMDecisionMemory()                                               ║
║                                                                              ║
║   # Установить задачу                                                        ║
║   memory.set_task(                                                           ║
║       steps=["повернуться", "поздороваться", "спросить про помощь"],          ║
║       reasoning="Вижу человека, хочу помочь",                                ║
║       task_name="поприветствовать человека"                                  ║
║   )                                                                          ║
║                                                                              ║
║   # Выполнять шаги                                                           ║
║   while memory.has_active_task():                                            ║
║       step = memory.get_current_step()                                       ║
║       result = await agent.process_with_llm(step, is_task_step=True)         ║
║       memory.advance_step(result)                                            ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Изменить время жизни кэша:                                              ║
║      memory = LLMDecisionMemory(default_expiry=20.0)  # 20 сек вместо 10     ║
║                                                                              ║
║   2. Изменить порог активации кэша:                                          ║
║      В agent_v5.py: if cached.get_current_weight() > 0.5  # вместо 0.3       ║
║                                                                              ║
║   3. Сохранять историю задач:                                                ║
║      memory.task_history — список выполненных задач, можно анализировать     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging 
import time
import uuid
import hashlib
import json
import os
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
from copy import deepcopy

logger = logging.getLogger(__name__)

# ================================================================
# КОНСТАНТЫ
# ================================================================

DECAY_FACTOR = 0.9          # затухание веса со временем (каждую секунду ×0.9)
INITIAL_WEIGHT = 0.5        # начальный вес нового решения
MIN_WEIGHT = 0.1            # минимальный вес (ниже — решение забыто)
MAX_ACTIVE_DECISIONS = 100  # максимум решений в кэше


@dataclass
class LLMDecision:
    """
    ОДНО РЕШЕНИЕ LLM, СОХРАНЁННОЕ В КЭШЕ
    
    Хранит не только команду, но и КОНТЕКСТ, в котором она была принята.
    Это позволяет найти решение, когда ситуация повторяется.
    """
    decision_id: str
    timestamp: float
    command: Dict[str, Any]          # {"action": "move_forward", "params": {...}}
    context_hash: str                # хеш контекста (для быстрого поиска)
    context_summary: str             # краткое описание ситуации
    reasoning: str                   # почему LLM приняла такое решение
    initial_weight: float = INITIAL_WEIGHT
    execution_result: Optional[Dict[str, Any]] = None
    expiry_time: float = 10.0        # время жизни решения в секундах
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_weight(self, current_time: float = None) -> float:
        """
        ТЕКУЩИЙ ВЕС РЕШЕНИЯ С УЧЁТОМ ЗАТУХАНИЯ
        
        Чем старше решение, тем меньше его вес.
        Формула: вес = начальный_вес × (DECAY_FACTOR ^ возраст)
        """
        if current_time is None:
            current_time = time.time()
        
        age = current_time - self.timestamp
        decay = DECAY_FACTOR ** age
        current_weight = self.initial_weight * decay
        return max(MIN_WEIGHT, current_weight)
    
    def is_active(self, current_time: float = None) -> bool:
        """Решение ещё не истекло?"""
        if current_time is None:
            current_time = time.time()
        return (current_time - self.timestamp) < self.expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "command": self.command,
            "context_hash": self.context_hash,
            "context_summary": self.context_summary,
            "reasoning": self.reasoning,
            "initial_weight": self.initial_weight,
            "current_weight": self.get_current_weight(),
            "execution_result": self.execution_result,
            "metadata": self.metadata
        }


class LLMDecisionMemory:
    """
    КЭШ РЕШЕНИЙ LLM + УПРАВЛЕНИЕ ТЕКУЩЕЙ ЗАДАЧЕЙ
    
    Две роли в одном классе:
    1. Кэш — чтобы не дёргать LLM для повторяющихся ситуаций
    2. Задачи — чтобы выполнять сложные планы пошагово
    """
    
    def __init__(self, max_decisions: int = MAX_ACTIVE_DECISIONS, 
                 default_expiry: float = 10.0):
        self.max_decisions = max_decisions
        self.default_expiry = default_expiry
        
        # КЭШ РЕШЕНИЙ
        self.decisions: Dict[str, LLMDecision] = {}
        self.context_index: Dict[str, str] = {}  # хеш -> decision_id
        self.timeline = deque(maxlen=max_decisions)
        
        # УПРАВЛЕНИЕ ЗАДАЧАМИ
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
        self.task_id_counter = 0
        
        self.lock = threading.Lock()
        
        self.stats = {
            "total_decisions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "active_decisions": 0,
            "tasks_completed": 0,
            "tasks_failed": 0
        }
        
        logger.info("✅ LLMDecisionMemory инициализирована (кэш + управление задачами)")
    
    # ================================================================
    # КЭШИРОВАНИЕ РЕШЕНИЙ
    # ================================================================
    
    def add_decision(self, 
                     command: Dict[str, Any],
                     context: Dict[str, Any],
                     reasoning: str = "",
                     metadata: Dict[str, Any] = None) -> str:
        """
        СОХРАНЯЕТ РЕШЕНИЕ LLM В КЭШ
        
        Вызывается каждый раз, когда LLM приняла решение.
        В будущем, если контекст повторится, решение будет взято из кэша.
        """
        with self.lock:
            decision_id = str(uuid.uuid4())
            timestamp = time.time()
            
            context_hash = self._hash_context(context)
            context_summary = self._summarize_context(context)
            
            decision = LLMDecision(
                decision_id=decision_id,
                timestamp=timestamp,
                command=deepcopy(command),
                context_hash=context_hash,
                context_summary=context_summary,
                reasoning=reasoning,
                initial_weight=INITIAL_WEIGHT,
                expiry_time=self.default_expiry,
                metadata=metadata or {}
            )
            
            self.decisions[decision_id] = decision
            self.context_index[context_hash] = decision_id
            self.timeline.append(decision_id)
            
            self._cleanup_old_decisions()
            
            self.stats["total_decisions"] += 1
            self.stats["active_decisions"] = len(self.get_active_decisions())
            
            logger.debug(f"💾 Решение сохранено в кэш: {context_summary}")
            return decision_id
    
    def find_similar(self, context: Dict[str, Any]) -> Optional[LLMDecision]:
        """
        ИЩЕТ ПОХОЖЕЕ РЕШЕНИЕ В КЭШЕ
        
        Если контекст точно совпадает (по хешу) и решение ещё активно —
        возвращает его. Это позволяет НЕ вызывать LLM повторно.
        """
        current_hash = self._hash_context(context)
        
        with self.lock:
            if current_hash in self.context_index:
                decision_id = self.context_index[current_hash]
                decision = self.decisions.get(decision_id)
                
                if decision and decision.is_active():
                    if decision.get_current_weight() > 0.3:
                        self.stats["cache_hits"] += 1
                        logger.debug(f"⚡ Кэш HIT: {decision.context_summary}")
                        return decision
            
            self.stats["cache_misses"] += 1
            return None
    
    def get_active_decisions(self, current_time: float = None) -> List[LLMDecision]:
        """Возвращает все активные (не истекшие) решения"""
        if current_time is None:
            current_time = time.time()
        
        with self.lock:
            active = []
            for decision in self.decisions.values():
                if decision.is_active(current_time):
                    active.append(decision)
            
            active.sort(key=lambda d: d.get_current_weight(current_time), reverse=True)
            return active
    
    # ================================================================
    # УПРАВЛЕНИЕ ЗАДАЧАМИ (ПОШАГОВОЕ ВЫПОЛНЕНИЕ)
    # ================================================================
    
    def set_task(self, steps: List[str], reasoning: str = "", 
                 context: Dict = None, task_name: str = None) -> str:
        """
        УСТАНАВЛИВАЕТ НОВУЮ ЗАДАЧУ ДЛЯ ВЫПОЛНЕНИЯ
        
        Вызывается, когда робот в интерактивном режиме сам придумал задачу
        (через StrategyLearner.generate_self_task()).
        
        Args:
            steps: список шагов (на естественном языке)
            reasoning: почему задача была создана
            context: контекст, в котором задача создана
            task_name: название задачи
        
        Returns:
            task_id: уникальный идентификатор задачи
        """
        with self.lock:
            self.task_id_counter += 1
            task_id = f"task_{self.task_id_counter}_{int(time.time())}"
            
            self.current_task = {
                "id": task_id,
                "name": task_name or f"задача_{self.task_id_counter}",
                "steps": steps,
                "current_step": 0,
                "reasoning": reasoning,
                "context": context or {},
                "created_at": time.time(),
                "status": "active",
                "history": []
            }
            
            logger.info(f"📋 Установлена задача '{self.current_task['name']}' ({len(steps)} шагов)")
            logger.info(f"   Причина: {reasoning[:100]}...")
            return task_id
    
    def get_current_step(self) -> Optional[str]:
        """Возвращает текущий шаг задачи (или None, если задачи нет)"""
        if self.current_task:
            step_idx = self.current_task["current_step"]
            if step_idx < len(self.current_task["steps"]):
                return self.current_task["steps"][step_idx]
        return None
    
    def get_current_step_index(self) -> int:
        """Возвращает индекс текущего шага (0-based)"""
        if self.current_task:
            return self.current_task["current_step"]
        return -1
    
    def get_task_progress(self) -> Dict[str, Any]:
        """Возвращает информацию о прогрессе задачи"""
        if not self.current_task:
            return {"has_task": False}
        
        steps = self.current_task["steps"]
        current = self.current_task["current_step"]
        
        return {
            "has_task": True,
            "task_id": self.current_task["id"],
            "task_name": self.current_task["name"],
            "current_step": current,
            "total_steps": len(steps),
            "progress": current / len(steps) if steps else 0,
            "remaining_steps": steps[current:] if current < len(steps) else [],
            "reasoning": self.current_task.get("reasoning", "")
        }
    
    def advance_step(self, result: Any = None) -> bool:
        """
        ПЕРЕХОДИТ К СЛЕДУЮЩЕМУ ШАГУ ЗАДАЧИ
        
        Вызывается после успешного выполнения текущего шага.
        
        Returns:
            True — если есть следующий шаг
            False — если задача завершена
        """
        with self.lock:
            if not self.current_task:
                return False
            
            step_idx = self.current_task["current_step"]
            steps = self.current_task["steps"]
            
            if step_idx >= len(steps):
                return False
            
            # Сохраняем выполненный шаг в историю
            self.current_task["history"].append({
                "step": step_idx,
                "description": steps[step_idx],
                "result": result,
                "completed_at": time.time()
            })
            
            # Переходим к следующему
            self.current_task["current_step"] += 1
            
            # Проверяем завершение
            if self.current_task["current_step"] >= len(steps):
                self.current_task["status"] = "completed"
                self.current_task["completed_at"] = time.time()
                self.task_history.append(self.current_task)
                self.stats["tasks_completed"] += 1
                
                logger.info(f"✅ Задача '{self.current_task['name']}' выполнена!")
                self.current_task = None
                return False
            
            next_step = self.current_task["steps"][self.current_task["current_step"]]
            logger.debug(f"➡️ Шаг {step_idx + 1}/{len(steps)}: {next_step[:50]}...")
            return True
    
    def fail_task(self, reason: str):
        """
        ПОМЕЧАЕТ ЗАДАЧУ КАК ПРОВАЛЕННУЮ
        
        Вызывается, если что-то пошло не так (рефлекс прервал, ошибка выполнения).
        """
        with self.lock:
            if not self.current_task:
                return
            
            self.current_task["status"] = "failed"
            self.current_task["failure_reason"] = reason
            self.current_task["failed_at"] = time.time()
            self.task_history.append(self.current_task)
            self.stats["tasks_failed"] += 1
            
            logger.warning(f"❌ Задача '{self.current_task['name']}' провалена: {reason}")
            self.current_task = None
    
    def cancel_task(self):
        """Отменяет текущую задачу (без пометки как проваленная)"""
        with self.lock:
            if not self.current_task:
                return
            
            logger.info(f"🛑 Задача '{self.current_task['name']}' отменена")
            self.current_task = None
    
    def has_active_task(self) -> bool:
        """Есть ли активная задача прямо сейчас?"""
        return self.current_task is not None
    
    def get_active_task(self) -> Optional[Dict]:
        """Возвращает текущую активную задачу"""
        return self.current_task
    
    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """Возвращает историю выполненных/проваленных задач"""
        return self.task_history[-limit:]
    
    # ================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Создаёт MD5-хеш контекста для быстрого сравнения"""
        normalized = self._normalize_context(context)
        context_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        НОРМАЛИЗУЕТ КОНТЕКСТ ДЛЯ СРАВНЕНИЯ
        
        Оставляет только то, что действительно влияет на решение:
        - намерение (intent)
        - наличие препятствий
        - наличие движущихся объектов
        """
        normalized = {}
        
        if "current_intent" in context:
            normalized["intent"] = context["current_intent"]
        
        if "sensors" in context:
            sensors = []
            for s in context["sensors"]:
                data = s.get("data", {})
                sensors.append({
                    "type": s.get("source_type"),
                    "has_obstacle": data.get("obstacle_front", False),
                    "has_moving": len(data.get("moving_objects", [])) > 0
                })
            normalized["sensors"] = sensors
        
        if "active_reflexes" in context and context["active_reflexes"]:
            normalized["has_reflexes"] = True
        
        return normalized
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Создаёт краткое описание контекста для логов"""
        parts = []
        
        intent = context.get("current_intent", "unknown")
        parts.append(f"intent:{intent}")
        
        sensors = context.get("sensors", [])
        if sensors:
            main_sensor = sensors[0]
            s_type = main_sensor.get("source_type", "?")
            data = main_sensor.get("data", {})
            
            if "obstacle_front" in data:
                parts.append(f"obs:{data['obstacle_front']}")
            if "moving_objects" in data:
                parts.append(f"moving:{len(data['moving_objects'])}")
        
        return " | ".join(parts)
    
    def _cleanup_old_decisions(self):
        """Удаляет самые старые решения при переполнении кэша"""
        if len(self.decisions) <= self.max_decisions:
            return
        
        sorted_ids = sorted(self.decisions.keys(), 
                           key=lambda did: self.decisions[did].timestamp)
        
        to_remove = len(self.decisions) - self.max_decisions
        for i in range(to_remove):
            old_id = sorted_ids[i]
            old_decision = self.decisions[old_id]
            if old_decision.context_hash in self.context_index:
                del self.context_index[old_decision.context_hash]
            del self.decisions[old_id]
        
        logger.debug(f"🧹 Кэш очищен: удалено {to_remove} старых решений")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша и задач"""
        with self.lock:
            total = self.stats["cache_hits"] + self.stats["cache_misses"]
            hit_rate = self.stats["cache_hits"] / total if total > 0 else 0
            
            return {
                "total_decisions": self.stats["total_decisions"],
                "active_decisions": len(self.get_active_decisions()),
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "hit_rate": hit_rate,
                "memory_usage": len(self.decisions),
                "has_active_task": self.current_task is not None,
                "tasks_completed": self.stats["tasks_completed"],
                "tasks_failed": self.stats["tasks_failed"]
            }
    
    def save_to_file(self, filepath: str):
        """Сохраняет кэш и историю задач в JSON"""
        with self.lock:
            data = {
                "decisions": {did: d.to_dict() for did, d in self.decisions.items()},
                "task_history": self.task_history,
                "stats": self.stats,
                "timestamp": time.time(),
                "version": "2.0"
            }
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Память решений сохранена в {filepath}")
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения: {e}")
    
    def load_from_file(self, filepath: str):
        """Загружает кэш и историю задач из JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                self.decisions.clear()
                self.context_index.clear()
                self.timeline.clear()
                
                for did, ddata in data.get("decisions", {}).items():
                    decision = LLMDecision(
                        decision_id=ddata["decision_id"],
                        timestamp=ddata["timestamp"],
                        command=ddata["command"],
                        context_hash=ddata["context_hash"],
                        context_summary=ddata["context_summary"],
                        reasoning=ddata.get("reasoning", ""),
                        initial_weight=ddata.get("initial_weight", INITIAL_WEIGHT),
                        execution_result=ddata.get("execution_result"),
                        metadata=ddata.get("metadata", {})
                    )
                    self.decisions[did] = decision
                    self.context_index[ddata["context_hash"]] = did
                    self.timeline.append(did)
                
                self.task_history = data.get("task_history", [])
                self.stats = data.get("stats", self.stats)
                self.stats["active_decisions"] = len(self.get_active_decisions())
            
            logger.info(f"📂 Память решений загружена из {filepath}")
        except FileNotFoundError:
            logger.info("🆕 Создана новая память решений")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ LLM DECISION MEMORY")
    print("="*60 + "\n")
    
    memory = LLMDecisionMemory()
    
    # 1. Устанавливаем задачу (как будто LLM сгенерировала в интерактивном режиме)
    print("📋 Устанавливаем задачу...")
    memory.set_task(
        steps=[
            "повернуться направо",
            "сказать: 'Здравствуйте! Вам нужна помощь?'",
            "выслушать ответ",
            "если нужно — помочь, иначе — извиниться и уйти"
        ],
        reasoning="Вижу человека справа, хочу предложить помощь",
        task_name="поприветствовать человека"
    )
    
    # 2. Показываем прогресс
    print("\n📊 Прогресс задачи:")
    progress = memory.get_task_progress()
    print(f"   Задача: {progress['task_name']}")
    print(f"   Прогресс: {progress['current_step']}/{progress['total_steps']}")
    print(f"   Осталось шагов: {progress['remaining_steps']}")
    
    # 3. Выполняем шаги (эмуляция)
    print("\n🚶 Выполнение шагов:")
    
    # Шаг 1
    step = memory.get_current_step()
    print(f"   Шаг 1: {step}")
    memory.advance_step({"status": "success", "action": "turn_right"})
    
    # Шаг 2
    step = memory.get_current_step()
    print(f"   Шаг 2: {step}")
    memory.advance_step({"status": "success", "action": "speak_to_human"})
    
    # Шаг 3
    step = memory.get_current_step()
    print(f"   Шаг 3: {step}")
    memory.advance_step({"status": "success", "human_response": "Нет, спасибо, я просто жду"})
    
    # Шаг 4 (завершающий)
    step = memory.get_current_step()
    print(f"   Шаг 4: {step}")
    has_next = memory.advance_step({"status": "success", "action": "speak_to_human", "text": "Хорошо, всего доброго!"})
    
    if not has_next:
        print("\n   ✅ Задача завершена!")
    
    # 4. Статистика
    print("\n📊 СТАТИСТИКА:")
    stats = memory.get_stats()
    print(f"   Задач выполнено: {stats['tasks_completed']}")
    print(f"   Задач провалено: {stats['tasks_failed']}")
    print(f"   Активная задача: {'да' if stats['has_active_task'] else 'нет'}")
    
    # 5. История задач
    print("\n📜 ИСТОРИЯ ЗАДАЧ:")
    for task in memory.get_task_history(5):
        print(f"   - {task['name']}: {task['status']} ({len(task['history'])} шагов)")
    
    print("\n" + "="*60)
    print("Демонстрация завершена.")
    print("="*60 + "\n")
