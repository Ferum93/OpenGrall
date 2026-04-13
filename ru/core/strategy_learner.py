#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ core/strategy_learner.py - ЭВОЛЮЦИОННОЕ САМООБУЧЕНИЕ АГЕНТА                   ║
║                                                                              ║
║ ⚠️ ВНИМАНИЕ: ЭКСПЕРИМЕНТАЛЬНЫЙ МОДУЛЬ ⚠️                                      ║
║                                                                              ║
║ Этот файл — ЗАДЕЛ НА БУДУЩЕЕ. Он работает в базовых сценариях, но            ║
║ для полноценного использования требует доработки (см. раздел "ЧТО ДОДЕЛАТЬ").║
║                                                                              ║
║ ГЛАВНАЯ ЦЕЛЬ — ВЫТЕСНИТЬ LLM ИЗ ТИПОВЫХ СИТУАЦИЙ:                            ║
║                                                                              ║
║   LLM медленная и дорогая. Вызывать её для каждой команды — расточительно.   ║
║   StrategyLearner решает эту проблему:                                       ║
║                                                                              ║
║     1. Сначала LLM генерирует стратегию (например, "ехать по коридору")     ║
║     2. Стратегия выполняется, получает оценку (успех/провал)                ║
║     3. Успешные стратегии накапливают баллы                                  ║
║     4. Когда балл стратегии > 70 — она выполняется БЕЗ ВЫЗОВА LLM            ║
║     5. LLM вызывается только для НОВЫХ или ПРОВАЛЬНЫХ ситуаций               ║
║                                                                              ║
║   Результат:                                                                 ║
║   - Робот "умнеет" со временем (база стратегий растёт)                       ║
║   - Типовые действия выполняются МГНОВЕННО (без задержки LLM)                ║
║   - LLM используется только как "учитель" для новых ситуаций                 ║
║   - В отсутствие связи с LLM робот продолжает работать на заученных стратегиях║
║                                                                              ║
║   Это и есть "зачатки разума" — робот не тупит в типовых ситуациях,          ║
║   а действует по отработанным паттернам.                                     ║
║                                                                              ║
║ ЧТО ТАКОЕ СТРАТЕГИЯ:                                                         ║
║   Стратегия — это Python-код, сгенерированный LLM.                           ║
║   Она живёт в файле strategies.json и загружается при старте.                ║
║                                                                              ║
║   Пример стратегии "cruise_corridor":                                        ║
║   async def execute(self, **kwargs):                                         ║
║       while True:                                                            ║
║           sensors = self.sensor_memory.get_summaries()                        ║
║           front = sensors.get('lidar', {}).get('front', 10)                   ║
║           if front < 0.5:                                                    ║
║               await self.stop()                                              ║
║               return {"status": "blocked"}                                   ║
║           await self.move_forward(speed=300)                                 ║
║           await asyncio.sleep(0.5)                                           ║
║                                                                              ║
║   Когда эта стратегия наберёт >70 баллов, робот будет ехать по коридору      ║
║   БЕЗ вызова LLM. LLM понадобится только если стратегия провалится.          ║
║                                                                              ║
║ КАК ПРОИСХОДИТ ЭВОЛЮЦИЯ:                                                     ║
║                                                                              ║
║   1. База стратегий хранится по типам задач ("navigation", "search", ...)   ║
║   2. Для каждой задачи — до 5 стратегий (фаворит + конкуренты)              ║
║   3. Стратегии получают баллы за успех, теряют за провал                     ║
║   4. Слабые стратегии (score < 5) заменяются новыми                          ║
║   5. Новые стратегии генерирует LLM, глядя на лучшие существующие            ║
║   6. Цикл повторяется → робот САМ находит оптимальное поведение              ║
║                                                                              ║
║ ОЦЕНКА ЧЕРЕЗ LLM (OutcomeEvaluator):                                         ║
║                                                                              ║
║   После выполнения действия, OutcomeEvaluator ждёт 2 секунды и спрашивает    ║
║   LLM: "Стало лучше или хуже?" LLM возвращает оценку от -1 до 3.             ║
║   Это позволяет оценивать действия БЕЗ жёстких правил (например,             ║
║   "проехал вперёд и не врезался" — это уже частичный успех).                 ║
║                                                                              ║
║   ПОХВАЛА ОТ ЧЕЛОВЕКА:                                                       ║
║   Если человек говорит "молодец", "хорошо", "умница" — стратегия получает    ║
║   +4 балла мгновенно. Это самый сильный сигнал обучения.                     ║
║                                                                              ║
║ ЧТО ДОДЕЛАТЬ (ДОРОЖНАЯ КАРТА):                                               ║
║                                                                              ║
║   [ ] Примитивы высокого уровня:                                             ║
║       - identify_place() — понять, в какой комнате робот                     ║
║       - go_to_place() — доехать до семантического места                       ║
║       - scan_for_object() — активный поиск объекта                           ║
║                                                                              ║
║   [ ] Режим "мастер-класс" от LLM:                                           ║
║       Если все стратегии провалились — LLM берёт управление на себя,         ║
║       выполняет задачу и записывает свои действия как новую стратегию.       ║
║                                                                              ║
║   [ ] Критерий "ошибочности":                                                ║
║       Отличать слабую стратегию (иногда работает) от ошибочной               ║
║       (никогда не работает, нужно удалить).                                  ║
║                                                                              ║
║   [ ] Безопасное выполнение сгенерированного кода:                           ║
║       Сейчас код выполняется через exec(). Нужна песочница с таймаутами.    ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Изменить параметры эволюции — в config.py:                              ║
║      - MAX_STRATEGIES_PER_TASK (сколько стратегий хранить на задачу)        ║
║      - SCORE_STEP (сколько баллов за успех/провал)                           ║
║      - COMPETITOR_START_SCORE (с каким счётом стартует новый конкурент)      ║
║                                                                              ║
║   2. Добавить новый тип задачи — просто начните использовать новый intent   ║
║      в диалоге. StrategyLearner автоматически создаст категорию.             ║
║                                                                              ║
║   3. Отключить StrategyLearner — просто не создавайте его в agent_v5.py.     ║
║      Робот будет всегда использовать LLM (медленнее, но надёжнее).           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
import json
import hashlib
import random
import re
import os
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# ================================================================
# КОНСТАНТЫ (можно менять в config.py)
# ================================================================

try:
    from config import (
        MAX_STRATEGIES_PER_TASK, SELECTION_BEST_RATIO, SELECTION_EXPLORE_RATIO,
        SCORE_STEP, MAX_SCORE, MIN_SCORE, DECAY_PER_DAY, COMPETITOR_START_SCORE,
        MIN_SCORE_TO_KEEP, MIN_ATTEMPTS_TO_REPLACE, PRAISE_BONUS,
        EVALUATION_TIMEOUT, EVALUATION_DELAY
    )
except ImportError:
    MAX_STRATEGIES_PER_TASK = 5
    SELECTION_BEST_RATIO = 0.7
    SELECTION_EXPLORE_RATIO = 0.3
    SCORE_STEP = 1
    MAX_SCORE = 100
    MIN_SCORE = 0
    DECAY_PER_DAY = 1
    COMPETITOR_START_SCORE = 10
    MIN_SCORE_TO_KEEP = 5
    MIN_ATTEMPTS_TO_REPLACE = 3
    PRAISE_BONUS = 4
    EVALUATION_TIMEOUT = 60.0
    EVALUATION_DELAY = 2.0


# ================================================================
# СТРАТЕГИЯ (основная единица эволюции)
# ================================================================

@dataclass
class Strategy:
    """
    ОДНА СТРАТЕГИЯ — ИСПОЛНЯЕМЫЙ КОД С МЕТРИКАМИ УСПЕШНОСТИ
    
    Стратегия — это Python-функция, которую LLM сгенерировала для решения
    конкретной задачи. Она хранится в базе и соревнуется с другими стратегиями.
    
    Атрибуты:
        - id: уникальный идентификатор
        - task_type: тип задачи ("navigation", "search", "conversation")
        - name: человекочитаемое имя
        - description: что делает стратегия
        - code: Python-код (строка)
        - score: текущий счёт (0-100)
        - attempts: сколько раз выполнялась
        - successes: сколько раз была успешной
        - generation: поколение (сколько раз эволюционировала)
        - parent_id: от какой стратегии произошла (если есть)
    """
    id: str
    task_type: str
    name: str
    description: str
    code: str
    score: int = 50
    attempts: int = 0
    successes: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    generation: int = 1
    parent_id: Optional[str] = None
    consecutive_failures: int = 0      # сколько раз подряд провалилась
    invalid: bool = False              # помечена как ошибочная
    invalid_reason: str = ""           # почему признана ошибочной

    def success_rate(self) -> float:
        """Возвращает долю успешных выполнений (0.0 - 1.0)"""
        return self.successes / self.attempts if self.attempts > 0 else 0.5

    def get_current_score(self) -> int:
        """Возвращает текущий счёт с учётом затухания от неиспользования"""
        days_since_use = (time.time() - self.last_used) / 86400
        decay = DECAY_PER_DAY * days_since_use
        current = self.score - decay
        return max(MIN_SCORE, min(MAX_SCORE, int(current)))

    def update_score(self, success: bool):
        """Обновляет счёт после выполнения"""
        self.attempts += 1
        self.last_used = time.time()
        
        if success:
            self.successes += 1
            self.score = min(MAX_SCORE, self.score + SCORE_STEP)
            self.consecutive_failures = 0
        else:
            self.score = max(MIN_SCORE, self.score - SCORE_STEP)
            self.consecutive_failures += 1
        
        logger.debug(f"📊 {self.name}: score={self.score} ({self.successes}/{self.attempts})")

    def add_bonus(self, bonus: int):
        """Добавляет бонусные баллы (похвала от человека)"""
        self.score = min(MAX_SCORE, self.score + bonus)
        logger.debug(f"🎉 {self.name}: бонус +{bonus}, score={self.score}")

    def add_penalty(self, penalty: int):
        """Добавляет штраф (серьёзная ошибка)"""
        self.score = max(MIN_SCORE, self.score - penalty)
        logger.debug(f"⚠️ {self.name}: штраф -{penalty}, score={self.score}")

    def is_weak(self) -> bool:
        """Стратегия слабая и подлежит замене?"""
        return (self.get_current_score() < MIN_SCORE_TO_KEEP and
                self.attempts >= MIN_ATTEMPTS_TO_REPLACE)

    def mark_invalid(self, reason: str):
        """Помечает стратегию как ошибочную (никогда не работает)"""
        self.invalid = True
        self.invalid_reason = reason
        self.score = 0
        logger.warning(f"❌ Стратегия {self.name} помечена как ошибочная: {reason}")

    def to_dict(self) -> Dict:
        """Сериализует стратегию для сохранения в JSON"""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "score": self.score,
            "attempts": self.attempts,
            "successes": self.successes,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "consecutive_failures": self.consecutive_failures,
            "invalid": self.invalid,
            "invalid_reason": self.invalid_reason
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Strategy':
        """Восстанавливает стратегию из JSON"""
        return cls(
            id=data["id"],
            task_type=data["task_type"],
            name=data["name"],
            description=data["description"],
            code=data["code"],
            score=data.get("score", 50),
            attempts=data.get("attempts", 0),
            successes=data.get("successes", 0),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", time.time()),
            generation=data.get("generation", 1),
            parent_id=data.get("parent_id"),
            consecutive_failures=data.get("consecutive_failures", 0),
            invalid=data.get("invalid", False),
            invalid_reason=data.get("invalid_reason", "")
        )


# ================================================================
# РЕЗУЛЬТАТ ОЦЕНКИ
# ================================================================

@dataclass
class EvaluationResult:
    """Результат оценки выполнения действия через LLM"""
    score: int                    # -1 до 3 (или 4 за похвалу)
    goal_achieved: bool
    reasoning: str
    next_steps: Optional[str] = None
    evaluation_time: float = 0.0


# ================================================================
# ОЦЕНЩИК РЕЗУЛЬТАТОВ (ВЫЗЫВАЕТ LLM ДЛЯ ОЦЕНКИ ДЕЙСТВИЙ)
# ================================================================

class OutcomeEvaluator:
    """
    ОЦЕНЩИК УСПЕШНОСТИ ДЕЙСТВИЙ ЧЕРЕЗ LLM
    
    Работает в фоне, не блокирует основной цикл.
    Учитывает текущие сенсоры и VLM.
    Поддерживает мгновенную похвалу от человека.
    """
    
    def __init__(self, strategy_learner):
        self.strategy_learner = strategy_learner
        self.agent = None
        self.pending_evaluations: Dict[str, Dict] = {}
        self.evaluation_queue = asyncio.Queue()
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        
        logger.info("✅ OutcomeEvaluator инициализирован")
    
    def set_agent(self, agent):
        """Устанавливает ссылку на агента (после создания)"""
        self.agent = agent
    
    async def start(self):
        """Запускает фоновый обработчик оценок"""
        self.is_running = True
        self.task = asyncio.create_task(self._process_queue())
        logger.info("🔄 OutcomeEvaluator запущен")
    
    async def stop(self):
        """Останавливает обработчик"""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 OutcomeEvaluator остановлен")
    
    def register_action(self, action_id: str, intent: str,
                        strategy_id: str, task_type: str,
                        context: Dict[str, Any]):
        """Регистрирует действие для последующей оценки"""
        self.pending_evaluations[action_id] = {
            "start_time": time.time(),
            "intent": intent,
            "strategy_id": strategy_id,
            "task_type": task_type,
            "context": context,
            "status": "pending"
        }
        logger.debug(f"📝 Зарегистрировано действие {action_id}: {intent}")
    
    async def evaluate(self, action_id: str,
                       forced: bool = False,
                       human_praise: bool = False) -> Optional[EvaluationResult]:
        """
        Оценивает результат действия
        
        Args:
            action_id: ID действия
            forced: принудительная оценка (по таймауту)
            human_praise: это оценка от человека (похвала)
        """
        if action_id not in self.pending_evaluations:
            logger.warning(f"⚠️ Действие {action_id} не найдено")
            return None
        
        eval_data = self.pending_evaluations[action_id]
        
        # Мгновенная похвала от человека
        if human_praise:
            logger.info(f"🎉 ПОХВАЛА! Действие {action_id} получает +{PRAISE_BONUS}")
            result = EvaluationResult(
                score=PRAISE_BONUS,
                goal_achieved=True,
                reasoning="Похвала от человека",
                next_steps=None
            )
            self._apply_result(action_id, result)
            return result
        
        # Оценка через LLM
        try:
            result = await self._evaluate_with_llm(action_id, eval_data, forced)
            self._apply_result(action_id, result)
            return result
        except Exception as e:
            logger.error(f"❌ Ошибка оценки {action_id}: {e}")
            result = EvaluationResult(
                score=0,
                goal_achieved=False,
                reasoning=f"Ошибка оценки: {e}"
            )
            self._apply_result(action_id, result)
            return result
    
    async def _evaluate_with_llm(self, action_id: str,
                                  eval_data: Dict,
                                  forced: bool) -> EvaluationResult:
        """Оценивает действие через LLM"""
        
        if not self.agent:
            logger.warning("Agent не установлен в OutcomeEvaluator")
            return EvaluationResult(score=0, goal_achieved=False, reasoning="Нет агента")
        
        # Собираем текущие сенсоры
        sensor_summary = self.agent.context_builder.get_last_sensors_summary()
        
        # Время выполнения
        elapsed = time.time() - eval_data["start_time"]
        
        # Формируем промпт для оценки
        prompt = f"""Ты оцениваешь результат выполнения действия роботом.

ДЕЙСТВИЕ:
{action_id}
Намерение: {eval_data['intent']}
Время выполнения: {elapsed:.1f} секунд

ТЕКУЩАЯ СИТУАЦИЯ:
{sensor_summary}

ИСХОДНЫЙ КОНТЕКСТ:
{self._format_context(eval_data['context'])}

Оцени результат от -1 до 3:
- -1: действие навредило, стало хуже
- 0: ничего не изменилось
- 1: небольшой прогресс
- 2: хороший прогресс, близко к цели
- 3: цель достигнута

Ответь ТОЛЬКО JSON:
{{
    "score": целое_число_от_-1_до_3,
    "goal_achieved": true/false,
    "reasoning": "почему такая оценка",
    "next_steps": "что делать дальше (если цель не достигнута)"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.agent.llm.generate(messages)
        
        content = response.content if hasattr(response, 'content') else str(response)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return EvaluationResult(
                    score=max(-1, min(3, data.get("score", 0))),
                    goal_achieved=data.get("goal_achieved", False),
                    reasoning=data.get("reasoning", ""),
                    next_steps=data.get("next_steps"),
                    evaluation_time=elapsed
                )
            except json.JSONDecodeError:
                pass
        
        return EvaluationResult(
            score=0,
            goal_achieved=False,
            reasoning="Не удалось распарсить ответ LLM"
        )
    
    def _apply_result(self, action_id: str, result: EvaluationResult):
        """Применяет результат оценки к стратегии"""
        eval_data = self.pending_evaluations.get(action_id)
        if not eval_data:
            return
        
        strategy_id = eval_data.get("strategy_id")
        task_type = eval_data.get("task_type")
        
        if strategy_id and self.strategy_learner:
            for strategy in self.strategy_learner.strategies.get(task_type, []):
                if strategy.id == strategy_id:
                    if result.score >= 3:
                        strategy.update_score(True)
                        strategy.add_bonus(result.score - 1)
                        logger.info(f"🏆 Стратегия {strategy.name} получила бонус +{result.score - 1}")
                    elif result.score >= 1:
                        strategy.update_score(True)
                    elif result.score == 0:
                        pass
                    else:
                        strategy.update_score(False)
                        if result.score == -1:
                            strategy.add_penalty(1)
                    break
        
        del self.pending_evaluations[action_id]
    
    def _format_context(self, context: Dict) -> str:
        """Форматирует контекст для промпта"""
        lines = []
        if context.get("current_intent"):
            lines.append(f"Намерение: {context['current_intent']}")
        for s in context.get("sensors", [])[:3]:
            lines.append(f"{s.get('source_type')}: {s.get('summary', '')[:100]}")
        return "\n".join(lines) if lines else "нет данных"
    
    async def _process_queue(self):
        """Фоновая обработка очереди оценок"""
        while self.is_running:
            try:
                action_id = await asyncio.wait_for(self.evaluation_queue.get(), timeout=1.0)
                await self.evaluate(action_id)
            except asyncio.TimeoutError:
                await self._check_timeouts()
                continue
            except Exception as e:
                logger.error(f"Ошибка в очереди оценок: {e}")
    
    async def _check_timeouts(self):
        """Проверяет действия, которые выполняются дольше таймаута"""
        now = time.time()
        timeouts = []
        for action_id, data in self.pending_evaluations.items():
            if data["status"] == "pending":
                elapsed = now - data["start_time"]
                if elapsed > EVALUATION_TIMEOUT:
                    timeouts.append((action_id, elapsed))
        
        for action_id, elapsed in timeouts:
            logger.info(f"⏰ Таймаут действия {action_id} ({elapsed:.0f}с), оцениваю...")
            await self.evaluate(action_id, forced=True)
    
    def human_praise(self, text: str) -> bool:
        """Проверяет, содержит ли текст похвалу"""
        praise_words = [
            "молодец", "хорошо", "отлично", "прекрасно", "супер",
            "умница", "класс", "здорово", "так держать", "хвалю",
            "great", "good", "excellent", "well done", "nice"
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in praise_words)
    
    def get_pending_count(self) -> int:
        return len(self.pending_evaluations)
    
    def get_stats(self) -> Dict:
        return {
            "pending": len(self.pending_evaluations),
            "queue_size": self.evaluation_queue.qsize(),
            "is_running": self.is_running
        }


# ================================================================
# ГЛАВНЫЙ КЛАСС — ЭВОЛЮЦИОННОЕ САМООБУЧЕНИЕ
# ================================================================

class StrategyLearner:
    """
    ЭВОЛЮЦИОННОЕ САМООБУЧЕНИЕ АГЕНТА
    
    Хранит базу стратегий, оценивает их, создаёт новые, заменяет слабые.
    Главная цель — со временем вытеснить LLM из типовых ситуаций.
    """
    
    def __init__(self, llm_client, storage_path: str):
        self.llm = llm_client
        self.storage_path = storage_path
        self.strategies: Dict[str, List[Strategy]] = defaultdict(list)
        self.lock = threading.Lock()
        
        self.evaluator = OutcomeEvaluator(self)
        
        self.stats = {
            "learning_sessions": 0,
            "strategies_generated": 0,
            "strategies_evolved": 0,
            "strategies_removed": 0,
            "best_score": 0,
            "best_strategy": "",
            "last_learning": 0,
            "self_tasks_generated": 0,
            "self_tasks_success": 0,
            "evaluations_completed": 0,
            "praise_received": 0
        }
        
        self.self_task_prompt_template = """Ты — {role}.

Твоё описание: {self_description}

ТЕКУЩАЯ ОБСТАНОВКА:
{sensor_summary}

ПАМЯТЬ:
- Исследованные зоны: {explored_areas}
- Интересные объекты: {interesting_objects}
- Недавние события: {recent_events}

У тебя есть инструменты: {available_tools}

Придумай себе задачу, которую ты хочешь выполнить.
Задача должна быть выполнимой и безопасной.

Ответь ТОЛЬКО JSON:
{{
    "task_name": "название задачи",
    "reasoning": "почему это хорошая задача сейчас",
    "steps": ["шаг1", "шаг2", "шаг3"],
    "expected_outcome": "что должно получиться"
}}"""
        
        self.load()
        logger.info(f"✅ StrategyLearner готов (max={MAX_STRATEGIES_PER_TASK} стратегий на задачу)")
    
    # ================================================================
    # ВЫБОР СТРАТЕГИИ
    # ================================================================
    
    def select_strategy(self, task_type: str) -> Optional[Strategy]:
        """
        ВЫБИРАЕТ ЛУЧШУЮ СТРАТЕГИЮ ДЛЯ ЗАДАЧИ
        
        70% — фаворит (лучшая стратегия)
        30% — исследование (случайная из топ-кандидатов)
        
        Это баланс между "использовать проверенное" и "пробовать новое".
        """
        with self.lock:
            strategies = self.strategies.get(task_type, [])
            # Отфильтровываем помеченные как ошибочные
            valid_strategies = [s for s in strategies if not s.invalid]
            if not valid_strategies:
                return None
            
            valid_strategies.sort(key=lambda s: s.get_current_score(), reverse=True)
            best = valid_strategies[0]
            best_score = best.get_current_score()
            candidates = [s for s in valid_strategies if s.get_current_score() >= best_score - 10]
            
            if len(candidates) > 1 and random.random() < SELECTION_EXPLORE_RATIO:
                non_best = [s for s in candidates if s.id != best.id]
                if non_best:
                    selected = random.choice(non_best)
                    logger.info(f"🔬 Исследование: {selected.name} (score={selected.get_current_score()})")
                    return selected
            
            logger.info(f"🎯 Эксплуатация: {best.name} (score={best_score})")
            return best
    
    def update_score(self, strategy: Strategy, success: bool):
        """Обновляет счёт стратегии и сохраняет базу"""
        with self.lock:
            strategy.update_score(success)
            current_score = strategy.get_current_score()
            if current_score > self.stats["best_score"]:
                self.stats["best_score"] = current_score
                self.stats["best_strategy"] = strategy.name
            self.save()
    
    # ================================================================
    # ЭВОЛЮЦИЯ В ПРОСТОЕ
    # ================================================================
    
    async def learn_in_idle(self, idle_time: float):
        """
        ЗАПУСКАЕТ ЭВОЛЮЦИЮ В ПРОСТОЕ
        
        Вызывается агентом, когда робот долго бездействует.
        Создаёт новые стратегии или улучшает существующие.
        """
        logger.info(f"📚 Простой {idle_time:.0f}с. Начинаю эволюцию...")
        
        for task_type in list(self.strategies.keys()):
            strategies = [s for s in self.strategies[task_type] if not s.invalid]
            
            if len(strategies) < MAX_STRATEGIES_PER_TASK:
                needed = MAX_STRATEGIES_PER_TASK - len(strategies)
                await self._generate_new_strategies(task_type, needed)
            elif len(strategies) == 1:
                await self._create_competitor(task_type, strategies[0])
            else:
                weak = [s for s in strategies if s.is_weak()]
                for weak_strategy in weak:
                    await self._evolve_strategy(task_type, weak_strategy)
        
        self.stats["last_learning"] = time.time()
        self.stats["learning_sessions"] += 1
        self.save()
        logger.info("✅ Эволюция завершена")
    
    async def _generate_new_strategies(self, task_type: str, count: int):
        """Генерирует новые стратегии через LLM"""
        logger.info(f"🌱 Генерация {count} новых стратегий для '{task_type}'")
        
        prompt = f"""
Ты эксперт по робототехнике. Задача: {task_type}

Сгенерируй {count} РАЗНЫХ стратегий выполнения этой задачи.
Используй доступные инструменты: self.ws, self.sensor_memory, self.vlm, self.tts.

Каждая стратегия — это асинхронная функция execute(self, **kwargs).
Она должна быть адаптивной (использовать сенсоры для принятия решений).

Формат ответа (JSON массив):
[
    {{
        "name": "strategy_name",
        "description": "что делает",
        "code": "async def execute(self, **kwargs):\\n    # код"
    }}
]
"""
        try:
            strategies_data = await self._ask_llm(prompt)
            for data in strategies_data[:count]:
                self._add_strategy(task_type, data, generation=1)
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
    
    async def _create_competitor(self, task_type: str, sole_survivor: Strategy):
        """Создаёт конкурента для единственной стратегии"""
        logger.info(f"⚔️ Создаю конкурента для {sole_survivor.name}")
        
        prompt = f"""
Единственная стратегия для "{task_type}": {sole_survivor.name}
{sole_survivor.code}

Создай АЛЬТЕРНАТИВНУЮ стратегию, которая принципиально отличается.

Формат ответа:
{{
    "name": "alternative_{sole_survivor.name}",
    "description": "альтернативная стратегия",
    "code": "async def execute(self, **kwargs):\\n    # код"
}}
"""
        try:
            data = await self._ask_llm(prompt)
            self._add_strategy(task_type, data,
                              generation=sole_survivor.generation + 1,
                              parent_id=sole_survivor.id,
                              initial_score=COMPETITOR_START_SCORE)
        except Exception as e:
            logger.error(f"Ошибка создания конкурента: {e}")
    
    async def _evolve_strategy(self, task_type: str, weak_strategy: Strategy):
        """Эволюционирует слабую стратегию"""
        logger.info(f"🧬 Эволюция: {weak_strategy.name}")
        
        strategies = [s for s in self.strategies[task_type] if not s.invalid]
        best = max(strategies, key=lambda s: s.get_current_score())
        
        prompt = f"""
Слабая стратегия ({weak_strategy.get_current_score()} баллов):
{weak_strategy.code}

Лучшая стратегия ({best.get_current_score()} баллов):
{best.code}

Создай УЛУЧШЕННУЮ стратегию, объединяющую лучшие идеи.

Формат ответа:
{{
    "name": "evolved_{weak_strategy.name}",
    "description": "улучшенная версия",
    "code": "async def execute(self, **kwargs):\\n    # код"
}}
"""
        try:
            data = await self._ask_llm(prompt)
            
            with self.lock:
                self.strategies[task_type] = [s for s in self.strategies[task_type]
                                              if s.id != weak_strategy.id]
                self.stats["strategies_removed"] += 1
            
            self._add_strategy(task_type, data,
                              generation=best.generation + 1,
                              parent_id=best.id,
                              initial_score=best.get_current_score())
            self.stats["strategies_evolved"] += 1
        except Exception as e:
            logger.error(f"Ошибка эволюции: {e}")
    
    def _add_strategy(self, task_type: str, data: Dict, generation: int = 1,
                      parent_id: Optional[str] = None, initial_score: Optional[int] = None):
        """Добавляет новую стратегию в базу"""
        strategy_id = hashlib.md5(f"{task_type}_{data['name']}_{time.time()}".encode()).hexdigest()[:8]
        
        strategy = Strategy(
            id=strategy_id,
            task_type=task_type,
            name=data["name"],
            description=data.get("description", ""),
            code=data["code"],
            score=initial_score if initial_score is not None else 50,
            generation=generation,
            parent_id=parent_id
        )
        
        with self.lock:
            self.strategies[task_type].append(strategy)
            if len(self.strategies[task_type]) > MAX_STRATEGIES_PER_TASK:
                valid = [s for s in self.strategies[task_type] if not s.invalid]
                if len(valid) > MAX_STRATEGIES_PER_TASK:
                    weakest = min(valid, key=lambda s: s.get_current_score())
                    self.strategies[task_type].remove(weakest)
                    self.stats["strategies_removed"] += 1
        
        self.stats["strategies_generated"] += 1
        logger.info(f"➕ Добавлена стратегия: {strategy.name}")
    
    # ================================================================
    # АВТОНОМНОЕ ЦЕЛЕПОЛАГАНИЕ
    # ================================================================
    
    async def generate_self_task(self,
                                  sensor_summary: str,
                                  explored_areas: List[str],
                                  interesting_objects: List[str],
                                  recent_events: List[str],
                                  available_tools: List[str],
                                  role: str = "",
                                  self_description: str = "") -> Optional[Dict[str, Any]]:
        """Генерирует задачу для автономного выполнения"""
        if not role:
            role = "автономный агент"
        
        prompt = self.self_task_prompt_template.format(
            role=role,
            self_description=self_description or "Я могу двигаться, наблюдать, взаимодействовать",
            sensor_summary=sensor_summary,
            explored_areas=explored_areas or ["нет данных"],
            interesting_objects=interesting_objects or ["нет"],
            recent_events=recent_events or ["нет"],
            available_tools=available_tools or ["нет"]
        )
        
        try:
            response = await self._ask_llm(prompt)
            if isinstance(response, dict) and "task_name" in response and "steps" in response:
                self.stats["self_tasks_generated"] += 1
                logger.info(f"🎯 Сгенерирована задача: {response['task_name']}")
                return response
        except Exception as e:
            logger.error(f"Ошибка генерации задачи: {e}")
        
        return None
    
    # ================================================================
    # ИНТЕГРАЦИЯ С АГЕНТОМ
    # ================================================================
    
    def register_action_for_evaluation(self, action_id: str, intent: str,
                                        strategy_id: str, task_type: str,
                                        context: Dict[str, Any]):
        """Регистрирует действие для оценки"""
        self.evaluator.register_action(action_id, intent, strategy_id, task_type, context)
        self.stats["evaluations_completed"] += 1
    
    def check_praise(self, text: str) -> bool:
        """Проверяет, содержит ли текст похвалу"""
        return self.evaluator.human_praise(text)
    
    async def start_evaluator(self):
        """Запускает фоновый оценщик"""
        await self.evaluator.start()
    
    async def stop_evaluator(self):
        """Останавливает оценщик"""
        await self.evaluator.stop()
    
    # ================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    async def _ask_llm(self, prompt: str) -> Dict:
        """Отправляет запрос LLM и парсит JSON"""
        response = await self.llm.generate([{"role": "user", "content": prompt}])
        content = response.content if hasattr(response, 'content') else str(response)
        
        json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Не удалось извлечь JSON из ответа: {content[:200]}")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику"""
        with self.lock:
            stats = self.stats.copy()
            stats["total_strategies"] = sum(len(s) for s in self.strategies.values())
            stats["tasks"] = {task: len(s) for task, s in self.strategies.items()}
            stats["evaluator"] = self.evaluator.get_stats()
            return stats
    
    def save(self):
        """Сохраняет базу стратегий в файл"""
        try:
            data = {
                "strategies": {
                    task: [s.to_dict() for s in strategies]
                    for task, strategies in self.strategies.items()
                },
                "stats": self.stats
            }
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
    
    def load(self):
        """Загружает базу стратегий из файла"""
        if not os.path.exists(self.storage_path):
            logger.info("🆕 Нет сохранённых стратегий, начинаю с нуля")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                self.strategies.clear()
                for task, strategies_data in data.get("strategies", {}).items():
                    self.strategies[task] = [Strategy.from_dict(s) for s in strategies_data]
                self.stats.update(data.get("stats", {}))
            
            logger.info(f"📂 Загружено {sum(len(s) for s in self.strategies.values())} стратегий")
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ STRATEGY LEARNER")
    print("="*60 + "\n")
    
    print("📚 StrategyLearner — это экспериментальный модуль.")
    print("   Он хранит стратегии, оценивает их и эволюционирует.")
    print("   В реальном роботе он подключается к LLM через агента.")
    print("\n   Пример стратегии (cruise_corridor):")
    print("   " + "-" * 50)
    print("""   async def execute(self, **kwargs):
       while True:
           sensors = self.sensor_memory.get_summaries()
           front = sensors.get('lidar', {}).get('front', 10)
           if front < 0.5:
               await self.stop()
               return {"status": "blocked"}
           await self.move_forward(speed=300)
           await asyncio.sleep(0.5)
    """)
    print("   " + "-" * 50)
    print("\n   Когда стратегия набирает >70 баллов — она выполняется БЕЗ LLM.")
    print("   Это ускоряет робота и даёт ему 'зачатки разума'.\n")
    
    print("="*60)
    print("Для полноценной работы нужен agent и LLM.")
    print("="*60 + "\n")
