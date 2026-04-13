#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ memory/episodic_memory.py - ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ СОБЫТИЙ                      ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это "дневник" робота. Он запоминает ВСЁ, что с ним происходит:             ║
║   - Рефлексы TinyML (где и почему остановился)                               ║
║   - Команды человека и результаты их выполнения                              ║
║   - Разговоры (что спросили, что ответил)                                    ║
║   - Наблюдения (что видел, куда ездил)                                       ║
║   - И САМОЕ ВАЖНОЕ — инструкции человека                                     ║
║                                                                              ║
║ ГЛАВНАЯ ФИШКА — ОБУЧЕНИЕ У ЧЕЛОВЕКА:                                         ║
║                                                                              ║
║   Робот столкнулся с непонятной ситуацией и спросил:                         ║
║       "Я вижу оранжевый ботинок. Это тот, что вы искали?"                    ║
║   Человек ответил:                                                           ║
║       "Нет, мне нужен чёрный ботинок, он обычно в прихожей"                  ║
║                                                                              ║
║   Эта инструкция сохраняется ВМЕСТЕ С КОНТЕКСТОМ (что видел робот,           ║
║   где находился, какое было намерение).                                      ║
║                                                                              ║
║   Через неделю робот снова ищет ботинок в похожей ситуации.                  ║
║   EpisodicMemory находит ту инструкцию и ВСПОМИНАЕТ:                         ║
║       "В прошлый раз человек сказал, что чёрный ботинок в прихожей!"         ║
║                                                                              ║
║   Робот НЕ переспрашивает. Он СРАЗУ едет в прихожую.                         ║
║   Это и есть "обучение у человека" — робот становится умнее с каждым диалогом.║
║                                                                              ║
║ КАК РАБОТАЕТ ПАМЯТЬ:                                                         ║
║                                                                              ║
║   1. Каждое событие — это Episode (эпизод)                                   ║
║   2. У эпизода есть:                                                         ║
║      - тип (reflex, command, conversation, human_instruction)                ║
║      - важность (importance, 0-1)                                            ║
║      - контекст (что видели сенсоры в этот момент)                           ║
║      - теги (для быстрого поиска)                                            ║
║   3. Эпизоды ЗАТУХАЮТ со временем (DECAY_FACTOR)                             ║
║   4. Старые и неважные эпизоды АВТОМАТИЧЕСКИ ЗАБЫВАЮТСЯ                      ║
║   5. При поиске учитывается и свежесть, и важность, и похожесть контекста    ║
║                                                                              ║
║ ПРИМЕР — КАК ИНСТРУКЦИЯ ВСПЛЫВАЕТ В НУЖНЫЙ МОМЕНТ:                           ║
║                                                                              ║
║   Текущий контекст:                                                          ║
║       intent: "find_boot"                                                    ║
║       objects: ["ботинок", "коридор"]                                        ║
║       location: "возле двери"                                                ║
║                                                                              ║
║   В памяти есть эпизод (сохранён 3 дня назад):                               ║
║       type: "human_instruction"                                              ║
║       question: "Я вижу оранжевый ботинок. Это тот?"                         ║
║       answer: "Нет, чёрный ботинок в прихожей"                               ║
║       context: {intent: "find_boot", objects: ["ботинок"], location: "дверь"}║
║                                                                              ║
║   _context_similarity() сравнивает контексты:                                ║
║       - intent совпадает (find_boot)                                         ║
║       - объекты похожи (ботинок)                                             ║
║       - локация похожа (дверь)                                               ║
║       → similarity = 0.85 (выше порога 0.6)                                  ║
║                                                                              ║
║   Инструкция ВСПЛЫВАЕТ, и робот едет в прихожую БЕЗ лишних вопросов.         ║
║                                                                              ║
║ ТИПЫ ЭПИЗОДОВ:                                                               ║
║   - reflex: срабатывание TinyML (важность 0.9)                               ║
║   - command: выполнение команды (0.7)                                        ║
║   - error: ошибка при выполнении (0.85)                                      ║
║   - conversation: диалог с человеком (0.5)                                   ║
║   - observation: наблюдение за окружением (0.3)                              ║
║   - plan: выполненный план действий (0.8)                                    ║
║   - human_instruction: инструкция от человека (0.8)                           ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from memory.episodic_memory import EpisodicMemory                          ║
║                                                                              ║
║   memory = EpisodicMemory(storage_path="episodes.json")                      ║
║                                                                              ║
║   # Сохранить инструкцию человека                                            ║
║   memory.add_human_instruction(                                              ║
║       question="Где лежит пульт?",                                           ║
║       answer="На журнальном столике",                                        ║
║       context={"intent": "find_remote", "room": "living_room"}               ║
║   )                                                                          ║
║                                                                              ║
║   # Позже, в похожей ситуации — найти инструкцию                             ║
║   instruction = memory.get_instruction_for_context(                          ║
║       context={"intent": "find_remote", "room": "living_room"}               ║
║   )                                                                          ║
║   if instruction:                                                            ║
║       print(f"Вспомнил: {instruction['answer']}")                            ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Изменить время жизни эпизодов:                                          ║
║      EPISODE_TTL = 3600 * 24 * 30  # 30 дней вместо 7                        ║
║                                                                              ║
║   2. Изменить скорость забывания:                                            ║
║      DECAY_FACTOR = 0.999  # медленнее (0.99 — быстрее)                      ║
║                                                                              ║
║   3. Добавить новый тип эпизода:                                             ║
║      - Создайте метод add_xxx() по аналогии                                  ║
║      - Определите важность и теги по умолчанию                               ║
║                                                                              ║
║   4. Улучшить сравнение контекстов:                                          ║
║      - Допишите _context_similarity() под свои сенсоры                       ║
║      - Добавьте сравнение локации, времени суток, etc.                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import uuid
import json
import os
import threading
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

# ================================================================
# КОНСТАНТЫ (можно менять под себя)
# ================================================================

MAX_EPISODES = 1000                # максимум эпизодов в памяти
DECAY_FACTOR = 0.99                # экспоненциальное затухание веса (в секунду)
MIN_WEIGHT = 0.01                  # минимальный вес (ниже — эпизод "забыт")
EPISODE_TTL = 3600 * 24 * 7        # время жизни эпизода (7 дней в секундах)


@dataclass
class Episode:
    """
    ОДИН ЭПИЗОД — ЕДИНИЦА ПАМЯТИ
    
    Это "запись в дневнике" о том, что произошло с роботом.
    Каждый эпизод имеет вес, который со временем затухает.
    Важные эпизоды живут дольше, неважные забываются быстрее.
    """
    episode_id: str
    timestamp: float
    description: str
    episode_type: str      # reflex, command, observation, conversation, error, plan, human_instruction
    importance: float = 0.5
    weight: float = 1.0    # начальный вес (будет затухать)
    context: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def get_age(self, current_time: float = None) -> float:
        """Возраст эпизода в секундах"""
        if current_time is None:
            current_time = time.time()
        return current_time - self.timestamp
    
    def get_current_weight(self, current_time: float = None) -> float:
        """
        ТЕКУЩИЙ ВЕС ЭПИЗОДА С УЧЁТОМ ЗАТУХАНИЯ
        
        Формула: вес = начальный_вес × (DECAY_FACTOR ^ возраст) × важность
        
        Пример:
            Эпизод создан с weight=1.0, importance=0.8
            Возраст = 1 час (3600 сек)
            DECAY_FACTOR = 0.99
            вес = 1.0 × (0.99 ^ 3600) × 0.8 ≈ 0.0 (почти забыт)
            
            Тот же эпизод с importance=1.0 и age=60 сек:
            вес = 1.0 × (0.99 ^ 60) × 1.0 ≈ 0.55
        """
        if current_time is None:
            current_time = time.time()
        
        age = current_time - self.timestamp
        decay = DECAY_FACTOR ** age
        current_weight = self.weight * decay * self.importance
        return max(MIN_WEIGHT, min(1.0, current_weight))
    
    def is_expired(self, current_time: float = None) -> bool:
        """Истёк ли срок жизни эпизода (TTL)"""
        if current_time is None:
            current_time = time.time()
        return (current_time - self.timestamp) > EPISODE_TTL
    
    def is_recent(self, max_age: float = 60.0) -> bool:
        """Произошло ли событие недавно (в пределах max_age секунд)"""
        return self.get_age() < max_age
    
    def to_dict(self) -> Dict:
        """Сериализация для сохранения в JSON"""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "description": self.description,
            "episode_type": self.episode_type,
            "importance": self.importance,
            "weight": self.weight,
            "context": self.context,
            "duration": self.duration,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Episode':
        """Восстановление из JSON"""
        return cls(
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
            description=data["description"],
            episode_type=data["episode_type"],
            importance=data.get("importance", 0.5),
            weight=data.get("weight", 1.0),
            context=data.get("context", {}),
            duration=data.get("duration", 0.0),
            tags=data.get("tags", [])
        )


class EpisodicMemory:
    """
    ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ — ДНЕВНИК РОБОТА
    
    Хранит события, умеет искать по контексту, автоматически забывает старое.
    Главная фишка — инструкции человека, которые всплывают в похожих ситуациях.
    """
    
    def __init__(self, max_episodes: int = MAX_EPISODES, storage_path: Optional[str] = None):
        self.max_episodes = max_episodes
        self.storage_path = storage_path
        
        # Основное хранилище: episode_id -> Episode
        self.episodes: Dict[str, Episode] = {}
        
        # Хронология (для быстрого доступа к последним событиям)
        self.timeline = deque(maxlen=max_episodes)
        
        # Индексы для быстрого поиска
        self.type_index: Dict[str, List[str]] = {}
        self.tag_index: Dict[str, List[str]] = {}
        
        self.lock = threading.Lock()
        
        self.stats = {
            "total_episodes": 0,
            "by_type": {},
            "oldest": 0,
            "newest": 0,
            "last_cleanup": time.time(),
            "human_instructions": 0
        }
        
        if storage_path:
            self.load_from_file(storage_path)
        
        logger.info(f"✅ EpisodicMemory инициализирована (макс: {max_episodes} эпизодов)")
    
    # ================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ================================================================
    
    def add_episode(self, 
                    description: str,
                    episode_type: str,
                    importance: float = 0.5,
                    context: Dict[str, Any] = None,
                    duration: float = 0.0,
                    tags: List[str] = None) -> str:
        """
        ДОБАВЛЯЕТ НОВЫЙ ЭПИЗОД В ПАМЯТЬ
        
        Args:
            description: что произошло (коротко)
            episode_type: тип события (reflex, command, conversation, ...)
            importance: важность (0-1). Важные события живут дольше.
            context: контекст (сенсоры, намерение, локация)
            duration: длительность события (если применимо)
            tags: теги для поиска
        
        Returns:
            episode_id: уникальный идентификатор эпизода
        """
        episode_id = str(uuid.uuid4())
        timestamp = time.time()
        
        episode = Episode(
            episode_id=episode_id,
            timestamp=timestamp,
            description=description,
            episode_type=episode_type,
            importance=importance,
            weight=1.0,
            context=context or {},
            duration=duration,
            tags=tags or []
        )
        
        with self.lock:
            self.episodes[episode_id] = episode
            self.timeline.append(episode_id)
            
            # Индекс по типу
            if episode_type not in self.type_index:
                self.type_index[episode_type] = []
            self.type_index[episode_type].append(episode_id)
            
            # Индекс по тегам
            for tag in episode.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(episode_id)
            
            self._cleanup_if_needed()
            
            # Статистика
            self.stats["total_episodes"] = len(self.episodes)
            self.stats["by_type"][episode_type] = self.stats["by_type"].get(episode_type, 0) + 1
            self.stats["newest"] = timestamp
            if self.stats["oldest"] == 0:
                self.stats["oldest"] = timestamp
        
        logger.debug(f"📝 Добавлен эпизод [{episode_type}]: {description[:50]}...")
        
        # Периодическое сохранение
        if self.storage_path and len(self.episodes) % 10 == 0:
            self.save_to_file(self.storage_path)
        
        return episode_id
    
    def recall(self, 
               query: Optional[str] = None,
               episode_type: Optional[str] = None,
               min_importance: float = 0.0,
               min_weight: float = 0.0,
               max_age: Optional[float] = None,
               limit: int = 10,
               tags: List[str] = None) -> List[Episode]:
        """
        ИЗВЛЕКАЕТ ЭПИЗОДЫ ИЗ ПАМЯТИ ПО ФИЛЬТРАМ
        
        Это основной метод поиска. Можно фильтровать по:
        - типу события
        - важности
        - свежести (max_age)
        - тегам
        - текстовому запросу в описании
        
        Returns:
            Список эпизодов, отсортированных от новых к старым
        """
        results = []
        current_time = time.time()
        
        with self.lock:
            # Идём от новых к старым
            for episode_id in reversed(self.timeline):
                episode = self.episodes.get(episode_id)
                if not episode:
                    continue
                
                if episode_type and episode.episode_type != episode_type:
                    continue
                
                if episode.importance < min_importance:
                    continue
                
                current_weight = episode.get_current_weight(current_time)
                if current_weight < min_weight:
                    continue
                
                if max_age and episode.get_age(current_time) > max_age:
                    continue
                
                if query and query.lower() not in episode.description.lower():
                    continue
                
                if tags and not any(tag in episode.tags for tag in tags):
                    continue
                
                results.append(episode)
                if len(results) >= limit:
                    break
        
        return results
    
    def recall_recent_reflexes(self, limit: int = 5, max_age: float = 30.0) -> List[Episode]:
        """Быстрый метод для получения последних рефлексов TinyML"""
        return self.recall(
            episode_type="reflex",
            limit=limit,
            max_age=max_age,
            min_weight=0.3
        )
    
    def get_context_episodes(self, 
                            current_context: Dict[str, Any],
                            limit: int = 5) -> List[Dict[str, Any]]:
        """
        ПОЛУЧАЕТ ЭПИЗОДЫ, РЕЛЕВАНТНЫЕ ТЕКУЩЕМУ КОНТЕКСТУ
        
        Используется для вставки в промпт LLM, чтобы она "помнила" недавние события.
        """
        current_intent = current_context.get("current_intent")
        current_time = time.time()
        
        relevant = []
        
        # 1. Ищем по intent (самое релевантное)
        if current_intent:
            relevant = self.recall(
                episode_type=current_intent,
                limit=limit,
                min_weight=0.2,
                max_age=3600  # последний час
            )
        
        # 2. Если нет — берём важные из последних 30 минут
        if not relevant:
            relevant = self.recall(
                limit=limit,
                min_weight=0.3,
                min_importance=0.5,
                max_age=1800
            )
        
        # 3. Если всё ещё пусто — последние любые
        if not relevant:
            with self.lock:
                for episode_id in list(self.timeline)[-limit:]:
                    episode = self.episodes.get(episode_id)
                    if episode:
                        relevant.append(episode)
        
        result = []
        for ep in relevant:
            result.append({
                "type": ep.episode_type,
                "description": ep.description,
                "timestamp": ep.timestamp,
                "importance": ep.importance,
                "weight": ep.get_current_weight(current_time),
                "context": ep.context
            })
        
        return result
    
    # ================================================================
    # СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ ДЛЯ РАЗНЫХ ТИПОВ СОБЫТИЙ
    # ================================================================
    
    def add_reflex(self, 
                   reflex_type: str,
                   distance_cm: float,
                   action_taken: str,
                   context: Dict[str, Any] = None) -> str:
        """
        ДОБАВЛЯЕТ РЕФЛЕКС ОТ TINYML
        
        Вызывается, когда TinyML на ESP32 среагировал на препятствие.
        Сохраняется с высокой важностью (0.9), чтобы робот "помнил" опасные места.
        """
        if context is None:
            context = {}
        
        description = f"РЕФЛЕКС: {reflex_type} на {distance_cm}см -> {action_taken}"
        context.update({
            "reflex_type": reflex_type,
            "distance_cm": distance_cm,
            "action_taken": action_taken,
            "source": "tinyml"
        })
        
        return self.add_episode(
            description=description,
            episode_type="reflex",
            importance=0.9,
            context=context,
            tags=["reflex", reflex_type, f"distance_{int(distance_cm)}"]
        )
    
    def add_command(self,
                    command: str,
                    params: Dict,
                    success: bool,
                    context: Dict[str, Any] = None) -> str:
        """Добавляет эпизод выполнения команды"""
        if context is None:
            context = {}
        
        description = f"{command}: {'успешно' if success else 'неудачно'}"
        context.update({
            "command": command,
            "params": params,
            "success": success
        })
        
        return self.add_episode(
            description=description,
            episode_type="command" if success else "error",
            importance=0.7 if success else 0.85,  # ошибки важнее
            context=context,
            tags=["command", command, "success" if success else "error"]
        )
    
    def add_conversation(self,
                         user_input: str,
                         agent_response: str,
                         context: Dict[str, Any] = None) -> str:
        """Добавляет эпизод разговора"""
        if context is None:
            context = {}
        
        description = f"Человек: {user_input[:50]}"
        context.update({
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        return self.add_episode(
            description=description,
            episode_type="conversation",
            importance=0.5,
            context=context,
            tags=["conversation"]
        )
    
    def add_observation(self, 
                        observation: str, 
                        importance: float = 0.3,
                        context: Dict[str, Any] = None) -> str:
        """Добавляет наблюдение (что видел, куда ездил)"""
        return self.add_episode(
            description=observation,
            episode_type="observation",
            importance=importance,
            context=context or {},
            tags=["observation"]
        )
    
    def add_plan(self, plan: Dict, success: bool = True) -> str:
        """Сохраняет выполненный план действий"""
        return self.add_episode(
            description=f"ПЛАН: {plan.get('task_name', 'безымянный')}",
            episode_type="plan",
            importance=0.8 if success else 0.3,
            context={
                "plan": plan,
                "success": success,
                "steps_completed": plan.get("steps", [])
            },
            tags=["plan", plan.get("task_name", "unknown")]
        )
    
    def get_successful_plans(self, limit: int = 5) -> List[Dict]:
        """Возвращает успешные планы (для автономного поведения)"""
        episodes = self.recall(
            episode_type="plan",
            min_weight=0.3,
            limit=limit
        )
        return [ep.context.get("plan", {}) for ep in episodes if ep.context.get("success")]
    
    # ================================================================
    # ИНСТРУКЦИИ ЧЕЛОВЕКА — ГЛАВНАЯ ФИШКА
    # ================================================================
    
    def add_human_instruction(self, 
                              question: str, 
                              answer: str, 
                              context: Dict[str, Any],
                              importance: float = 0.8) -> str:
        """
        СОХРАНЯЕТ ИНСТРУКЦИЮ ОТ ЧЕЛОВЕКА
        
        Это ключевой метод для "обучения у человека".
        Робот спросил — человек ответил — инструкция сохранена ВМЕСТЕ С КОНТЕКСТОМ.
        
        Args:
            question: что спросил робот
            answer: что ответил человек
            context: контекст (сенсоры, намерение, локация, объекты вокруг)
            importance: важность (по умолчанию 0.8)
        
        Returns:
            episode_id: ID сохранённой инструкции
        """
        description = f"ИНСТРУКЦИЯ: {question[:40]} → {answer[:40]}"
        
        enriched_context = {
            **context,
            "question": question,
            "answer": answer,
            "learned_at": time.time(),
            "type": "human_instruction"
        }
        
        episode_id = self.add_episode(
            description=description,
            episode_type="human_instruction",
            importance=importance,
            context=enriched_context,
            tags=["human_instruction", "learned", self._extract_keywords(answer)]
        )
        
        self.stats["human_instructions"] += 1
        logger.info(f"📝 Сохранена инструкция человека: {question[:50]} → {answer[:50]}")
        
        return episode_id
    
    def get_instruction_for_context(self, 
                                    context: Dict[str, Any], 
                                    similarity_threshold: float = 0.6) -> Optional[Dict[str, Any]]:
        """
        НАХОДИТ ИНСТРУКЦИЮ ЧЕЛОВЕКА ДЛЯ ПОХОЖЕЙ СИТУАЦИИ
        
        Это метод, который позволяет роботу "вспомнить", что ему говорили
        в похожей ситуации, и НЕ переспрашивать.
        
        Args:
            context: текущий контекст (intent, objects, location, ...)
            similarity_threshold: порог похожести (0.6 = 60%)
        
        Returns:
            Словарь с полями question, answer, learned_at, similarity
            или None, если ничего не найдено
        """
        instructions = self.recall(
            episode_type="human_instruction",
            min_weight=0.3,
            limit=10
        )
        
        best_match = None
        best_score = 0
        
        for ep in instructions:
            stored_context = ep.context.get("original_context", {}) or ep.context
            score = self._context_similarity(stored_context, context)
            
            if score > similarity_threshold and score > best_score:
                best_score = score
                best_match = {
                    "question": ep.context.get("question"),
                    "answer": ep.context.get("answer"),
                    "learned_at": ep.context.get("learned_at"),
                    "context": stored_context,
                    "similarity": score,
                    "episode_id": ep.episode_id
                }
        
        if best_match:
            logger.debug(f"🔍 Найдена инструкция для контекста (score={best_score:.2f})")
        
        return best_match
    
    def get_all_instructions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Возвращает все сохранённые инструкции человека (для отладки)"""
        episodes = self.recall(
            episode_type="human_instruction",
            limit=limit
        )
        
        return [{
            "question": ep.context.get("question"),
            "answer": ep.context.get("answer"),
            "learned_at": ep.context.get("learned_at"),
            "importance": ep.importance,
            "weight": ep.get_current_weight()
        } for ep in episodes]
    
    def _context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """
        ОЦЕНИВАЕТ ПОХОЖЕСТЬ ДВУХ КОНТЕКСТОВ
        
        Упрощённая версия. В будущем можно добавить:
        - сравнение локации
        - сравнение времени суток
        - эмбеддинги для семантической похожести
        """
        score = 0.0
        total = 0
        
        # 1. Сравниваем намерение (intent)
        intent1 = ctx1.get("current_intent", "")
        intent2 = ctx2.get("current_intent", "")
        if intent1 and intent2:
            total += 1
            if intent1 == intent2:
                score += 1.0
            elif intent1 in intent2 or intent2 in intent1:
                score += 0.5
        
        # 2. Сравниваем объекты вокруг
        objects1 = ctx1.get("objects", [])
        objects2 = ctx2.get("objects", [])
        if objects1 and objects2:
            total += 1
            names1 = set(o.get("name", "") for o in objects1)
            names2 = set(o.get("name", "") for o in objects2)
            common = names1 & names2
            score += len(common) / max(len(names1), len(names2))
        
        # 3. Сравниваем наличие препятствий
        obs1 = ctx1.get("obstacle_front", False)
        obs2 = ctx2.get("obstacle_front", False)
        if obs1 or obs2:
            total += 1
            if obs1 == obs2:
                score += 1.0
        
        return score / total if total > 0 else 0.5
    
    def _extract_keywords(self, text: str) -> str:
        """Извлекает ключевое слово для тега"""
        words = text.lower().split()
        stop_words = {"это", "то", "так", "вот", "да", "нет", "а", "и", "но", "на", "в", "с", "по"}
        for w in words[:3]:
            if w not in stop_words and len(w) > 2:
                return w
        return "instruction"
    
    # ================================================================
    # УПРАВЛЕНИЕ ПАМЯТЬЮ
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику памяти (для отладки)"""
        with self.lock:
            current_time = time.time()
            active = [e for e in self.episodes.values() 
                     if e.get_current_weight(current_time) > MIN_WEIGHT and not e.is_expired(current_time)]
            
            return {
                "total_episodes": len(self.episodes),
                "active_episodes": len(active),
                "by_type": dict(self.stats["by_type"]),
                "oldest_age": time.time() - self.stats["oldest"] if self.stats["oldest"] else 0,
                "newest_age": time.time() - self.stats["newest"] if self.stats["newest"] else 0,
                "max_episodes": self.max_episodes,
                "usage_percent": len(self.episodes) / self.max_episodes * 100,
                "human_instructions": self.stats["human_instructions"]
            }
    
    def save_to_file(self, filepath: str):
        """Сохраняет память в JSON-файл"""
        data = {
            "episodes": {},
            "stats": self.stats,
            "timestamp": time.time(),
            "version": "4.0"
        }
        
        with self.lock:
            for eid, episode in self.episodes.items():
                data["episodes"][eid] = episode.to_dict()
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Эпизодическая память сохранена в {filepath} ({len(self.episodes)} эпизодов)")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения: {e}")
    
    def load_from_file(self, filepath: str):
        """Загружает память из JSON-файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self.lock:
                self.episodes.clear()
                self.timeline.clear()
                self.type_index.clear()
                self.tag_index.clear()
                
                for eid, edata in data.get("episodes", {}).items():
                    episode = Episode.from_dict(edata)
                    self.episodes[eid] = episode
                    self.timeline.append(eid)
                    
                    if episode.episode_type not in self.type_index:
                        self.type_index[episode.episode_type] = []
                    self.type_index[episode.episode_type].append(eid)
                    
                    for tag in episode.tags:
                        if tag not in self.tag_index:
                            self.tag_index[tag] = []
                        self.tag_index[tag].append(eid)
                
                self.stats = data.get("stats", self.stats)
                self.stats["total_episodes"] = len(self.episodes)
            
            logger.info(f"📂 Эпизодическая память загружена из {filepath} ({len(self.episodes)} эпизодов)")
        except FileNotFoundError:
            logger.info("🆕 Создана новая эпизодическая память")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки: {e}")
    
    def _cleanup_if_needed(self):
        """Удаляет старые и неважные эпизоды при переполнении"""
        if len(self.episodes) <= self.max_episodes:
            return
        
        current_time = time.time()
        
        # Сортируем по текущему весу (самые слабые — первые)
        weighted = [(eid, ep.get_current_weight(current_time)) 
                   for eid, ep in self.episodes.items()]
        weighted.sort(key=lambda x: x[1])
        
        to_remove = len(self.episodes) - self.max_episodes
        for i in range(to_remove):
            eid, _ = weighted[i]
            episode = self.episodes.pop(eid, None)
            if episode:
                if episode.episode_type in self.type_index:
                    if eid in self.type_index[episode.episode_type]:
                        self.type_index[episode.episode_type].remove(eid)
                
                for tag in episode.tags:
                    if tag in self.tag_index and eid in self.tag_index[tag]:
                        self.tag_index[tag].remove(eid)
                
                self.stats["by_type"][episode.episode_type] = max(0, 
                    self.stats["by_type"].get(episode.episode_type, 0) - 1)
        
        self.stats["last_cleanup"] = current_time
        logger.debug(f"🧹 Очищено {to_remove} старых эпизодов")
    
    def clear(self):
        """Полностью очищает память"""
        with self.lock:
            self.episodes.clear()
            self.timeline.clear()
            self.type_index.clear()
            self.tag_index.clear()
            self.stats = {
                "total_episodes": 0,
                "by_type": {},
                "oldest": 0,
                "newest": 0,
                "last_cleanup": time.time(),
                "human_instructions": 0
            }
        logger.info("🧹 Эпизодическая память очищена")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ЭПИЗОДИЧЕСКОЙ ПАМЯТИ")
    print("="*60 + "\n")
    
    memory = EpisodicMemory()
    
    # 1. Сохраняем инструкцию человека
    print("📝 Сохраняем инструкцию человека...")
    memory.add_human_instruction(
        question="Я вижу оранжевый ботинок. Это тот, что вы искали?",
        answer="Нет, мне нужен чёрный ботинок. Он обычно в прихожей.",
        context={
            "current_intent": "find_boot",
            "objects": [{"name": "ботинок", "color": "оранжевый"}],
            "location": "коридор"
        }
    )
    
    # 2. Сохраняем рефлекс
    print("📝 Сохраняем рефлекс TinyML...")
    memory.add_reflex(
        reflex_type="obstacle_front",
        distance_cm=25,
        action_taken="emergency_stop",
        context={"location": "кухня", "sensors": {"lidar": {"front": 0.25}}}
    )
    
    # 3. Сохраняем успешную команду
    print("📝 Сохраняем выполнение команды...")
    memory.add_command(
        command="move_forward",
        params={"speed": 300},
        success=True,
        context={"intent": "explore"}
    )
    
    # 4. Позже, в похожей ситуации — ищем инструкцию
    print("\n🔍 Ищем инструкцию для похожего контекста...")
    instruction = memory.get_instruction_for_context(
        context={
            "current_intent": "find_boot",
            "objects": [{"name": "ботинок"}],
            "location": "прихожая"
        }
    )
    
    if instruction:
        print(f"   ✅ НАЙДЕНА ИНСТРУКЦИЯ (similarity={instruction['similarity']:.2f}):")
        print(f"      Вопрос: {instruction['question']}")
        print(f"      Ответ: {instruction['answer']}")
    else:
        print("   ❌ Инструкция не найдена")
    
    # 5. Статистика
    print("\n📊 СТАТИСТИКА ПАМЯТИ:")
    stats = memory.get_stats()
    print(f"   Всего эпизодов: {stats['total_episodes']}")
    print(f"   Активных: {stats['active_episodes']}")
    print(f"   По типам: {stats['by_type']}")
    print(f"   Инструкций человека: {stats['human_instructions']}")
    
    print("\n" + "="*60)
    print("Демонстрация завершена.")
    print("="*60 + "\n")
