#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ memory/dialog_context.py - КОНТЕКСТ ДИАЛОГА С ЧЕЛОВЕКОМ                       ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «кратковременная память» разговора. Хранит историю общения,            ║
║   отслеживает текущее намерение человека, взвешивает реплики по времени.     ║
║                                                                              ║
║ ЗАЧЕМ ЭТО НУЖНО:                                                             ║
║   LLM не помнит, что было 5 минут назад. DialogContext решает эту проблему:  ║
║   он хранит последние N реплик и передаёт их в промпт.                        ║
║                                                                              ║
║   Кроме того, он ОПРЕДЕЛЯЕТ НАМЕРЕНИЕ человека.                               ║
║   Если человек сказал «поехали», потом «стой», потом «найди ботинок» —       ║
║   DialogContext понимает, что сейчас главное намерение — «find_boot».        ║
║                                                                              ║
║ КАК РАБОТАЕТ ОПРЕДЕЛЕНИЕ НАМЕРЕНИЯ:                                          ║
║                                                                              ║
║   1. Каждая реплика человека может иметь intent (определяется агентом).      ║
║   2. Intent'ы накапливаются с весами.                                        ║
║   3. Веса ЗАТУХАЮТ со временем (decay = 0.9).                                ║
║   4. get_primary_intent() возвращает intent с самым большим весом.           ║
║                                                                              ║
║   Пример:                                                                    ║
║   - 10 секунд назад: "поехали" (intent: move, вес: 0.9^10 ≈ 0.35)           ║
║   - 2 секунды назад: "стой" (intent: stop, вес: 1.0)                        ║
║   → primary_intent = "stop"                                                  ║
║                                                                              ║
║ КАК ИСПОЛЬЗУЕТСЯ В АГЕНТЕ:                                                   ║
║                                                                              ║
║   1. Когда человек говорит — agent вызывает dialog.add_turn(text, intent)   ║
║   2. Перед вызовом LLM — agent берёт dialog.get_context_dict()               ║
║   3. ContextBuilder вставляет историю в промпт                               ║
║   4. StrategyLearner использует primary_intent для выбора стратегии          ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. max_turns — сколько реплик хранить (по умолчанию 20)                    ║
║   2. decay — скорость затухания весов (0.9 = каждую секунду вес ×0.9)        ║
║   3. Добавить свои intent'ы — просто используйте их в add_turn()             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import uuid
from typing import List, Dict, Any, Optional


class DialogContext:
    """
    КОНТЕКСТ ДИАЛОГА С ИСТОРИЕЙ И ВЗВЕШИВАНИЕМ НАМЕРЕНИЙ
    
    Хранит последние реплики, отслеживает текущее намерение человека.
    Старые реплики и намерения автоматически теряют вес.
    """
    
    def __init__(self, dialog_id: Optional[str] = None, max_turns: int = 20):
        """
        Args:
            dialog_id: уникальный ID диалога (если не указан — генерируется)
            max_turns: сколько последних реплик хранить
        """
        self.dialog_id = dialog_id or str(uuid.uuid4())
        self.turns = []
        self.max_turns = max_turns
        self.created_at = time.time()
        self.last_updated = self.created_at
        
        # Текущее намерение и веса всех намерений
        self.current_intent: Optional[str] = None
        self.intent_weights: Dict[str, float] = {}
    
    def add_turn(self, 
                 human_input: str, 
                 agent_output: str, 
                 intent: Optional[str] = None, 
                 confidence: float = 1.0,
                 source: str = "human"):
        """
        ДОБАВЛЯЕТ ОБОРОТ ДИАЛОГА
        
        Args:
            human_input: что сказал человек
            agent_output: что ответил робот
            intent: намерение (move, stop, find, ask, ...)
            confidence: уверенность в намерении (0-1)
            source: источник реплики (human / tinyml / agent)
        """
        turn = {
            "human": human_input,
            "agent": agent_output,
            "intent": intent,
            "confidence": confidence,
            "source": source,
            "timestamp": time.time(),
            "weight": 1.0
        }
        
        self.turns.append(turn)
        self.last_updated = turn["timestamp"]
        
        # Ограничиваем историю
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        
        # Обновляем веса намерений
        if intent:
            self.current_intent = intent
            self._update_intent_weight(intent, confidence)
    
    def _update_intent_weight(self, intent: str, confidence: float):
        """
        ОБНОВЛЯЕТ ВЕСА НАМЕРЕНИЙ С ЗАТУХАНИЕМ
        
        Каждый вызов уменьшает веса ВСЕХ намерений на 10%,
        затем добавляет вес новому намерению.
        """
        decay = 0.9
        
        # Затухание старых намерений
        for old_intent in list(self.intent_weights.keys()):
            self.intent_weights[old_intent] *= decay
            if self.intent_weights[old_intent] < 0.1:
                del self.intent_weights[old_intent]
        
        # Добавляем новое намерение
        self.intent_weights[intent] = self.intent_weights.get(intent, 0) + confidence
    
    def get_primary_intent(self) -> Optional[str]:
        """
        ВОЗВРАЩАЕТ ОСНОВНОЕ НАМЕРЕНИЕ (С САМЫМ БОЛЬШИМ ВЕСОМ)
        
        Используется агентом для выбора стратегии.
        """
        if not self.intent_weights:
            return None
        
        return max(self.intent_weights.items(), key=lambda x: x[1])[0]
    
    def get_recent_turns(self, n: int = 5) -> List[Dict]:
        """
        ВОЗВРАЩАЕТ ПОСЛЕДНИЕ N РЕПЛИК С ОБНОВЛЁННЫМИ ВЕСАМИ
        
        Веса обновляются по времени: старые реплики весят меньше.
        """
        now = time.time()
        
        for turn in self.turns:
            age = now - turn["timestamp"]
            turn["weight"] = 0.9 ** age
        
        return self.turns[-n:] if self.turns else []
    
    def get_context_dict(self) -> Dict[str, Any]:
        """
        ВОЗВРАЩАЕТ ПОЛНЫЙ КОНТЕКСТ ДЛЯ CONTEXTBUILDER
        
        Используется агентом перед вызовом LLM.
        """
        return {
            "dialog_id": self.dialog_id,
            "turns": self.get_recent_turns(10),
            "current_intent": self.get_primary_intent(),
            "intent_weights": self.intent_weights.copy(),
            "last_updated": self.last_updated
        }
    
    def get_context_text(self) -> str:
        """
        ВОЗВРАЩАЕТ ТЕКСТОВОЕ ПРЕДСТАВЛЕНИЕ ДИАЛОГА
        
        Используется для отладки.
        """
        recent = self.get_recent_turns(5)
        lines = ["История разговора:"]
        
        for turn in recent:
            if turn.get("human"):
                lines.append(f"Человек: {turn['human']}")
            if turn.get("agent"):
                lines.append(f"Робот: {turn['agent']}")
        
        if self.current_intent:
            lines.append(f"Текущее намерение: {self.current_intent}")
        
        return "\n".join(lines)
    
    def clear(self):
        """ОЧИЩАЕТ ИСТОРИЮ ДИАЛОГА"""
        self.turns.clear()
        self.intent_weights.clear()
        self.current_intent = None
        self.last_updated = time.time()


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ DIALOG CONTEXT")
    print("="*60 + "\n")
    
    dialog = DialogContext(max_turns=10)
    
    # Имитируем диалог
    print("💬 Имитация диалога:")
    print("-" * 40)
    
    # Реплика 1
    dialog.add_turn("Гралл, поехали", "Еду вперёд", intent="move")
    print("Человек: Гралл, поехали")
    print("Робот: Еду вперёд")
    print(f"   primary_intent: {dialog.get_primary_intent()}")
    
    time.sleep(0.5)
    
    # Реплика 2
    dialog.add_turn("Стой!", "Останавливаюсь", intent="stop")
    print("\nЧеловек: Стой!")
    print("Робот: Останавливаюсь")
    print(f"   primary_intent: {dialog.get_primary_intent()}")
    
    time.sleep(0.5)
    
    # Реплика 3
    dialog.add_turn("Найди мой ботинок", "Ищу ботинок", intent="find")
    print("\nЧеловек: Найди мой ботинок")
    print("Робот: Ищу ботинок")
    print(f"   primary_intent: {dialog.get_primary_intent()}")
    
    # Показываем веса намерений
    print(f"\n📊 Веса намерений: {dialog.intent_weights}")
    
    # Ждём, чтобы показать затухание
    print("\n⏳ Ждём 2 секунды (веса затухают)...")
    time.sleep(2)
    
    print(f"   Веса после затухания: {dialog.intent_weights}")
    print(f"   primary_intent: {dialog.get_primary_intent()}")
    
    # Показываем историю
    print("\n📜 История диалога (с весами):")
    for turn in dialog.get_recent_turns(5):
        print(f"   [{turn['weight']:.2f}] {turn['human']} → {turn['agent']}")
    
    print("\n" + "="*60)
    print("✅ Демонстрация завершена.")
    print("="*60 + "\n")
