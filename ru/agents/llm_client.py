#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ agents/llm_client.py — КЛИЕНТ ДЛЯ ЛОКАЛЬНОЙ LLM (OLLAMA)                     ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Асинхронный клиент для работы с Ollama. Отправляет промпт и получает       ║
║   ответ в формате JSON. Системный промпт уже зашит в модель и НЕ передаётся. ║
║                                                                              ║
║ УПРАВЛЕНИЕ КОНТЕКСТОМ (СВОРАЧИВАНИЕ ДИАЛОГА):                                ║
║                                                                              ║
║   LLM опрашивается постоянно (до 1 раза в секунду). Чтобы контекст           ║
║   не переполнялся и не вызывал галлюцинации:                                 ║
║                                                                              ║
║     1. Лимит — 100 сообщений (~50 циклов, ~4000 токенов для Vikhr 1.5B)     ║
║     2. При превышении — мягкое сворачивание:                                 ║
║        - LLM создаёт сводку первых 80 сообщений                              ║
║        - История = сводка + последние 20 сообщений                           ║
║     3. Агент может принудительно свернуть контекст (например, после задачи)  ║
║                                                                              ║
║   Это сохраняет важный контекст и предотвращает деградацию ответов.          ║
║                                                                              ║
║ РЕКОМЕНДУЕМАЯ МОДЕЛЬ: VIKHR 2.0 1.5B (РУССКОЯЗЫЧНАЯ, СВЕЖАЯ)                 ║
║                                                                              ║
║   Vikhr-2.0-1.5B-instruct (март 2026) — идеальна для смартфона:              ║
║     • 1.5B параметров → работает на 3-4 ГБ RAM                              ║
║     • Русский язык — родной, отлично понимает команды                        ║
║     • Instruct-версия — выдаёт строгий JSON                                  ║
║     • Контекстное окно 8192 токена                                          ║
║                                                                              ║
║   КАК УСТАНОВИТЬ VIKHR 2.0 В OLLAMA:                                         ║
║                                                                              ║
║     1. Скачайте GGUF (Q4_K_M — оптимальный баланс):                          ║
║        wget https://huggingface.co/Vikhr/Vikhr-2.0-1.5B-instruct-GGUF/resolve/main/vikhr-2.0-1.5b-instruct-Q4_K_M.gguf
║                                                                              ║
║     2. Создайте Modelfile:                                                   ║
║        echo 'FROM /path/to/vikhr-2.0-1.5b-instruct-Q4_K_M.gguf' > Modelfile  ║
║                                                                              ║
║     3. Импортируйте в Ollama:                                                ║
║        ollama create vikhr -f Modelfile                                      ║
║                                                                              ║
║     4. В config.py укажите: LLM_MODEL = "vikhr"                              ║
║                                                                              ║
║ АЛЬТЕРНАТИВЫ:                                                                ║
║     • qwen2.5:1.5b — хорош, но хуже с русским                                ║
║     • llama3.2:3b — мощнее, но тяжелее                                       ║
║     • saiga_llama3_8b — отличный русский, но для мощных ПК                   ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from agents.llm_client import LocalLLM                                     ║
║                                                                              ║
║   llm = LocalLLM(model="vikhr", base_url="http://localhost:11434")           ║
║   response = await llm.generate([{"role": "user", "content": "поехали"}])    ║
║                                                                              ║
║   if response.action:                                                        ║
║       print(response.action)  # {"action": "move_forward", "params": {...}}  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    АСИНХРОННЫЙ КЛИЕНТ ДЛЯ OLLAMA С УПРАВЛЕНИЕМ КОНТЕКСТОМ
    
    Отправляет запросы к локальной LLM и парсит JSON-ответы.
    Системный промпт уже зашит в модель — не передаётся!
    Автоматически сворачивает длинные диалоги.
    """
    
    def __init__(self, 
                 model: str, 
                 base_url: str = "http://localhost:11434",
                 max_history_messages: int = 100,
                 auto_summarize: bool = True):
        """
        Args:
            model: имя модели в Ollama
            base_url: URL Ollama сервера
            max_history_messages: максимум сообщений до сворачивания (100 = ~50 циклов)
            auto_summarize: автоматически сворачивать при превышении
        """
        self.model = model
        self.base_url = base_url
        self.max_history_messages = max_history_messages
        self.auto_summarize = auto_summarize
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.history: List[Dict] = []  # история диалога
        
        logger.info(f"✅ LocalLLM инициализирован (model={model}, max_history={max_history_messages})")
    
    async def ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate(self, messages: List[Dict]) -> Any:
        """
        ОТПРАВЛЯЕТ ДИАЛОГ В LLM
        
        Автоматически добавляет сообщения в историю и сворачивает при превышении.
        
        Args:
            messages: список новых сообщений [{"role": "user", "content": "..."}]
        
        Returns:
            Объект Response с полями:
                - content: str — сырой ответ
                - action: dict — {"action", "params", "reasoning"} (если есть)
                - text: str — текстовый ответ (если нет action)
        """
        await self.ensure_session()
        
        # Добавляем новые сообщения в историю
        self.history.extend(messages)
        
        # Проверяем, не пора ли свернуть контекст
        if self.auto_summarize and len(self.history) > self.max_history_messages:
            logger.info(f"📚 История достигла {len(self.history)} сообщений, сворачиваю...")
            await self._summarize_and_compress()
        
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": False,
            "temperature": 0.7,
            "format": "json"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = self._parse_response(data)
                    
                    # Добавляем ответ в историю
                    self.history.append({"role": "assistant", "content": response.content})
                    
                    return response
                else:
                    error_text = await resp.text()
                    logger.error(f"Ollama error {resp.status}: {error_text[:200]}")
                    return self._error_response(f"Ошибка LLM: {resp.status}")
        except Exception as e:
            logger.error(f"Ollama exception: {e}")
            return self._error_response(f"Ошибка соединения: {e}")
    
    async def _summarize_and_compress(self):
        """
        МЯГКОЕ СВОРАЧИВАНИЕ ИСТОРИИ ДИАЛОГА
        
        Просит LLM кратко описать, что произошло в первой части диалога,
        и заменяет её на сводку, сохраняя последние 20 сообщений.
        """
        # Сохраняем последние 20 сообщений (чтобы не потерять ближайший контекст)
        keep_last = 20
        if len(self.history) <= keep_last:
            return
        
        recent = self.history[-keep_last:]
        old_history = self.history[:-keep_last]
        
        # Формируем промпт для суммаризации
        summary_prompt = old_history + [
            {
                "role": "user", 
                "content": "Кратко опиши (3-5 предложений), что произошло за эту часть диалога: какие команды выполнялись, что робот видел, куда ездил, с кем говорил. Не пиши JSON, просто текст."
            }
        ]
        
        payload = {
            "model": self.model,
            "messages": summary_prompt,
            "stream": False,
            "temperature": 0.5
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    summary = data.get('message', {}).get('content', '').strip()
                    
                    if summary:
                        # Заменяем историю на сводку + последние сообщения
                        self.history = [
                            {"role": "user", "content": f"[Сводка предыдущего диалога]: {summary}"}
                        ] + recent
                        logger.info(f"📝 Контекст свёрнут. Новая длина: {len(self.history)}")
                    else:
                        # Если сводка пустая — оставляем последние 30 сообщений
                        self.history = self.history[-30:]
                        logger.warning("⚠️ Не удалось получить сводку, оставляю последние 30 сообщений")
                else:
                    self.history = self.history[-30:]
                    logger.warning(f"⚠️ Ошибка суммаризации, оставляю последние 30 сообщений")
        except Exception as e:
            self.history = self.history[-30:]
            logger.error(f"❌ Ошибка сворачивания контекста: {e}")
    
    async def summarize_and_reset(self, keep_last: int = 10) -> str:
        """
        ПРИНУДИТЕЛЬНО СВОРАЧИВАЕТ КОНТЕКСТ И ВОЗВРАЩАЕТ СВОДКУ
        
        Используется агентом для явного управления контекстом
        (например, после завершения сложной задачи).
        
        Args:
            keep_last: сколько последних сообщений сохранить
        
        Returns:
            str: текст сводки
        """
        if len(self.history) < 10:
            return "История слишком короткая для сводки"
        
        recent = self.history[-keep_last:] if keep_last > 0 else []
        old_history = self.history[:-keep_last] if keep_last > 0 else self.history
        
        summary_prompt = old_history + [
            {"role": "user", "content": "Кратко опиши (3-5 предложений), что произошло за этот диалог: какие задачи выполнялись, что было сделано, что робот узнал."}
        ]
        
        payload = {
            "model": self.model,
            "messages": summary_prompt,
            "stream": False,
            "temperature": 0.5
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    summary = data.get('message', {}).get('content', '').strip()
                    
                    if summary:
                        self.history = [
                            {"role": "user", "content": f"[Сводка предыдущей сессии]: {summary}"}
                        ] + recent
                        logger.info(f"📝 Ручное сворачивание выполнено. Длина: {len(self.history)}")
                        return summary
        except Exception as e:
            logger.error(f"❌ Ошибка ручного сворачивания: {e}")
        
        return ""
    
    def _parse_response(self, data: Dict) -> Any:
        """Парсит ответ от Ollama"""
        message = data.get('message', {})
        content = message.get('content', '').strip()
        
        if not content:
            return self._error_response("Пустой ответ от LLM")
        
        # Пробуем распарсить JSON
        try:
            resp = json.loads(content)
            
            class Response:
                def __init__(self, content, action=None, text=None):
                    self.content = content
                    self.action = action
                    self.text = text
                    self.tool_calls = None
            
            if "action" in resp:
                return Response(
                    content=content,
                    action={
                        "action": resp["action"],
                        "params": resp.get("params", {}),
                        "reasoning": resp.get("reasoning", "")
                    }
                )
            
            if "text" in resp:
                return Response(content=content, text=resp["text"])
            
            logger.warning(f"Неизвестный формат JSON: {content[:200]}")
            return Response(content=content, text=str(resp)[:200])
                
        except json.JSONDecodeError:
            logger.debug(f"Текстовый ответ: {content[:100]}")
            
            class Response:
                def __init__(self, content, text):
                    self.content = content
                    self.action = None
                    self.text = text
                    self.tool_calls = None
            return Response(content=content, text=content[:200])
    
    def _error_response(self, error_msg: str) -> Any:
        class Response:
            def __init__(self, content, text):
                self.content = content
                self.action = None
                self.text = text
                self.tool_calls = None
        return Response(content=error_msg, text=error_msg)
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.history = []
        logger.info("🧹 История диалога очищена")
    
    def get_history(self) -> List[Dict]:
        """Возвращает текущую историю"""
        return self.history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику"""
        total_chars = sum(len(msg.get("content", "")) for msg in self.history)
        estimated_tokens = total_chars // 4
        
        return {
            "history_length": len(self.history),
            "max_history": self.max_history_messages,
            "estimated_tokens": estimated_tokens,
            "needs_summarize": len(self.history) > self.max_history_messages,
            "usage_percent": (len(self.history) / self.max_history_messages) * 100 if self.max_history_messages else 0
        }
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ LLM CLIENT С УПРАВЛЕНИЕМ КОНТЕКСТОМ")
        print("="*60 + "\n")
        
        llm = LocalLLM(model="vikhr", max_history_messages=100)
        
        print(f"📊 Настройки: max_history={llm.max_history_messages} сообщений")
        print(f"   (~{llm.max_history_messages // 2} циклов, ~{llm.max_history_messages * 40} токенов)")
        print("\n" + "-"*40)
        
        # Имитируем несколько команд
        messages = [
            "поехали вперёд",
            "поверни налево",
            "что ты видишь?",
            "продолжай движение"
        ]
        
        for i, msg in enumerate(messages):
            print(f"\n📤 Сообщение {i+1}: {msg}")
            response = await llm.generate([{"role": "user", "content": msg}])
            
            if response.action:
                print(f"   ✅ Действие: {response.action['action']}")
            elif response.text:
                print(f"   💬 Ответ: {response.text[:50]}...")
            
            stats = llm.get_stats()
            print(f"   📊 История: {stats['history_length']} сообщений, ~{stats['estimated_tokens']} токенов")
        
        # Принудительная сводка
        print("\n" + "="*40)
        print("📝 Принудительная сводка (ручное сворачивание):")
        summary = await llm.summarize_and_reset(keep_last=4)
        print(f"   Сводка: {summary}")
        print(f"   📊 История после сводки: {len(llm.history)} сообщений")
        
        await llm.close()
        print("\n" + "="*60)
        print("✅ Демонстрация завершена")
        print("="*60 + "\n")
    
    asyncio.run(demo())
