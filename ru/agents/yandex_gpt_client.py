#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ agents/yandex_gpt_client.py — ОБЛАЧНЫЙ ИНТЕЛЛЕКТ ДЛЯ СЛАБОГО ЖЕЛЕЗА           ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Клиент для YandexGPT API. Позволяет роботу «думать» на мощностях облака,   ║
║   даже если локальное железо — слабый смартфон или Raspberry Pi.             ║
║                                                                              ║
║ ПОЧЕМУ YANDEXGPT:                                                            ║
║   • Поддерживает instruction (системный промпт без расхода токенов)          ║
║   • Дёшево (модель Light)                                                    ║
║   • Отличный русский язык                                                    ║
║   • Работает через простой HTTP API                                          ║
║   • Имеет доступ к поиску в интернете (через правильные промпты)             ║
║                                                                              ║
║ РЕЖИМ «КОНЦЕНТРАЦИИ»:                                                        ║
║                                                                              ║
║   Локальная Vikhr (1.5B) быстра, но иногда не справляется со сложными       ║
║   задачами. В такие моменты робот может переключиться в режим концентрации:  ║
║                                                                              ║
║     1. Отправить сложный запрос в YandexGPT.                                 ║
║     2. Получить качественный ответ (план, стратегию, анализ).                ║
║     3. Сохранить результат в память как образец.                             ║
║     4. Продолжить работу на локальной модели.                                ║
║                                                                              ║
║   Это даёт «интеллект по требованию» без необходимости держать мощное        ║
║   железо постоянно включённым.                                               ║
║                                                                              ║
║ ПОИСК В ИНТЕРНЕТЕ:                                                           ║
║                                                                              ║
║   YandexGPT имеет доступ к поиску Яндекса. Метод search_web() позволяет      ║
║   задать вопрос, требующий актуальной информации из сети:                    ║
║                                                                              ║
║     • «Какая погода в Москве?»                                               ║
║     • «Курс доллара сегодня»                                                 ║
║     • «Когда Пасха в 2026 году?»                                             ║
║     • «Что такое квантовый компьютер?»                                       ║
║                                                                              ║
║   Робот сам определяет, что вопрос требует поиска, и вызывает этот метод.    ║
║                                                                              ║
║ БУДУЩЕЕ — DEEPSEEK:                                                          ║
║   API DeepSeek совместим с OpenAI. Когда появится возможность,               ║
║   мы добавим DeepSeekClient с таким же интерфейсом.                          ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from agents.yandex_gpt_client import YandexGPTClient                        ║
║                                                                              ║
║   client = YandexGPTClient(                                                  ║
║       folder_id="your-folder-id",                                            ║
║       api_key="your-api-key",                                                ║
║       instruction=SYSTEM_PROMPT                                              ║
║   )                                                                          ║
║                                                                              ║
║   # Обычный запрос                                                           ║
║   response = await client.generate(messages)                                 ║
║                                                                              ║
║   # Поиск в интернете                                                        ║
║   answer = await client.search_web("Какая погода в Москве?")                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import aiohttp
import json
import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class YandexGPTClient:
    """
    КЛИЕНТ ДЛЯ YANDEXGPT API
    
    Поддерживает persistent instruction (системный промпт).
    Формат ответа полностью совместим с LocalLLM.
    Имеет специальный режим для поиска в интернете.
    """
    
    def __init__(self, 
                 folder_id: str,
                 api_key: str = None,
                 iam_token: str = None,
                 model: str = "yandexgpt/latest",
                 instruction: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        """
        Args:
            folder_id: ID каталога в Yandex Cloud
            api_key: API ключ (рекомендуется)
            iam_token: IAM токен (альтернатива api_key)
            model: yandexgpt/latest, yandexgpt/rc, yandexgpt/lite
            instruction: системный промпт (не расходует токены контекста!)
            temperature: 0-1
            max_tokens: максимальная длина ответа
        """
        self.folder_id = folder_id
        self.api_key = api_key
        self.iam_token = iam_token
        self.model = model
        self.instruction = instruction
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"✅ YandexGPTClient инициализирован (model={model})")
    
    async def ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    def _get_auth_header(self) -> Dict[str, str]:
        """Получить заголовок авторизации"""
        if self.api_key:
            return {"Authorization": f"Api-Key {self.api_key}"}
        elif self.iam_token:
            return {"Authorization": f"Bearer {self.iam_token}"}
        else:
            raise ValueError("Необходимо указать api_key или iam_token")
    
    async def generate(self, messages: List[Dict], use_search: bool = False) -> Any:
        """
        ГЕНЕРАЦИЯ ОТВЕТА ОТ YANDEXGPT
        
        Args:
            messages: список сообщений [{"role": "user", "content": "..."}]
            use_search: использовать ли поиск по интернету
        
        Returns:
            Объект Response с полями:
                - content: сырой текст
                - action: {"action", "params", "reasoning"} (если есть)
                - text: текстовый ответ
        """
        await self.ensure_session()
        
        # Форматируем сообщения для YandexGPT
        formatted_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                formatted_messages.append({"role": role, "text": content})
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": formatted_messages
        }
        
        # Включаем поиск, если нужно
        if use_search:
            payload["search"] = {"enable": True}
        
        if self.instruction:
            payload["instruction"] = self.instruction
        
        headers = self._get_auth_header()
        headers["Content-Type"] = "application/json"
        
        try:
            async with self.session.post(
                f"{self.base_url}/completion",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_response(data)
                else:
                    error_text = await resp.text()
                    logger.error(f"YandexGPT error {resp.status}: {error_text[:200]}")
                    return self._error_response(f"Ошибка API: {resp.status}")
        except Exception as e:
            logger.error(f"YandexGPT exception: {e}")
            return self._error_response(f"Ошибка соединения: {e}")
    
    async def search_web(self, query: str) -> str:
        """
        ПОИСК В ИНТЕРНЕТЕ ЧЕРЕЗ YANDEXGPT
        
        Используется для вопросов, требующих актуальной информации:
        погода, курсы валют, новости, даты, факты.
        
        Args:
            query: поисковый запрос
        
        Returns:
            str: краткий фактологический ответ
        """
        await self.ensure_session()
        
        # Специальный instruction для поисковых запросов
        search_instruction = """Ты — поисковый ассистент. 
Отвечай на вопросы кратко, только фактами. 
Если не знаешь точного ответа — скажи "не знаю".
Не пиши JSON, просто текст."""
        
        messages = [
            {"role": "user", "text": query}
        ]
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": 0.3,  # меньше креативности для фактов
                "maxTokens": 200
            },
            "messages": messages,
            "instruction": search_instruction,
            "search": {"enable": True}  # включаем поиск Яндекса
        }
        
        headers = self._get_auth_header()
        headers["Content-Type"] = "application/json"
        
        try:
            async with self.session.post(
                f"{self.base_url}/completion",
                json=payload,
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    alternatives = result.get("alternatives", [])
                    if alternatives:
                        message = alternatives[0].get("message", {})
                        return message.get("text", "").strip()
                    return "Не удалось найти информацию"
                else:
                    logger.error(f"Search error {resp.status}")
                    return f"Ошибка поиска: {resp.status}"
        except Exception as e:
            logger.error(f"Search exception: {e}")
            return f"Ошибка соединения: {e}"
    
    def _parse_response(self, data: Dict) -> Any:
        """Парсинг ответа от YandexGPT"""
        try:
            result = data.get("result", {})
            alternatives = result.get("alternatives", [])
            
            if not alternatives:
                return self._error_response("Пустой ответ")
            
            message = alternatives[0].get("message", {})
            content = message.get("text", "").strip()
            
            if not content:
                return self._error_response("Пустой ответ")
            
            class Response:
                def __init__(self, content, action=None, text=None):
                    self.content = content
                    self.action = action
                    self.text = text
                    self.tool_calls = None
            
            # Парсим JSON из ответа
            try:
                resp = json.loads(content)
                
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
                
                return Response(content=content, text=content[:200])
                
            except json.JSONDecodeError:
                return Response(content=content, text=content[:200])
                
        except Exception as e:
            logger.error(f"Ошибка парсинга: {e}")
            return self._error_response(f"Ошибка парсинга: {e}")
    
    def _error_response(self, error_msg: str) -> Any:
        class Response:
            def __init__(self, content, text):
                self.content = content
                self.action = None
                self.text = text
                self.tool_calls = None
        return Response(content=error_msg, text=error_msg)
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None


class YandexGPTClientWithHistory(YandexGPTClient):
    """
    РАСШИРЕННЫЙ КЛИЕНТ С УПРАВЛЕНИЕМ ИСТОРИЕЙ
    
    Автоматически хранит историю диалога и сворачивает при превышении.
    """
    
    def __init__(self, max_history_messages: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.history: List[Dict] = []
        self.max_history_messages = max_history_messages
    
    async def chat(self, user_message: str, context: str = None, use_search: bool = False) -> Any:
        """
        ОТПРАВИТЬ СООБЩЕНИЕ С АВТОМАТИЧЕСКИМ ДОБАВЛЕНИЕМ В ИСТОРИЮ
        """
        message = user_message
        if context:
            message = f"{context}\n\n{user_message}"
        
        self.history.append({"role": "user", "content": message})
        
        if len(self.history) > self.max_history_messages:
            keep_last = 20
            recent = self.history[-keep_last:]
            summary_prompt = self.history[:-keep_last] + [
                {"role": "user", "content": "Кратко опиши, что произошло (3-5 предложений)."}
            ]
            
            try:
                summary_response = await self.generate(summary_prompt)
                summary = summary_response.text or summary_response.content
                self.history = [
                    {"role": "user", "content": f"[Сводка]: {summary}"}
                ] + recent
                logger.info(f"📝 История свёрнута до {len(self.history)} сообщений")
            except Exception as e:
                self.history = self.history[-30:]
                logger.warning(f"⚠️ Ошибка сворачивания: {e}")
        
        response = await self.generate(self.history, use_search=use_search)
        
        if response and hasattr(response, 'content'):
            self.history.append({"role": "assistant", "content": response.content})
        
        return response
    
    async def search_web(self, query: str) -> str:
        """Поиск в интернете (без сохранения в историю диалога)"""
        return await super().search_web(query)
    
    def clear_history(self):
        self.history = []
        logger.info("🧹 История диалога очищена")
    
    def get_stats(self) -> Dict:
        total_chars = sum(len(msg.get("content", "")) for msg in self.history)
        return {
            "messages_count": len(self.history),
            "estimated_tokens": total_chars // 4,
            "has_instruction": self.instruction is not None
        }
    
    async def close(self):
        await super().close()


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ YANDEXGPT CLIENT")
        print("="*60 + "\n")
        
        print("⚠️ Для работы нужен API ключ Yandex Cloud.")
        print("   Укажите folder_id и api_key в коде демонстрации.\n")
        
        # Пример (замените на свои данные)
        FOLDER_ID = "your-folder-id"
        API_KEY = "your-api-key"
        
        if FOLDER_ID == "your-folder-id":
            print("❌ Укажите реальные FOLDER_ID и API_KEY для демонстрации.")
            return
        
        client = YandexGPTClient(
            folder_id=FOLDER_ID,
            api_key=API_KEY,
            temperature=0.7
        )
        
        try:
            # Демонстрация поиска в интернете
            print("🔍 Поиск в интернете: 'Какая погода в Москве?'")
            weather = await client.search_web("Какая погода в Москве?")
            print(f"   Ответ: {weather}")
            
            print("\n🔍 Поиск в интернете: 'Курс доллара сегодня'")
            usd_rate = await client.search_web("Курс доллара сегодня")
            print(f"   Ответ: {usd_rate}")
            
            print("\n🔍 Поиск в интернете: 'Когда Пасха в 2026 году?'")
            easter = await client.search_web("Когда Пасха в 2026 году?")
            print(f"   Ответ: {easter}")
            
        finally:
            await client.close()
        
        print("\n" + "="*60)
        print("✅ Демонстрация завершена")
        print("="*60 + "\n")
    
    asyncio.run(demo())
