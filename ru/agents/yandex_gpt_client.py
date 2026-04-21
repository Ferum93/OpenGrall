#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ agents/yandex_gpt_client.py — ОБЛАЧНЫЙ ИНТЕЛЛЕКТ ДЛЯ СЛАБОГО ЖЕЛЕЗА           ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Клиент для YandexGPT API. Позволяет роботу «думать» на мощностях облака,   ║
║   даже если локальное железо — слабый смартфон или Raspberry Pi.             ║
║                                                                              ║
║ ВОЗМОЖНОСТИ:                                                                 ║
║   • Обычная генерация (до 4000 токенов по умолчанию)                         ║
║   • Потоковая генерация (streaming) для длинного кода                        ║
║   • Поиск в интернете с гибким лимитом токенов                               ║
║   • Расширенный поиск для инженера                                           ║
║   • Информация о лимитах модели                                              ║
║                                                                              ║
║ РЕЖИМЫ ИСПОЛЬЗОВАНИЯ:                                                        ║
║   • ПИЛОТ: короткие ответы, управление роботом                               ║
║   • ИНЖЕНЕР: длинные генерации кода, адаптация драйверов                     ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from agents.yandex_gpt_client import YandexGPTClient                        ║
║                                                                              ║
║   client = YandexGPTClient(                                                  ║
║       folder_id="your-folder-id",                                            ║
║       api_key="your-api-key"                                                 ║
║   )                                                                          ║
║                                                                              ║
║   # Обычный запрос                                                           ║
║   response = await client.generate(messages)                                 ║
║                                                                              ║
║   # Потоковая генерация (для длинного кода)                                  ║
║   async for chunk in client.generate_stream(messages):                       ║
║       print(chunk, end="")                                                   ║
║                                                                              ║
║   # Поиск в интернете                                                        ║
║   answer = await client.search_web("Какая погода в Москве?")                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional, AsyncIterator

logger = logging.getLogger(__name__)


class YandexGPTClient:
    """
    КЛИЕНТ ДЛЯ YANDEXGPT API С ПОДДЕРЖКОЙ СТРИМИНГА
    
    Поддерживает persistent instruction (системный промпт).
    Формат ответа полностью совместим с LocalLLM.
    Имеет специальный режим для поиска в интернете.
    """
    
    # Лимиты моделей (input, output токены)
    MODEL_LIMITS = {
        "yandexgpt/lite": {"max_input_tokens": 8000, "max_output_tokens": 2000},
        "yandexgpt/latest": {"max_input_tokens": 8000, "max_output_tokens": 4000},
        "yandexgpt/rc": {"max_input_tokens": 8000, "max_output_tokens": 8000},
        "yandexgpt/pro": {"max_input_tokens": 8000, "max_output_tokens": 8000},
    }
    
    def __init__(self, 
                 folder_id: str,
                 api_key: str = None,
                 iam_token: str = None,
                 model: str = "yandexgpt/latest",
                 instruction: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 4000):
        """
        Args:
            folder_id: ID каталога в Yandex Cloud
            api_key: API ключ (рекомендуется)
            iam_token: IAM токен (альтернатива api_key)
            model: yandexgpt/latest, yandexgpt/rc, yandexgpt/lite, yandexgpt/pro
            instruction: системный промпт (не расходует токены контекста!)
            temperature: 0-1
            max_tokens: максимальная длина ответа (по умолчанию 4000)
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
        
        logger.info(f"✅ YandexGPTClient инициализирован (model={model}, max_tokens={max_tokens})")
    
    async def ensure_session(self):
        """Создаёт HTTP-сессию, если её нет"""
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
    
    def get_model_limits(self) -> Dict[str, int]:
        """Возвращает лимиты текущей модели (input/output токены)"""
        return self.MODEL_LIMITS.get(
            self.model, 
            {"max_input_tokens": 8000, "max_output_tokens": 2000}
        )
    
    async def generate(self, messages: List[Dict], use_search: bool = False) -> Any:
        """
        ГЕНЕРАЦИЯ ОТВЕТА ОТ YANDEXGPT (НЕПОТОКОВАЯ)
        
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
            if role in ("user", "assistant", "system"):
                formatted_messages.append({"role": role, "text": content})
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
                "stream": False
            },
            "messages": formatted_messages
        }
        
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
    
    async def generate_stream(self, messages: List[Dict], use_search: bool = False) -> AsyncIterator[str]:
        """
        ПОТОКОВАЯ ГЕНЕРАЦИЯ ОТВЕТА ОТ YANDEXGPT
        
        Используется для длинных генераций (код, драйверы, конфигурации).
        Позволяет получать ответ по частям, не дожидаясь полной генерации.
        
        Args:
            messages: список сообщений
            use_search: использовать ли поиск по интернету
        
        Yields:
            str: фрагменты ответа (чанки)
        """
        await self.ensure_session()
        
        # Форматируем сообщения
        formatted_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                formatted_messages.append({"role": role, "text": content})
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": self.temperature,
                "maxTokens": self.max_tokens,
                "stream": True  # Включаем стриминг
            },
            "messages": formatted_messages
        }
        
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
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"YandexGPT stream error {resp.status}: {error_text[:200]}")
                    yield f"Ошибка API: {resp.status}"
                    return
                
                # Читаем поток построчно (Server-Sent Events)
                async for line in resp.content:
                    line_text = line.decode('utf-8').strip()
                    if not line_text or line_text.startswith(':'):
                        continue
                    
                    if line_text.startswith('data: '):
                        data_str = line_text[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            chunk_text = self._extract_stream_chunk(data)
                            if chunk_text:
                                yield chunk_text
                        except json.JSONDecodeError:
                            logger.warning(f"Невалидный JSON в потоке: {data_str[:100]}")
                            continue
                            
        except Exception as e:
            logger.error(f"YandexGPT stream exception: {e}")
            yield f"Ошибка соединения: {e}"
    
    def _extract_stream_chunk(self, data: Dict) -> Optional[str]:
        """Извлекает текст из чанка потокового ответа"""
        try:
            result = data.get("result", {})
            alternatives = result.get("alternatives", [])
            if alternatives:
                message = alternatives[0].get("message", {})
                return message.get("text", "")
        except Exception:
            pass
        return None
    
    async def search_web(self, query: str, max_tokens: int = 500) -> str:
        """
        ПОИСК В ИНТЕРНЕТЕ ЧЕРЕЗ YANDEXGPT (КРАТКИЙ ОТВЕТ)
        
        Используется для вопросов, требующих актуальной информации:
        погода, курсы валют, новости, даты, факты.
        
        Args:
            query: поисковый запрос
            max_tokens: максимальная длина ответа (по умолчанию 500)
        
        Returns:
            str: краткий фактологический ответ
        """
        await self.ensure_session()
        
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
                "temperature": 0.3,
                "maxTokens": max_tokens,
                "stream": False
            },
            "messages": messages,
            "instruction": search_instruction,
            "search": {"enable": True}
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
    
    async def search_and_summarize(self, query: str, max_tokens: int = 2000) -> str:
        """
        ПОИСК С РАСШИРЕННЫМ ОТВЕТОМ (ДЛЯ РЕЖИМА ИНЖЕНЕРА)
        
        Используется, когда нужно найти и проанализировать информацию
        (например, драйвер на GitHub, документацию).
        
        Args:
            query: поисковый запрос
            max_tokens: максимальная длина ответа (по умолчанию 2000)
        
        Returns:
            str: развёрнутый ответ с анализом
        """
        await self.ensure_session()
        
        engineer_instruction = """Ты — инженер-робототехник. 
Найди информацию по запросу и предоставь структурированный ответ.
Если это код или драйвер — опиши, как его адаптировать под робота.
Отвечай подробно, но по делу."""
        
        messages = [
            {"role": "user", "text": query}
        ]
        
        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": 0.5,
                "maxTokens": max_tokens,
                "stream": False
            },
            "messages": messages,
            "instruction": engineer_instruction,
            "search": {"enable": True}
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
        """Создаёт объект ответа для ошибки"""
        class Response:
            def __init__(self, content, text):
                self.content = content
                self.action = None
                self.text = text
                self.tool_calls = None
        return Response(content=error_msg, text=error_msg)
    
    async def close(self):
        """Закрывает HTTP-сессию"""
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
        """ОТПРАВИТЬ СООБЩЕНИЕ С АВТОМАТИЧЕСКИМ ДОБАВЛЕНИЕМ В ИСТОРИЮ"""
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
    
    async def chat_stream(self, user_message: str, context: str = None) -> AsyncIterator[str]:
        """ПОТОКОВЫЙ ЧАТ С АВТОМАТИЧЕСКИМ ДОБАВЛЕНИЕМ В ИСТОРИЮ"""
        message = user_message
        if context:
            message = f"{context}\n\n{user_message}"
        
        self.history.append({"role": "user", "content": message})
        
        full_response = ""
        async for chunk in self.generate_stream(self.history):
            full_response += chunk
            yield chunk
        
        if full_response:
            self.history.append({"role": "assistant", "content": full_response})
    
    async def search_web(self, query: str, max_tokens: int = 500) -> str:
        """Поиск в интернете (без сохранения в историю диалога)"""
        return await super().search_web(query, max_tokens)
    
    async def search_and_summarize(self, query: str, max_tokens: int = 2000) -> str:
        """Расширенный поиск (без сохранения в историю диалога)"""
        return await super().search_and_summarize(query, max_tokens)
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.history = []
        logger.info("🧹 История диалога очищена")
    
    def get_stats(self) -> Dict:
        """Возвращает статистику"""
        total_chars = sum(len(msg.get("content", "")) for msg in self.history)
        return {
            "messages_count": len(self.history),
            "estimated_tokens": total_chars // 4,
            "has_instruction": self.instruction is not None,
            "model_limits": self.get_model_limits()
        }
    
    async def close(self):
        """Закрывает HTTP-сессию"""
        await super().close()


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ YANDEXGPT CLIENT С ПОДДЕРЖКОЙ СТРИМИНГА")
        print("="*60 + "\n")
        
        print("⚠️ Для работы нужен API ключ Yandex Cloud.")
        print("   Укажите folder_id и api_key в коде демонстрации.\n")
        
        FOLDER_ID = "your-folder-id"
        API_KEY = "your-api-key"
        
        if FOLDER_ID == "your-folder-id":
            print("❌ Укажите реальные FOLDER_ID и API_KEY для демонстрации.")
            return
        
        client = YandexGPTClient(
            folder_id=FOLDER_ID,
            api_key=API_KEY,
            temperature=0.7,
            max_tokens=4000
        )
        
        try:
            # Лимиты модели
            limits = client.get_model_limits()
            print(f"📊 Лимиты модели: input={limits['max_input_tokens']}, output={limits['max_output_tokens']}\n")
            
            # Демонстрация потоковой генерации
            print("📝 Потоковая генерация кода:")
            print("-" * 40)
            async for chunk in client.generate_stream([
                {"role": "user", "content": "Напиши функцию на Python для расчёта расстояния между двумя точками в 3D."}
            ]):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 40)
            
            # Демонстрация поиска
            print("\n🔍 Поиск в интернете: 'Курс доллара сегодня'")
            usd_rate = await client.search_web("Курс доллара сегодня", max_tokens=200)
            print(f"   Ответ: {usd_rate}")
            
            # Демонстрация расширенного поиска
            print("\n📚 Расширенный поиск: 'ROS2 драйвер для YDLIDAR X4'")
            driver_info = await client.search_and_summarize("ROS2 драйвер для YDLIDAR X4", max_tokens=1000)
            print(f"   Ответ: {driver_info[:300]}...")
            
        finally:
            await client.close()
        
        print("\n" + "="*60)
        print("✅ Демонстрация завершена")
        print("="*60 + "\n")
    
    asyncio.run(demo())