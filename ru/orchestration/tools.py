#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ orchestration/tools.py — ИНСТРУМЕНТЫ АГЕНТА (ТО, ЧТО РОБОТ МОЖЕТ ДЕЛАТЬ)     ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «руки» робота. Каждый инструмент — это одно действие, которое          ║
║   робот может выполнить: поехать, повернуться, заговорить, запомнить.        ║
║                                                                              ║
║   LLM принимает решение и возвращает JSON:                                   ║
║   {"action": "move_forward", "params": {"speed": 300}}                       ║
║                                                                              ║
║   Агент находит инструмент с именем "move_forward" и вызывает его.           ║
║                                                                              ║
║ ГЛАВНАЯ ЦЕННОСТЬ — ВОЗМОЖНОСТЬ РАСШИРЕНИЯ:                                   ║
║                                                                              ║
║   OpenGrall — это НЕ набор жёстко заданных команд. Это ПЛАТФОРМА,            ║
║   которую вы можете расширять под свои задачи.                               ║
║                                                                              ║
║   Хотите, чтобы робот:                                                       ║
║     • управлял манипулятором?                                                ║
║     • включал тепловизор?                                                    ║
║     • танцевал заученный танец?                                              ║
║     • преодолевал высокое препятствие?                                       ║
║     • снимал показания GPS?                                                  ║
║     • искал информацию в интернете?                                          ║
║     • сохранял калибровки и заметки между сессиями?                          ║
║                                                                              ║
║   Всё это делается добавлением НОВОГО ИНСТРУМЕНТА.                           ║
║   Не нужно менять ядро. Не нужно перекомпилировать агент.                    ║
║   Просто создайте класс, унаследованный от Tool, и добавьте его в список.    ║
║                                                                              ║
║ КАК ДОБАВИТЬ СВОЙ ИНСТРУМЕНТ:                                                ║
║                                                                              ║
║   1. Создайте класс, наследующий от Tool:                                    ║
║                                                                              ║
║      class MyGPSTool(Tool):                                                  ║
║          name = "get_gps"                                                    ║
║          description = "Получить текущие координаты GPS"                      ║
║                                                                              ║
║          def __init__(self, gps_module):                                     ║
║              self.gps = gps_module                                           ║
║              self.latency = 0.5                                              ║
║                                                                              ║
║          async def forward(self):                                            ║
║              coords = await self.gps.get_coordinates()                       ║
║              return f"Координаты: {coords['lat']}, {coords['lon']}"          ║
║                                                                              ║
║   2. Добавьте инструмент в agent_v5.py (метод _create_tools):                ║
║                                                                              ║
║      self.tools.append(MyGPSTool(gps_module))                                ║
║                                                                              ║
║   3. Опишите инструмент в системном промпте LLM:                            ║
║                                                                              ║
║      "ТВОИ ВОЗМОЖНОСТИ: ..., get_gps()"                                      ║
║                                                                              ║
║   ГОТОВО. LLM теперь может вызывать ваш инструмент.                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any

from core.protocol_v5 import RobotProtocolV5

logger = logging.getLogger(__name__)


# ================================================================
# БАЗОВЫЙ КЛАСС ИНСТРУМЕНТА
# ================================================================

class Tool:
    """
    БАЗОВЫЙ КЛАСС ДЛЯ ВСЕХ ИНСТРУМЕНТОВ
    
    Чтобы создать свой инструмент, унаследуйтесь от этого класса
    и переопределите метод forward().
    
    АТРИБУТЫ:
        name: str — имя инструмента (используется в JSON от LLM)
        description: str — описание (для системного промпта)
        latency: float — примерное время выполнения в секундах
    
    ПРИМЕР:
        class MyTool(Tool):
            name = "my_action"
            description = "Делает что-то полезное"
            
            async def forward(self, param1: str, param2: int = 10):
                return "Результат выполнения"
    """
    name = "base_tool"
    description = "Базовый инструмент"
    latency = 0.1
    
    async def forward(self, **kwargs):
        raise NotImplementedError("Переопределите forward() в своём инструменте")


# ================================================================
# ИНСТРУМЕНТЫ ДВИЖЕНИЯ
# ================================================================

class MoveForwardTool(Tool):
    name = "move_forward"
    description = "Двигаться вперёд"
    
    def __init__(self, ws_client, agent):
        self.ws = ws_client
        self.agent = agent
        self.latency = 0.5
    
    async def forward(self, speed: int = 512, duration: Optional[float] = None, distance: Optional[float] = None):
        if self.agent.vlm_scanner:
            self.agent.vlm_scanner.on_movement()
        
        msg = RobotProtocolV5.create_motor_command(
            left=speed, right=speed,
            duration=duration,
            distance=distance,
            source="agent"
        )
        await self.ws.send(msg)
        return f"Движение вперёд (speed={speed})"


class MoveBackwardTool(Tool):
    name = "move_backward"
    description = "Двигаться назад"
    
    def __init__(self, ws_client, agent):
        self.ws = ws_client
        self.agent = agent
        self.latency = 0.5
    
    async def forward(self, speed: int = 512, duration: Optional[float] = None, distance: Optional[float] = None):
        if self.agent.vlm_scanner:
            self.agent.vlm_scanner.on_movement()
        
        msg = RobotProtocolV5.create_motor_command(
            left=-speed, right=-speed,
            duration=duration,
            distance=distance,
            source="agent"
        )
        await self.ws.send(msg)
        return f"Движение назад (speed={speed})"


class TurnLeftTool(Tool):
    name = "turn_left"
    description = "Повернуть налево"
    
    def __init__(self, ws_client, agent):
        self.ws = ws_client
        self.agent = agent
        self.latency = 0.4
    
    async def forward(self, speed: int = 512, duration: Optional[float] = None, angle: Optional[float] = None):
        if self.agent.vlm_scanner:
            self.agent.vlm_scanner.on_movement()
        
        msg = RobotProtocolV5.create_motor_command(
            left=-speed, right=speed,
            duration=duration,
            angle=angle,
            source="agent"
        )
        await self.ws.send(msg)
        return f"Поворот налево (speed={speed})"


class TurnRightTool(Tool):
    name = "turn_right"
    description = "Повернуть направо"
    
    def __init__(self, ws_client, agent):
        self.ws = ws_client
        self.agent = agent
        self.latency = 0.4
    
    async def forward(self, speed: int = 512, duration: Optional[float] = None, angle: Optional[float] = None):
        if self.agent.vlm_scanner:
            self.agent.vlm_scanner.on_movement()
        
        msg = RobotProtocolV5.create_motor_command(
            left=speed, right=-speed,
            duration=duration,
            angle=angle,
            source="agent"
        )
        await self.ws.send(msg)
        return f"Поворот направо (speed={speed})"


class StopTool(Tool):
    name = "stop"
    description = "Остановить двигатели"
    
    def __init__(self, ws_client):
        self.ws = ws_client
        self.latency = 0.1
    
    async def forward(self):
        msg = RobotProtocolV5.create_message(
            source="agent",
            capability="motor.stop",
            source_type="agent"
        )
        await self.ws.send(msg)
        return "Двигатели остановлены"


class WaitTool(Tool):
    name = "wait"
    description = "Подождать указанное количество секунд"
    
    def __init__(self):
        self.latency = 0.0
    
    async def forward(self, seconds: float):
        self.latency = seconds
        await asyncio.sleep(seconds)
        return f"Ожидание {seconds}с завершено"


class SetLightTool(Tool):
    name = "set_light"
    description = "Включить/выключить свет"
    
    def __init__(self, ws_client):
        self.ws = ws_client
        self.latency = 0.1
    
    async def forward(self, state: bool):
        msg = RobotProtocolV5.create_message(
            source="agent",
            capability="light.set",
            data={"state": state},
            source_type="agent"
        )
        await self.ws.send(msg)
        return f"Свет {'включён' if state else 'выключен'}"


# ================================================================
# ИНСТРУМЕНТЫ ОБЩЕНИЯ
# ================================================================

class SpeakTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ РЕЧИ
    
    Может просто произнести текст или инициировать диалог с ожиданием ответа.
    """
    name = "speak"
    description = "Произнести текст. Если wait=True — ждать ответа человека."
    
    def __init__(self, tts_engine, agent):
        self.tts = tts_engine
        self.agent = agent
        self.latency = 0.5
    
    async def forward(self, text: str, wait: bool = False):
        if wait:
            human_nearby = await self._is_human_nearby()
            if not human_nearby:
                return "Никого рядом нет"
        
        await self.tts.speak(text)
        logger.info(f"🤖 Робот говорит: {text[:50]}...")
        self.agent.dialog.add_turn("", text, intent="conversation", source="agent")
        
        if wait:
            try:
                answer = await asyncio.wait_for(self.agent._wait_for_speech(), timeout=10.0)
                if answer:
                    logger.info(f"👤 Человек: {answer}")
                    self.agent.episodic_memory.add_conversation(text, answer)
                    self.agent.dialog.add_turn(answer, "", intent="conversation", source="human")
                    return f"Человек ответил: {answer}"
                return "Человек не ответил"
            except asyncio.TimeoutError:
                return "Человек не ответил"
        
        return f"Сказано: {text[:50]}..."
    
    async def _is_human_nearby(self) -> bool:
        if self.agent.vlm_scanner:
            latest = self.agent.vlm_scanner.get_latest()
            if latest:
                objects = latest.get("data", {}).get("objects", [])
                for obj in objects:
                    if obj.get("name") == "человек" and obj.get("distance", 10) < 3.0:
                        return True
        
        sensor_data = self.agent.sensor_memory.get("lidar")
        if sensor_data:
            clusters = sensor_data.data.get("clusters", [])
            for c in clusters:
                if c.get("type") == "human" and c.get("min_distance", 10) < 3.0:
                    return True
        
        return False


class AskHumanTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ВОПРОСА ЧЕЛОВЕКУ С ЗАПОМИНАНИЕМ ОТВЕТА
    """
    name = "ask_human"
    description = "Спросить человека о чём-то и запомнить ответ для будущего"
    
    def __init__(self, agent):
        self.agent = agent
        self.latency = 0.5
    
    async def forward(self, question: str):
        speak_tool = None
        for t in self.agent.tools:
            if t.name == "speak":
                speak_tool = t
                break
        
        if speak_tool:
            human_nearby = await speak_tool._is_human_nearby()
            if not human_nearby:
                return "Никого рядом нет"
        
        await self.agent.tts.speak(question)
        logger.info(f"🤖 Робот спрашивает: {question}")
        
        try:
            answer = await asyncio.wait_for(self.agent._wait_for_speech(), timeout=10.0)
            if answer:
                logger.info(f"👤 Человек ответил: {answer}")
                self.agent.episodic_memory.add_human_instruction(
                    question=question,
                    answer=answer,
                    context={
                        "asked_at": time.time(),
                        "intent": self.agent.dialog.get_primary_intent()
                    }
                )
                return f"Человек ответил: {answer}. Запомнил."
            return "Человек не ответил"
        except asyncio.TimeoutError:
            return "Человек не ответил"


# ================================================================
# ПОИСК В ИНТЕРНЕТЕ
# ================================================================

class SearchWebTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ПОИСКА ИНФОРМАЦИИ В ИНТЕРНЕТЕ
    
    Использует YandexGPT с включённым поиском для получения актуальной информации.
    Вызывается, когда локальная LLM не знает ответа или вопрос требует свежих данных.
    """
    name = "search_web"
    description = "Найти информацию в интернете (погода, новости, курсы валют, даты, факты)"
    
    def __init__(self, agent):
        self.agent = agent
        self.latency = 2.0  # поиск может занять пару секунд
    
    async def forward(self, query: str):
        if not self.agent.yandex_client:
            return "Поиск в интернете недоступен (не настроен YandexGPT)"
        
        try:
            logger.info(f"🔍 Поиск в интернете: {query}")
            answer = await self.agent.yandex_client.search_web(query)
            return answer if answer else "Не удалось найти информацию"
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return f"Ошибка поиска: {e}"


# ================================================================
# ИНСТРУМЕНТЫ ВИЗУАЛЬНОЙ ПАМЯТИ
# ================================================================

class RememberObjectTool(Tool):
    name = "remember_object"
    description = "Запомнить текущий видимый объект"
    
    def __init__(self, vision, ws_client):
        self.vision = vision
        self.ws = ws_client
        self.latency = 1.0
    
    async def forward(self, name: str):
        self.vision.frame_future = asyncio.Future()
        await self.ws.send({"type": "capture_frame"})
        
        try:
            import base64
            import numpy as np
            import cv2
            
            image_b64 = await asyncio.wait_for(self.vision.frame_future, timeout=5.0)
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.vision.memory.save_object(name, img)
            return f"Объект '{name}' запомнен"
        except asyncio.TimeoutError:
            return "Не удалось получить кадр"


class FindObjectTool(Tool):
    name = "find_object"
    description = "Найти запомненный объект"
    
    def __init__(self, vision, ws_client):
        self.vision = vision
        self.ws = ws_client
        self.latency = 1.0
    
    async def forward(self, name: str):
        self.vision.frame_future = asyncio.Future()
        await self.ws.send({"type": "capture_frame"})
        
        try:
            import base64
            import numpy as np
            import cv2
            
            image_b64 = await asyncio.wait_for(self.vision.frame_future, timeout=5.0)
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            result = self.vision.memory.find_object(name, img)
            if result["found"]:
                return f"Объект '{name}' найден! (метод: {result['method']}, уверенность: {result['confidence']:.2f})"
            return f"Объект '{name}' не обнаружен"
        except asyncio.TimeoutError:
            return "Не удалось получить кадр"


class SearchByTextTool(Tool):
    name = "search_by_text"
    description = "Поиск объектов по текстовому описанию"
    
    def __init__(self, vision):
        self.vision = vision
        self.latency = 0.3
    
    async def forward(self, query: str):
        results = self.vision.memory.search_by_text(query)
        if results:
            names = [r["name"] for r in results[:3]]
            return f"Найдено: {', '.join(names)}"
        return "Ничего не найдено"


# ================================================================
# ИНСТРУМЕНТЫ МАРШРУТОВ
# ================================================================

class RecordRouteTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ЗАПИСИ МАРШРУТА
    """
    name = "record_route"
    description = "Записать маршрут. action='start' или 'stop', name — имя маршрута"
    
    def __init__(self, agent):
        self.agent = agent
        self.latency = 0.05
    
    async def forward(self, action: str, name: str = None):
        if action == "start":
            if not name:
                return "Укажите имя маршрута"
            self.agent.recording_route = name
            self.agent.route_commands = []
            return f"Запись маршрута '{name}' начата"
        
        elif action == "stop":
            if not self.agent.recording_route:
                return "Нет активной записи"
            name = self.agent.recording_route
            self.agent.route_memory.save_route(name, self.agent.route_commands)
            self.agent.recording_route = None
            self.agent.route_commands = []
            return f"Маршрут '{name}' сохранён ({len(self.agent.route_memory.get_route(name))} команд)"
        
        else:
            return f"Неизвестное действие: {action}. Используйте 'start' или 'stop'"


class ExecuteRouteTool(Tool):
    name = "execute_route"
    description = "Выполнить сохранённый маршрут"
    
    def __init__(self, agent):
        self.agent = agent
        self.latency = 0.1
    
    async def forward(self, name: str):
        commands = self.agent.route_memory.get_route(name)
        if not commands:
            return f"Маршрут '{name}' не найден"
        asyncio.create_task(self.agent.execute_route_async(commands))
        return f"Начинаю выполнение маршрута '{name}' ({len(commands)} команд)"


# ================================================================
# ИНСТРУМЕНТЫ ПЛАНИРОВАНИЯ
# ================================================================

class ComposePlanTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ДОЛГОСРОЧНОГО ПЛАНИРОВАНИЯ
    """
    name = "compose_plan"
    description = "Составить план действий для сложной задачи"
    
    def __init__(self, agent):
        self.agent = agent
        self.latency = 0.5
    
    async def forward(self, goal: str, context: Dict = None):
        sensor_summary = self.agent.sensor_memory.get_summaries()
        available_tools = [t.name for t in self.agent.tools]
        
        prompt = f"""Ты планировщик действий робота.

Цель: {goal}

Текущая обстановка:
{sensor_summary}

Доступные инструменты: {', '.join(available_tools)}

Разбей достижение цели на последовательность шагов. Каждый шаг должен быть выполним с помощью доступных инструментов.

Ответь ТОЛЬКО JSON:
{{
    "steps": [
        {{"action": "название_инструмента", "parameters": {{...}}, "description": "что делаем"}}
    ],
    "reasoning": "почему такой план"
}}"""
        
        try:
            response = await self.agent.llm.generate([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                if plan.get("steps"):
                    steps_text = [s.get("description", s.get("action", "")) for s in plan["steps"]]
                    self.agent.decision_memory.set_task(
                        steps=steps_text,
                        reasoning=plan.get("reasoning", ""),
                        context=context or {},
                        task_name=goal[:50]
                    )
                    return f"План составлен: {len(plan['steps'])} шагов"
            return "Не удалось составить план"
        except Exception as e:
            logger.error(f"Ошибка составления плана: {e}")
            return f"Ошибка: {e}"


# ================================================================
# ИНСТРУМЕНТЫ ДЛЯ РАБОТЫ С ФАЙЛАМИ (ДОЛГОСРОЧНАЯ ПАМЯТЬ АГЕНТА)
# ================================================================

class FileWriteTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ЗАПИСИ ФАЙЛОВ
    
    Позволяет LLM сохранять информацию между сессиями:
    - Калибровочные данные (смещение лидара, скорость моторов)
    - Результаты экспериментов (какая стратегия сработала лучше)
    - Конфигурационные правки
    - Собственные заметки "для себя"
    
    Безопасность:
    - Работает только внутри BASE_DIR
    - Не перезаписывает системные файлы
    - Логирует все операции
    """
    name = "write_file"
    description = "Создать или перезаписать файл на диске. Используй для сохранения калибровок, настроек, заметок между сессиями."
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "..", "data", "agent_files")
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.latency = 0.05
        logger.info(f"✅ FileWriteTool инициализирован (base_path={self.base_path})")
    
    async def forward(self, path: str, content: str, append: bool = False) -> str:
        full_path = os.path.abspath(os.path.join(self.base_path, path))
        if not full_path.startswith(self.base_path):
            logger.warning(f"⚠️ Попытка записи за пределами BASE_DIR: {path}")
            return f"Ошибка: запрещено писать файлы за пределами {self.base_path}"
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        try:
            mode = "a" if append else "w"
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            
            action = "добавлен в" if append else "записан в"
            logger.info(f"📄 Файл {path} {action} агентом")
            return f"Файл {path} успешно {action} ({len(content)} символов)"
        except Exception as e:
            logger.error(f"❌ Ошибка записи файла {path}: {e}")
            return f"Ошибка записи файла: {e}"


class FileReadTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ЧТЕНИЯ ФАЙЛОВ
    
    Позволяет LLM читать сохранённые ранее данные.
    """
    name = "read_file"
    description = "Прочитать содержимое файла. Используй для получения сохранённых калибровок, настроек, заметок."
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "..", "data", "agent_files")
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.latency = 0.03
    
    async def forward(self, path: str) -> str:
        full_path = os.path.abspath(os.path.join(self.base_path, path))
        if not full_path.startswith(self.base_path):
            return f"Ошибка: запрещено читать файлы за пределами {self.base_path}"
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content if content else "(файл пуст)"
        except FileNotFoundError:
            return f"Файл {path} не найден"
        except Exception as e:
            return f"Ошибка чтения файла: {e}"


class FileListTool(Tool):
    """
    ИНСТРУМЕНТ ДЛЯ ПРОСМОТРА СПИСКА ФАЙЛОВ
    
    Позволяет LLM узнать, какие файлы она сохраняла ранее.
    """
    name = "list_files"
    description = "Показать список сохранённых файлов в указанной директории"
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "..", "data", "agent_files")
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.latency = 0.02
    
    async def forward(self, subdir: str = "") -> str:
        full_path = os.path.abspath(os.path.join(self.base_path, subdir))
        if not full_path.startswith(self.base_path):
            return f"Ошибка: запрещён доступ за пределами {self.base_path}"
        
        try:
            items = os.listdir(full_path)
            if not items:
                return f"Директория {subdir or '.'} пуста"
            
            result = [f"Содержимое {subdir or 'корневой директории'}:"]
            for item in sorted(items):
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    result.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    result.append(f"  📄 {item} ({size} байт)")
            return "\n".join(result)
        except FileNotFoundError:
            return f"Директория {subdir} не найдена"
        except Exception as e:
            return f"Ошибка чтения директории: {e}"