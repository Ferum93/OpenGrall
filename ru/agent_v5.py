#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ agent_v5.py — ГЛАВНЫЙ КЛАСС РОБОТА (МОЗГ)                                    ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «сознание» Гралла. Агент слушает команды (голосом или по сети),        ║
║   собирает данные с сенсоров, думает через LLM, отдаёт команды движения.     ║
║                                                                              ║
║ АРХИТЕКТУРА:                                                                 ║
║   • SensorMemory — единое хранилище сенсоров с весами и затуханием           ║
║   • WeightCalculator — динамический расчёт весов (аномалии, латентность)     ║
║   • VLMScanner — асинхронное фоновое сканирование                            ║
║   • LLMDecisionMemory — кэширование решений + управление задачами            ║
║   • StrategyLearner — эволюционное самообучение + автономное целеполагание   ║
║   • EpisodicMemory — хранение инструкций человека                            ║
║   • OutcomeEvaluator — оценка результатов через LLM и похвала                ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ:                                                            ║
║   1. Робот видит мир через сенсоры (лидар, VLM, одометрия)                   ║
║   2. Все данные складываются в SensorMemory                                  ║
║   3. Контекст собирается ContextBuilder и отправляется в LLM                 ║
║   4. LLM возвращает JSON-команду: {"action": "move_forward", "params": {...}}║
║   5. Инструмент (Tool) выполняет команду                                     ║
║   6. Если опасно — TinyML на ESP32 блокирует команду (право вето)            ║
║                                                                              ║
║ РЕЖИМЫ РАБОТЫ:                                                               ║
║   • Обычный — ждёт wake word «Гралл»                                         ║
║   • Интерактивный — слушает всё, сам инициирует диалог                       ║
║   • Заморозка (freeze) — приостанавливает принятие решений для отладки       ║
║                                                                              ║
║ ЧТО НУЖНО НАСТРОИТЬ ПОД ВАШУ ПЛАТФОРМУ:                                      ║
║   1. config.py — адрес сервера, модель LLM, путь к Vosk                      ║
║   2. Если у вас другой микроконтроллер — измените protocol_adapters.py       ║
║   3. Если у вас другая платформа (гексапод, дрон) —                          ║
║      переобучите TinyML и добавьте новые инструменты в tools.py              ║
║   4. Если нет VLM — в config.py поставьте VLM_MODEL = None                   ║
║                                                                              ║
║ КАК ЗАПУСТИТЬ:                                                               ║
║   python agent_v5.py                                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
import random
import os
from typing import Optional, Dict, Any, List

# Подключаем все модули фреймворка
from config import *
from core.protocol_v5 import RobotProtocolV5, SourceType
from core.websocket_client import WebSocketClient
from core.feedback_learner import FeedbackLearner
from core.weight_calculator import WeightCalculator
from core.protocol_adapters import ProtocolAdapters
from core.context_builder import ContextBuilder
from core.llm_decision_memory import LLMDecisionMemory
from core.strategy_learner import StrategyLearner
from core.sensor_memory import SensorMemory

from sensors.listener import VoiceListener
from sensors.speaker import TTSEngine
from sensors.lidar_processor import SensorDataCollector
from sensors.vlm_scanner import VLMScanner

from vision.visual_memory import VisualMemory
from vision.vlm_client import VLMClient

from memory.dialog_context import DialogContext
from memory.route_memory import RouteMemory
from memory.episodic_memory import EpisodicMemory

from orchestration.tools import *
from agents.llm_client import LocalLLM
from agents.yandex_gpt_client import YandexGPTClient

logger = logging.getLogger(__name__)


class RobotAgentV5:
    """
    ГЛАВНЫЙ КЛАСС РОБОТА — МОЗГ ГРАЛЛА
    
    Внутри него живут все компоненты:
    - Память (сенсорная, эпизодическая, решений, маршрутов, визуальная)
    - Сенсоры (лидар, VLM, одометрия, речь)
    - Инструменты (движение, свет, речь, память)
    - LLM (локальная через Ollama или YandexGPT)
    """
    
    def __init__(self):
        # ========== СОСТОЯНИЕ ==========
        self.running = True
        self.connected = False
        self.frozen = False  # заморозка агента для отладки
        self.last_command_time = time.time()
        
        # ========== ОСНОВНЫЕ КОМПОНЕНТЫ ==========
        self.ws: Optional[WebSocketClient] = None
        self.listener: Optional[VoiceListener] = None
        self.tts: Optional[TTSEngine] = None
        self.llm: Optional[LocalLLM] = None
        self.vlm: Optional[VLMClient] = None
        self.yandex_client: Optional[YandexGPTClient] = None  # для поиска в интернете
        
        # ========== ПАМЯТЬ (5 видов) ==========
        self.vision_memory: Optional[VisualMemory] = None
        self.route_memory: Optional[RouteMemory] = None
        self.episodic_memory: Optional[EpisodicMemory] = None
        self.sensor_memory = SensorMemory(max_age=SENSOR_MAX_AGE)
        
        # ========== ОБУЧЕНИЕ И ВЕСА ==========
        self.feedback_learner = FeedbackLearner(FEEDBACK_FILE)
        self.weight_calculator = WeightCalculator()
        
        # ========== КОНТЕКСТ И РЕШЕНИЯ ==========
        self.dialog = DialogContext()
        self.context_builder = ContextBuilder(self.weight_calculator, self.sensor_memory)
        self.decision_memory = LLMDecisionMemory()
        
        # ========== СЕНСОРЫ ==========
        self.sensor_collector = SensorDataCollector()
        self.vlm_scanner: Optional[VLMScanner] = None
        self.strategy_learner: Optional[StrategyLearner] = None
        
        # ========== ИНСТРУМЕНТЫ ==========
        self.tools: List[Tool] = []
        
        # ========== ЗАПИСЬ МАРШРУТОВ ==========
        self.recording_route: Optional[str] = None
        self.route_commands: List[Dict] = []
        
        # ========== ИНТЕРАКТИВНЫЙ РЕЖИМ ==========
        self.interactive_mode = INTERACTIVE_MODE_DEFAULT
        self._speech_future: Optional[asyncio.Future] = None
        self._conversation_active = False
        
        # ========== ДЛЯ ОЦЕНКИ ДЕЙСТВИЙ ==========
        self._last_action_id: Optional[str] = None
        self._last_strategy_id: Optional[str] = None
        self._last_task_type: Optional[str] = None
        
        # ========== СТАТИСТИКА ==========
        self.stats = {
            "messages_received": 0,
            "commands_sent": 0,
            "reflexes_received": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "strategy_hits": 0,
            "strategy_success": 0,
            "vlm_scans": 0,
            "interactive_questions": 0,
            "praise_received": 0,
            "evaluations_completed": 0,
            "wake_activations": 0,
            "emergency_stops": 0,
            "start_time": time.time()
        }
        
        self.adapters = ProtocolAdapters()
        self.last_reflex_time = 0
        
        # Единый диалог для LLM (системный промпт зашит в модель)
        self.conversation: List[Dict] = []
        
        logger.info("🤖 RobotAgentV5 создан")
    
    # ================================================================
    # ИНИЦИАЛИЗАЦИЯ LLM
    # ================================================================
    
    async def _setup_llm_model(self):
        """
        СОЗДАЁТ КАСТОМНУЮ МОДЕЛЬ OLLAMA С СИСТЕМНЫМ ПРОМПТОМ
        
        Системный промпт зашивается В МОДЕЛЬ, а не передаётся в каждом запросе.
        Это экономит токены и гарантирует, что LLM не забудет формат ответа.
        """
        import subprocess
        
        # Обновлённый список инструментов (22+ штук)
        tools_list = """
move_forward(speed, duration, distance), move_backward(speed, duration, distance),
turn_left(speed, duration, angle), turn_right(speed, duration, angle),
stop(), wait(seconds), set_light(state),
speak(text, wait), ask_human(question),
search_web(query), web_fetch(url),
remember_object(name), find_object(name), search_by_text(query),
record_route(action, name), execute_route(name),
compose_plan(goal),
write_file(path, content, append), read_file(path), list_files(subdir),
apply_patch(path, find, replace), execute_code(code)
"""
        
        modelfile = f"""FROM {LLM_MODEL}
PARAMETER temperature {LLM_TEMPERATURE}
PARAMETER top_p {LLM_TOP_P}
SYSTEM \"\"\"
Ты робот Гралл. Габариты: {ROBOT_DIMENSIONS['length']}×{ROBOT_DIMENSIONS['width']}×{ROBOT_DIMENSIONS['height']}см.

СИСТЕМА НАВИГАЦИИ:
- 0° = ↑ = прямо (куда смотрит робот)
- 90° = → = вправо
- 180° = ↓ = назад
- 270° = ← = влево

ДАННЫЕ ЛИДАРА:
8 секторов: front, front_left, left, back_left, back, back_right, right, front_right
Стрелки (↑ ↓ ← → ↖ ↗ ↙ ↘) — направление движения объектов.

ОБЪЕКТЫ:
- человек: соблюдай дистанцию >1м
- стена: статическое препятствие

ТВОИ ВОЗМОЖНОСТИ:
{tools_list}

ФОРМАТ ОТВЕТА:
Только JSON: {{\"action\":\"move_forward\",\"params\":{{\"speed\":300}},\"reasoning\":\"...\"}}
Если хочешь просто ответить текстом: {{\"text\":\"твой ответ\"}}

ПРАВИЛА:
- Лидар — основной источник (вес 0.95)
- При опасности (<{DANGER_DISTANCE_CM}см) — остановись
- Ты помнишь всю историю диалога
- Можешь прогнозировать движение по стрелкам
- Если не знаешь ответа — используй search_web(query)
- Не стесняйся говорить с человеком, если есть что обсудить
\"\"\"""
"""
        modelfile_path = "/tmp/GrallModelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile)
        
        try:
            subprocess.run(["ollama", "create", "grall-robot", "-f", modelfile_path], 
                          check=True, capture_output=True)
            logger.info("✅ Модель grall-robot создана")
            return "grall-robot"
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка создания модели: {e.stderr.decode()}")
            return LLM_MODEL
    
    # ================================================================
    # ИНИЦИАЛИЗАЦИЯ ВСЕХ КОМПОНЕНТОВ
    # ================================================================
    
    async def setup(self):
        """ЗАПУСКАЕТ ВСЕ КОМПОНЕНТЫ РОБОТА"""
        logger.info("🚀 Запуск RobotAgentV5")
        
        # 1. WebSocket
        self.ws = WebSocketClient(WS_URL, self.on_ws_message)
        asyncio.create_task(self.ws.connect())
        
        # 2. Речь
        self.listener = VoiceListener(
            VOSK_MODEL_PATH,
            threshold=SOUND_THRESHOLD,
            wake_words=WAKE_WORDS,
            active_timeout=VOICE_ACTIVE_TIMEOUT
        )
        self.listener.set_callback(self.on_speech_recognized)
        self.tts = TTSEngine(
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            pitch=TTS_PITCH,
            amplitude=TTS_AMPLITUDE
        )
        
        # 3. Память
        self.vision_memory = VisualMemory(VISUAL_MEMORY_FILE)
        self.route_memory = RouteMemory(ROUTES_FILE)
        self.episodic_memory = EpisodicMemory(
            max_episodes=1000,
            storage_path=EPISODES_FILE
        )
        
        # Загружаем сохранённые данные
        self.decision_memory.load_from_file(DECISIONS_FILE)
        if os.path.exists(SENSOR_MEMORY_FILE):
            self.sensor_memory.load_from_file(SENSOR_MEMORY_FILE)
        
        # 4. LLM
        model_name = await self._setup_llm_model()
        self.llm = LocalLLM(
            model=model_name,
            base_url=OLLAMA_URL,
            max_history_messages=LLM_MAX_HISTORY
        )
        
        # 5. YandexGPT (для поиска в интернете)
        try:
            if YANDEX_FOLDER_ID and YANDEX_API_KEY:
                from agents.yandex_gpt_client import YandexGPTClient
                self.yandex_client = YandexGPTClient(
                    folder_id=YANDEX_FOLDER_ID,
                    api_key=YANDEX_API_KEY
                )
                logger.info("✅ YandexGPT клиент активирован")
        except Exception as e:
            logger.warning(f"⚠️ YandexGPT недоступен: {e}")
        
        # 6. StrategyLearner
        self.strategy_learner = StrategyLearner(self.llm, STRATEGIES_FILE)
        self.strategy_learner.evaluator.set_agent(self)
        await self.strategy_learner.start_evaluator()
        
        # 7. VLM Scanner
        if VLM_MODEL and self.ws:
            try:
                from vision.vlm_client import VLMClient
                self.vlm = VLMClient(
                    model_name=VLM_MODEL,
                    load_in_4bit=VLM_LOAD_IN_4BIT
                )
                self.vlm_scanner = VLMScanner(
                    vlm_client=self.vlm,
                    ws_client=self.ws,
                    scan_interval=VLM_SCAN_INTERVAL,
                    idle_threshold=VLM_IDLE_THRESHOLD
                )
                self.vlm_scanner.set_weight_calculator(self.weight_calculator)
                self.vlm_scanner.set_sensor_memory(self.sensor_memory)
                await self.vlm_scanner.start()
                logger.info("✅ VLMScanner активирован")
                self.stats["vlm_available"] = True
            except Exception as e:
                logger.warning(f"⚠️ VLM недоступен: {e}")
                self.vlm_scanner = None
                self.stats["vlm_available"] = False
        else:
            self.stats["vlm_available"] = False
        
        # 8. Инструменты
        self._create_tools()
        
        # 9. Запускаем слушание
        await self.listener.start_listening()
        
        # 10. Приветствие
        await self.tts.speak("Гибридный агент активирован. Назовите меня Гралл.")
        logger.info("✅ Агент v5.0 готов")
        
        # 11. Фоновые задачи
        asyncio.create_task(self._stats_logger())
        asyncio.create_task(self._idle_learning_loop())
        asyncio.create_task(self._sensor_memory_cleanup())
    
    # ================================================================
    # СОЗДАНИЕ ИНСТРУМЕНТОВ
    # ================================================================
    
    def _create_tools(self):
        """
        СОЗДАЁТ ВСЕ ИНСТРУМЕНТЫ, КОТОРЫМИ РОБОТ МОЖЕТ ПОЛЬЗОВАТЬСЯ
        
        Всего 22+ инструментов:
        - Движение: 6 (move_forward, move_backward, turn_left, turn_right, stop, wait)
        - Общение: 2 (speak, ask_human)
        - Поиск: 2 (search_web, web_fetch)
        - Визуальная память: 3 (remember_object, find_object, search_by_text)
        - Маршруты: 2 (record_route, execute_route)
        - Планирование: 1 (compose_plan)
        - Система: 1 (set_light)
        - Файловые операции: 4 (write_file, read_file, list_files, apply_patch)
        - Выполнение кода: 1 (execute_code)
        """
        self.tools = [
            # Движение
            MoveForwardTool(self.ws, self),
            MoveBackwardTool(self.ws, self),
            TurnLeftTool(self.ws, self),
            TurnRightTool(self.ws, self),
            StopTool(self.ws),
            WaitTool(),
            # Система
            SetLightTool(self.ws),
            # Общение
            SpeakTool(self.tts, self),
            AskHumanTool(self),
            # Поиск
            SearchWebTool(self),
            WebFetchTool(),
            # Визуальная память
            RememberObjectTool(self.vision_memory, self.ws),
            FindObjectTool(self.vision_memory, self.ws),
            SearchByTextTool(self.vision_memory),
            # Маршруты
            RecordRouteTool(self),
            ExecuteRouteTool(self),
            # Планирование
            ComposePlanTool(self),
            # Файловые операции (долгосрочная память агента)
            FileWriteTool(),
            FileReadTool(),
            FileListTool(),
            ApplyPatchTool(),
            # Выполнение кода (самообучение)
            CodeExecutionTool(timeout=10),
        ]
        logger.info(f"✅ Создано {len(self.tools)} инструментов")
    
    # ================================================================
    # ОСНОВНОЙ ЦИКЛ
    # ================================================================
    
    async def run(self):
        """ЗАПУСКАЕТ ОСНОВНОЙ ЦИКЛ РОБОТА"""
        await self.setup()
        logger.info("🤖 Агент запущен")
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Остановка")
        finally:
            await self.shutdown()
    
    # ================================================================
    # ОБРАБОТКА WEBSOCKET СООБЩЕНИЙ
    # ================================================================
    
    async def on_ws_message(self, data: Dict):
        """ОБРАБОТЧИК ВСЕХ ВХОДЯЩИХ WEBSOCKET СООБЩЕНИЙ"""
        self.stats["messages_received"] += 1
        msg_type = data.get('type')
        
        # Рефлексы TinyML
        if msg_type == 'reflex' or data.get('capability') == 'tinyml.reflex':
            self.stats["reflexes_received"] += 1
            await self.handle_reflex_notification(data)
        
        # Подтверждение регистрации
        elif msg_type == 'registered':
            self.connected = True
            logger.info("✅ Зарегистрирован на сервере")
        
        # Батарея
        elif msg_type == 'battery':
            logger.info(f"🔋 Батарея: {data.get('level')}%")
        
        # Кадр для VLM
        elif msg_type == 'frame_data':
            if hasattr(self, '_pending_frame'):
                self._pending_frame.set_result(data.get('image', ''))
                delattr(self, '_pending_frame')
        
        # Результат выполнения команды
        elif msg_type == 'execution_result':
            await self.handle_execution_result(data)
        
        # ЗАМОРОЗКА АГЕНТА (для отладки)
        elif msg_type == 'freeze_agent':
            self.frozen = True
            await self.ws.send({"type": "agent_frozen"})
            logger.info("🧊 Агент заморожен")
            await self.tts.speak("Режим отладки. Я на паузе.")
        
        # РАЗМОРОЗКА АГЕНТА
        elif msg_type == 'resume_agent':
            self.frozen = False
            await self.ws.send({"type": "agent_resumed"})
            logger.info("▶ Агент разморожен")
            await self.tts.speak("Продолжаю работу.")
        
        # ОТЛАДОЧНЫЙ ПРОМПТ (от дашборда)
        elif msg_type == 'debug_prompt':
            asyncio.create_task(self._handle_debug_prompt(data))
        
        # КОМАНДА ОТ ОПЕРАТОРА
        elif msg_type == 'motor_command':
            # Логируем для обучения
            self.feedback_learner.add_feedback({
                "task_type": "motor_command",
                "success": True,
                "components_used": [{"id": "operator", "type": "manual"}],
                "context": {"left": data.get('left'), "right": data.get('right')}
            })
        
        # ГОЛОСОВОЙ ЗАПРОС ОТ ОПЕРАТОРА
        elif msg_type == 'human_query':
            text = data.get('text', '')
            if text:
                logger.info(f"👤 Запрос от оператора: {text}")
                self.dialog.add_turn(text, "", intent="query", source="operator")
                await self.process_with_llm(text)
    
    async def _handle_debug_prompt(self, data: Dict):
        """
        ОБРАБОТКА ОТЛАДОЧНОГО ПРОМПТА ОТ ДАШБОРДА
        
        Ответ не влияет на управление и не сохраняется в боевую историю.
        """
        prompt = data.get('prompt', '')
        if not prompt:
            return
        
        logger.info(f"🐛 Debug prompt: {prompt[:100]}...")
        
        # Получаем текущий контекст (сенсоры, диалог)
        sensor_data = await self.collect_sensor_data()
        context = self.context_builder.build_context(
            dialog_context=self.dialog.get_context_dict(),
            sensor_data=sensor_data,
            active_reflexes=[],
            current_intent=self.dialog.get_primary_intent() or "unknown"
        )
        situation = self.context_builder.format_for_llm(context)
        
        # Формируем полный промпт с контекстом
        full_prompt = f"СИТУАЦИЯ:\n{situation}\n\nВОПРОС ОТ РАЗРАБОТЧИКА:\n{prompt}\n\nОтветь подробно, на человеческом языке, без JSON."
        
        # Отправляем в LLM
        response = await self.llm.generate([{"role": "user", "content": full_prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Отправляем ответ обратно в дашборд
        await self.ws.send({
            "type": "debug_response",
            "prompt": prompt,
            "response": response_text
        })
        
        logger.info(f"🐛 Debug response: {response_text[:100]}...")
    
    async def handle_reflex_notification(self, data: Dict):
        """ОБРАБОТКА УВЕДОМЛЕНИЯ О РЕФЛЕКСЕ ОТ TINYML"""
        reflex = data.get('reflex', data.get('data', {}))
        reflex_type = reflex.get('type', 'unknown')
        distance = reflex.get('distance_cm', 0)
        action_taken = reflex.get('action', 'stop')
        
        logger.warning(f"🚨 РЕФЛЕКС: {reflex_type} на {distance}см → {action_taken}")
        
        self.episodic_memory.add_reflex(
            reflex_type=reflex_type,
            distance_cm=distance,
            action_taken=action_taken,
            context={"raw": reflex, "timestamp": time.time(), "source": "tinyml"}
        )
        
        self.dialog.add_turn(
            "",
            f"[TinyML] {reflex_type} на {distance}см, {action_taken}",
            intent="reflex",
            source="tinyml"
        )
        
        self.last_reflex_time = time.time()
        
        if self.decision_memory.has_active_task():
            logger.warning("🛑 Рефлекс прервал активную задачу")
            self.decision_memory.fail_task(f"рефлекс {reflex_type}")
    
    async def handle_execution_result(self, data: Dict):
        """ОБРАБОТКА РЕЗУЛЬТАТА ВЫПОЛНЕНИЯ КОМАНДЫ"""
        command_id = data.get('in_response_to')
        result = data.get('data', {})
        
        if command_id:
            self.decision_memory.update_execution_result(command_id, result)
            
            if result.get('executed', False) and self.decision_memory.has_active_task():
                self.decision_memory.advance_step(result)
                await self._execute_next_step()
            
            self.feedback_learner.add_feedback({
                "task_type": "execution",
                "success": result.get('executed', False),
                "duration": result.get('execution_time', 0),
                "components_used": [{"id": "tinyml", "type": "executor"}],
                "context": {"command_id": command_id}
            })
    
    # ================================================================
    # ОБРАБОТКА РЕЧИ
    # ================================================================
    
    async def on_speech_recognized(self, text: str, wake: bool = False,
                                    emergency_stop: bool = False,
                                    interactive: bool = False):
        """ВЫЗЫВАЕТСЯ ПРИ КАЖДОЙ РАСПОЗНАННОЙ ФРАЗЕ"""
        logger.info(f"👤 Человек: {text} (wake={wake}, emergency={emergency_stop})")
        
        self.last_command_time = time.time()
        
        # 1. АВАРИЙНАЯ ОСТАНОВКА
        if emergency_stop:
            self.stats["emergency_stops"] += 1
            logger.warning("🚨 АВАРИЙНАЯ ОСТАНОВКА")
            await self.ws.send(RobotProtocolV5.create_message(
                source="agent", capability="motor.stop", source_type="agent"
            ))
            if self.decision_memory.has_active_task():
                self.decision_memory.cancel_task()
            await self.tts.speak("Останавливаюсь")
            return
        
        # 2. Если ожидаем ответ (интерактивный режим)
        if self._speech_future and not self._speech_future.done():
            self._speech_future.set_result(text)
            return
        
        # 3. ПОХВАЛА
        if self.strategy_learner and self.strategy_learner.check_praise(text):
            self.stats["praise_received"] += 1
            logger.info(f"🎉 ПОХВАЛА!")
            if self._last_action_id:
                await self.strategy_learner.evaluator.evaluate(
                    self._last_action_id, human_praise=True
                )
                await self.tts.speak("Спасибо! Я стараюсь.")
            else:
                await self.tts.speak("Спасибо!")
            return
        
        self.dialog.add_turn(text, "", intent="query")
        
        if self.vlm_scanner:
            self.vlm_scanner.on_command()
        
        # 4. УПРАВЛЕНИЕ ИНТЕРАКТИВНЫМ РЕЖИМОМ
        text_lower = text.lower()
        
        if "включи интерактивный режим" in text_lower or "включи интерактив" in text_lower:
            self.interactive_mode = True
            if self.listener:
                self.listener.set_interactive_mode(True)
            await self.tts.speak("Интерактивный режим включён")
            logger.info("🔛 Интерактивный режим ВКЛЮЧЁН")
            return
        
        if "выключи интерактивный режим" in text_lower or "выключи интерактив" in text_lower:
            self.interactive_mode = False
            if self.listener:
                self.listener.set_interactive_mode(False)
            await self.tts.speak("Интерактивный режим выключён")
            logger.info("🔚 Интерактивный режим ВЫКЛЮЧЁН")
            return
        
        if wake:
            self.stats["wake_activations"] += 1
        
        await self.process_with_llm(text)
    
    # ================================================================
    # СБОР СЕНСОРНЫХ ДАННЫХ
    # ================================================================
    
    async def collect_sensor_data(self) -> List[Dict[str, Any]]:
        """СОБИРАЕТ ДАННЫЕ СО ВСЕХ СЕНСОРОВ"""
        raw_messages = await self.sensor_collector.collect_sensor_data()
        
        if self.vlm_scanner:
            vlm_result = self.vlm_scanner.get_latest()
            if vlm_result:
                raw_messages.append(vlm_result)
                self.stats["vlm_scans"] += 1
        
        enriched_messages = []
        for msg in raw_messages:
            source = msg.get('source_type', 'unknown')
            data = msg.get('data', {})
            timestamp = msg.get('timestamp', time.time())
            priority = msg.get('priority', 5)
            confidence = msg.get('confidence', 1.0)
            
            enriched = self.weight_calculator.process_with_meta(
                source_type=source,
                data=data,
                timestamp=timestamp,
                priority=priority,
                confidence=confidence,
                latency=msg.get('latency', None)
            )
            
            weight = enriched.get('_meta', {}).get('weight', 0.5)
            meta = enriched.get('_meta', {})
            self.sensor_memory.update(source, enriched, weight, meta)
            enriched_messages.append(enriched)
        
        return enriched_messages
    
    async def _wait_for_speech(self) -> Optional[str]:
        """ОЖИДАЕТ РЕЧЬ ЧЕЛОВЕКА"""
        self._speech_future = asyncio.Future()
        try:
            return await self._speech_future
        finally:
            self._speech_future = None
    
    # ================================================================
    # ВЫПОЛНЕНИЕ СТРАТЕГИЙ
    # ================================================================
    
    async def _execute_strategy(self, strategy, user_input: str) -> bool:
        """ВЫПОЛНЯЕТ СТРАТЕГИЮ (ЭВОЛЮЦИОННОЕ ОБУЧЕНИЕ)"""
        try:
            logger.info(f"🔧 Стратегия: {strategy.name} (score={strategy.get_current_score()})")
            namespace = {
                "self": self,
                "user_input": user_input,
                "asyncio": asyncio,
                "logger": logger,
                "time": time
            }
            exec(strategy.code, namespace)
            if "execute" in namespace:
                result = await namespace["execute"](self, user_input=user_input)
                return result if isinstance(result, bool) else True
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка стратегии: {e}")
            return False
    
    # ================================================================
    # ГЛАВНЫЙ МЕТОД — ОБРАБОТКА ЧЕРЕЗ LLM
    # ================================================================
    
    async def process_with_llm(self, text: str, is_task_step: bool = False):
        """
        ГЛАВНЫЙ МЕТОД — ОТПРАВЛЯЕТ ЗАПРОС В LLM И ВЫПОЛНЯЕТ КОМАНДУ
        
        Если агент заморожен (frozen=True) — не выполняет действия.
        """
        # ПРОВЕРКА ЗАМОРОЗКИ
        if self.frozen:
            logger.debug("🧊 Агент заморожен, пропускаю вызов LLM")
            await self.tts.speak("Я на паузе. Разморозьте меня через дашборд.")
            return
        
        self.stats["llm_calls"] += 1
        start_time = time.time()
        
        # 1. Собираем сенсоры
        sensor_data = await self.collect_sensor_data()
        
        # 2. Активные рефлексы (только для логирования)
        active_reflexes = []
        recent_episodes = self.episodic_memory.recall(
            episode_type="reflex", limit=3, max_age=30.0
        )
        for episode in recent_episodes:
            if episode.context.get("source") == "tinyml":
                active_reflexes.append({
                    "type": episode.context.get("reflex_type", "unknown"),
                    "distance_cm": episode.context.get("distance_cm", 0),
                    "action_taken": episode.context.get("action_taken", "stop"),
                })
        
        # 3. Текущее намерение
        if is_task_step and self.decision_memory.has_active_task():
            current_intent = text
        else:
            current_intent = self.dialog.get_primary_intent() or "unknown"
        
        # 4. Строим контекст
        context = self.context_builder.build_context(
            dialog_context=self.dialog.get_context_dict(),
            sensor_data=sensor_data,
            active_reflexes=active_reflexes,
            current_intent=current_intent
        )
        
        # 5. ПРОВЕРКА СТРАТЕГИИ
        strategy = None
        if self.strategy_learner:
            strategy = self.strategy_learner.select_strategy(current_intent)
        
        if strategy and strategy.get_current_score() > 70:
            self.stats["strategy_hits"] += 1
            success = await self._execute_strategy(strategy, text)
            if self.strategy_learner:
                self.strategy_learner.update_score(strategy, success)
            if success:
                self.stats["strategy_success"] += 1
                await self.tts.speak("Выполнено")
                if is_task_step and self.decision_memory.has_active_task():
                    self.decision_memory.advance_step({"success": True})
                    await self._execute_next_step()
            else:
                await self.tts.speak("Не получилось, пробую иначе")
                strategy = None
        
        # 6. ПРОВЕРКА КЭША
        if not strategy or strategy.get_current_score() <= 70:
            cached_decision = self.decision_memory.find_similar(context)
            
            if cached_decision and cached_decision.get_current_weight() > 0.3:
                self.stats["cache_hits"] += 1
                await self.execute_decision(cached_decision.command, cached=True)
            else:
                self.stats["cache_misses"] += 1
                prompt = self.context_builder.format_for_llm(context)
                
                # ИНТЕРАКТИВНЫЙ ХИНТ
                if self.interactive_mode and not is_task_step and not self.decision_memory.has_active_task():
                    interactive_hint = """
ИНТЕРАКТИВНЫЙ РЕЖИМ:
НЕ используй move_forward, move_backward, turn_left, turn_right без плана.
Вместо этого используй:
- compose_plan — если хочешь выполнить сложное действие (подъехать, осмотреть, запомнить)
- speak_to_human — если хочешь инициировать разговор с человеком
- ask_human — если хочешь задать вопрос человеку
- remember_object / find_object / search_by_text — если хочешь работать с памятью

Ты можешь:
- общаться с человеком (speak_to_human, ask_human)
- запоминать объекты (remember_object)
- искать объекты (find_object, search_by_text)
- составлять планы (compose_plan)

Если хочешь двигаться — сначала составь план через compose_plan.
Делай что хочешь. Прояви инициативу.
"""
                    sensor_message = f"СИТУАЦИЯ ({time.strftime('%H:%M:%S')}):\n{prompt}\n{interactive_hint}"
                else:
                    sensor_message = f"СИТУАЦИЯ ({time.strftime('%H:%M:%S')}):\n{prompt}"
                
                self.conversation.append({"role": "user", "content": text})
                self.conversation.append({"role": "user", "content": sensor_message})
                
                if len(self.conversation) > 20:
                    self.conversation = self.conversation[-20:]
                
                # ВЫЗОВ LLM
                response = await self.llm.generate(self.conversation)
                
                self.conversation.append({"role": "assistant", "content": response.content})
                
                # ОТПРАВЛЯЕМ TRACE В ДАШБОРД
                await self._send_llm_trace(prompt, response.content, context)
                
                # ПАРСИМ ОТВЕТ
                if hasattr(response, 'action') and response.action:
                    action = response.action
                    action_name = action.get("action")
                    params = action.get("params", {})
                    reasoning = action.get("reasoning", "")
                    
                    # ИСПРАВЛЕНИЕ: используем полные имена напрямую, без конвертации
                    decision = {
                        "action": action_name,
                        "params": params,
                        "reasoning": reasoning
                    }
                    
                    self.decision_memory.add_decision(
                        command=decision,
                        context=context,
                        reasoning=reasoning
                    )
                    
                    await self.execute_decision(decision, reasoning)
                    
                    if is_task_step and self.decision_memory.has_active_task():
                        self.decision_memory.advance_step({"success": True})
                        await self._execute_next_step()
                    
                elif hasattr(response, 'text') and response.text:
                    await self.tts.speak(response.text)
                    if self.dialog.turns:
                        self.dialog.turns[-1]["agent"] = response.text
                else:
                    logger.warning(f"Неизвестный формат ответа")
                    await self.tts.speak("Не понял")
        
        elapsed = time.time() - start_time
        logger.debug(f"⏱️ Обработка: {elapsed:.2f}с")
    
    async def _send_llm_trace(self, prompt: str, response: str, context: Dict):
        """ОТПРАВЛЯЕТ ТРАССУ ВЫЗОВА LLM В ДАШБОРД"""
        if not self.ws or not self.connected:
            return
        
        try:
            await self.ws.send({
                "type": "llm_trace",
                "data": {
                    "timestamp": time.time(),
                    "prompt": prompt,
                    "response": response,
                    "context": {
                        "sensors": context.get("sensors", [])[:5],
                        "intent": context.get("current_intent")
                    }
                }
            })
        except Exception as e:
            logger.debug(f"Не удалось отправить llm_trace: {e}")
    
    # ================================================================
    # ВЫПОЛНЕНИЕ РЕШЕНИЙ
    # ================================================================
    
    async def execute_decision(self, decision: Dict, reasoning: str = "", cached: bool = False):
        """ВЫПОЛНЯЕТ РЕШЕНИЕ LLM"""
        action_type = decision.get('action')
        action_id = f"{action_type}_{int(time.time() * 1000)}"
        
        current_intent = self.dialog.get_primary_intent() or "unknown"
        current_strategy = None
        if self.strategy_learner:
            current_strategy = self.strategy_learner.select_strategy(current_intent)
        
        strategy_id = current_strategy.id if current_strategy else None
        task_type = current_intent
        
        self._last_action_id = action_id
        self._last_strategy_id = strategy_id
        self._last_task_type = task_type
        
        if self.strategy_learner:
            self.strategy_learner.register_action_for_evaluation(
                action_id=action_id,
                intent=reasoning or action_type,
                strategy_id=strategy_id,
                task_type=task_type,
                context={
                    "current_intent": current_intent,
                    "sensors": self.context_builder.get_last_sensors_summary()
                }
            )
        
        for tool in self.tools:
            if tool.name == action_type:
                logger.info(f"🔧 Вызов: {action_type} (cached={cached})")
                try:
                    result = await tool.forward(**decision.get('params', {}))
                    if result and isinstance(result, str):
                        await self.tts.speak(result)
                    if self.dialog.turns:
                        self.dialog.turns[-1]["agent"] = f"[{action_type}]"
                    self.feedback_learner.add_feedback({
                        "task_type": "command",
                        "success": True,
                        "duration": tool.latency,
                        "components_used": [{"id": tool.name, "type": "tool"}],
                        "context": {"command": action_type, "cached": cached}
                    })
                    asyncio.create_task(self._schedule_evaluation(action_id, delay=EVALUATION_DELAY))
                except Exception as e:
                    logger.error(f"❌ Ошибка {action_type}: {e}")
                    if self.strategy_learner:
                        await self.strategy_learner.evaluator.evaluate(action_id, forced=True)
                break
    
    async def _schedule_evaluation(self, action_id: str, delay: float = 2.0):
        """ОТЛОЖЕННАЯ ОЦЕНКА ДЕЙСТВИЯ"""
        await asyncio.sleep(delay)
        if self.strategy_learner:
            await self.strategy_learner.evaluator.evaluation_queue.put(action_id)
    
    # ================================================================
    # ФОНОВЫЕ ЗАДАЧИ
    # ================================================================
    
    async def _stats_logger(self):
        """ФОНОВЫЙ ЛОГГЕР СТАТИСТИКИ"""
        while self.running:
            await asyncio.sleep(60)
            uptime = time.time() - self.stats["start_time"]
            cache_stats = self.decision_memory.get_stats()
            sensor_stats = self.sensor_memory.get_stats()
            
            logger.info(f"📊 Статистика за {uptime:.0f}с:")
            logger.info(f"   Сообщений: {self.stats['messages_received']}")
            logger.info(f"   Рефлексов: {self.stats['reflexes_received']}")
            logger.info(f"   LLM вызовов: {self.stats['llm_calls']}")
            logger.info(f"   Кеш: hit rate {cache_stats['hit_rate']:.1%}")
            logger.info(f"   Сенсоры: активных={sensor_stats['active_sources']}")
            logger.info(f"   Интерактивный режим: {'ВКЛ' if self.interactive_mode else 'ВЫКЛ'}")
            logger.info(f"   Заморожен: {'ДА' if self.frozen else 'НЕТ'}")
            logger.info(f"   Похвал: {self.stats['praise_received']}")
            logger.info(f"   Аварийных стопов: {self.stats['emergency_stops']}")
    
    async def _sensor_memory_cleanup(self):
        """ФОНОВАЯ ОЧИСТКА СЕНСОРНОЙ ПАМЯТИ"""
        while self.running:
            await asyncio.sleep(30)
            self.sensor_memory._cleanup_old()
    
    async def _idle_learning_loop(self):
        """ОБУЧЕНИЕ В ПРОСТОЕ И АВТОНОМНОЕ ЦЕЛЕПОЛАГАНИЕ"""
        while self.running:
            idle = time.time() - self.last_command_time
            
            # Обучение стратегий
            if 20 < idle < 25:
                logger.info(f"📚 Простой {idle:.0f}с, обучение...")
                if self.strategy_learner:
                    await self.strategy_learner.learn_in_idle(idle)
                await asyncio.sleep(30)
            
            # Автономное целеполагание в интерактивном режиме
            elif (self.interactive_mode and
                  not self.frozen and
                  not self.decision_memory.has_active_task() and
                  not self._conversation_active):
                
                # РАНДОМАЙЗЕР: от 40 до 180 секунд
                auto_task_time = random.randint(40, 180)
                
                if idle > auto_task_time:
                    logger.info(f"🎯 Простой {idle:.0f}с, генерирую задачу...")
                    task = await self._generate_self_task()
                    if task and task.get("steps"):
                        self.decision_memory.set_task(
                            steps=task["steps"],
                            reasoning=task.get("reasoning", ""),
                            task_name=task.get("task_name", "автономная задача")
                        )
                        logger.info(f"📋 Новая задача: {task.get('task_name')}")
                        await self._execute_next_step()
                    await asyncio.sleep(AUTO_TASK_COOLDOWN)
            
            await asyncio.sleep(5)
    
    async def _generate_self_task(self) -> Optional[Dict]:
        """ГЕНЕРИРУЕТ АВТОНОМНУЮ ЗАДАЧУ"""
        if not self.strategy_learner:
            return None
        
        sensor_summaries = self.sensor_memory.get_summaries()
        sensor_text = ", ".join([f"{k}: {v['summary'][:100]}" for k, v in sensor_summaries.items()])
        
        explored = [ep.description for ep in self.episodic_memory.recall(
            episode_type="observation", limit=5, min_weight=0.3
        )]
        
        interesting = [ep.context.get("object", "") for ep in self.episodic_memory.recall(
            tags=["interesting"], limit=5
        )]
        
        recent = [ep.description for ep in self.episodic_memory.recall(
            limit=3, min_weight=0.5, max_age=300
        )]
        
        available_tools = [t.name for t in self.tools]
        
        return await self.strategy_learner.generate_self_task(
            sensor_summary=sensor_text,
            explored_areas=explored,
            interesting_objects=interesting,
            recent_events=recent,
            available_tools=available_tools,
            role="автономный агент",
            self_description="Я могу двигаться, наблюдать, говорить, искать в интернете и взаимодействовать"
        )
    
    async def _execute_next_step(self):
        """ВЫПОЛНЯЕТ СЛЕДУЮЩИЙ ШАГ ЗАДАЧИ"""
        current_step = self.decision_memory.get_current_step()
        if not current_step:
            return
        
        logger.info(f"🚶 Выполняю шаг: {current_step}")
        await self.process_with_llm(current_step, is_task_step=True)
    
    async def execute_route_async(self, commands: List[Dict]):
        """ВЫПОЛНЯЕТ МАРШРУТ"""
        logger.info(f"🚶 Маршрут из {len(commands)} команд")
        for i, cmd in enumerate(commands):
            if not self.running or self.frozen:
                break
            action = cmd.get('action')
            params = cmd.get('params', {})
            for tool in self.tools:
                if tool.name == action:
                    try:
                        await tool.forward(**params)
                        await asyncio.sleep(0.3)
                    except Exception as e:
                        logger.error(f"❌ {action}: {e}")
                        await self.tts.speak("Ошибка маршрута")
                        return
                    break
        logger.info("✅ Маршрут выполнен")
        await self.tts.speak("Маршрут выполнен")
    
    # ================================================================
    # ЗАВЕРШЕНИЕ РАБОТЫ
    # ================================================================
    
    async def shutdown(self):
        """КОРРЕКТНО ЗАВЕРШАЕТ РАБОТУ РОБОТА"""
        logger.info("🛑 Завершение...")
        self.running = False
        
        if self.listener:
            self.listener.stop()
        if self.vlm_scanner:
            await self.vlm_scanner.stop()
        if self.strategy_learner:
            await self.strategy_learner.stop_evaluator()
        if self.llm:
            await self.llm.close()
        if self.yandex_client:
            await self.yandex_client.close()
        if self.ws:
            await self.ws.close()
        
        self.feedback_learner.save()
        if self.strategy_learner:
            self.strategy_learner.save()
        if self.vision_memory:
            self.vision_memory.save()
        if self.route_memory:
            self.route_memory.save()
        if self.episodic_memory:
            self.episodic_memory.save()
        
        self.decision_memory.save_to_file(DECISIONS_FILE)
        self.sensor_memory.save_to_file(SENSOR_MEMORY_FILE)
        
        uptime = time.time() - self.stats["start_time"]
        cache_stats = self.decision_memory.get_stats()
        sensor_stats = self.sensor_memory.get_stats()
        
        logger.info("=" * 50)
        logger.info("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
        logger.info(f"⏱️  Время работы: {uptime:.0f}с")
        logger.info(f"📨 Сообщений: {self.stats['messages_received']}")
        logger.info(f"🚨 Рефлексов: {self.stats['reflexes_received']}")
        logger.info(f"🧠 LLM вызовов: {self.stats['llm_calls']}")
        logger.info(f"⚡ Кеш: hit rate {cache_stats['hit_rate']:.1%}")
        logger.info(f"📡 Сенсоры: активных={sensor_stats['active_sources']}")
        logger.info(f"💬 Интерактивный режим: {'ВКЛ' if self.interactive_mode else 'ВЫКЛ'}")
        logger.info(f"🎉 Похвал: {self.stats['praise_received']}")
        logger.info(f"🛑 Аварийных стопов: {self.stats['emergency_stops']}")
        
        if self.strategy_learner:
            strategy_stats = self.strategy_learner.get_stats()
            logger.info(f"📚 Стратегий: {strategy_stats.get('total_strategies', 0)}")
            logger.info(f"🎯 Автономных задач: {strategy_stats.get('self_tasks_generated', 0)}")
        
        if self.vlm_scanner:
            logger.info(f"📸 VLM сканов: {self.vlm_scanner.get_stats().get('scans_completed', 0)}")
        
        logger.info("=" * 50)
        logger.info("👋 Агент остановлен")


# ================================================================
# ТОЧКА ВХОДА
# ================================================================

async def main():
    agent = RobotAgentV5()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())