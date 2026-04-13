#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ memory/route_memory.py - ПАМЯТЬ МАРШРУТОВ                                     ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «пространственная память» робота. Хранит последовательности команд,    ║
║   которые образуют маршрут от точки А до точки Б.                            ║
║                                                                              ║
║ КАК РАБОТАЕТ:                                                                ║
║                                                                              ║
║   1. Человек (или LLM) говорит: «запомни маршрут "на кухню"»                 ║
║   2. Агент начинает записывать все команды движения в список                 ║
║   3. Человек говорит: «стоп, маршрут закончен»                               ║
║   4. Маршрут сохраняется в JSON-файл                                         ║
║   5. Позже можно сказать: «поезжай по маршруту "на кухню"»                   ║
║   6. Робот последовательно выполняет все команды из маршрута                 ║
║                                                                              ║
║ ФОРМАТ МАРШРУТА:                                                             ║
║                                                                              ║
║   {                                                                          ║
║     "на кухню": {                                                            ║
║       "commands": [                                                          ║
║         {"action": "move_forward", "params": {"speed": 300, "duration": 2.0}},║
║         {"action": "turn_left", "params": {"speed": 400, "duration": 0.8}},  ║
║         {"action": "move_forward", "params": {"speed": 250, "duration": 1.5}}║
║       ],                                                                     ║
║       "created": 1712345678.123,                                             ║
║       "steps": 3                                                             ║
║     }                                                                        ║
║   }                                                                          ║
║                                                                              ║
║ ИСПОЛЬЗОВАНИЕ В АГЕНТЕ:                                                      ║
║                                                                              ║
║   Инструменты, которые используют RouteMemory:                               ║
║     • RememberRouteTool — начать запись маршрута                             ║
║     • EndRouteTool — закончить запись и сохранить                            ║
║     • ExecuteRouteTool — выполнить сохранённый маршрут                       ║
║                                                                              ║
║ ОГРАНИЧЕНИЯ:                                                                 ║
║                                                                              ║
║   • Маршруты «слепые» — робот просто повторяет команды без учёта             ║
║     изменившейся обстановки. Для безопасного выполнения нужен                ║
║     TinyML с правом вето (он остановит робота при препятствии).              ║
║                                                                              ║
║   • В будущем можно добавить «якоря» — привязку команд к семантическим       ║
║     ориентирам («повернуть у красной двери»).                                ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Изменить формат команд — добавить новые поля в save_route()             ║
║   2. Добавить проверку безопасности — в execute_route_async()                ║
║   3. Добавить редактирование маршрутов — новый метод update_route()          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RouteMemory:
    """
    ПАМЯТЬ МАРШРУТОВ — ХРАНИТ И ВОСПРОИЗВОДИТ ПУТИ
    
    Простая, но важная часть автономности. Позволяет роботу
    «запомнить дорогу» и повторить её позже.
    """
    
    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: путь к JSON-файлу для сохранения маршрутов
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        
        # Хранилище маршрутов: name → {commands, created, steps}
        self.routes: Dict[str, Dict[str, Any]] = {}
        
        self.load()
        logger.info(f"✅ RouteMemory инициализирована (файл: {storage_path})")
    
    # ================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ================================================================
    
    def save_route(self, name: str, commands: List[Dict]):
        """
        СОХРАНЯЕТ МАРШРУТ ПОД УКАЗАННЫМ ИМЕНЕМ
        
        Вызывается через EndRouteTool, когда запись маршрута завершена.
        
        Args:
            name: имя маршрута (например, "на кухню", "в гараж")
            commands: список команд в формате [{action, params}, ...]
        """
        self.routes[name] = {
            "commands": commands,
            "created": time.time(),
            "steps": len(commands)
        }
        self.save()
        logger.info(f"✅ Маршрут '{name}' сохранён ({len(commands)} команд)")
    
    def get_route(self, name: str) -> Optional[List[Dict]]:
        """
        ВОЗВРАЩАЕТ КОМАНДЫ МАРШРУТА ПО ИМЕНИ
        
        Вызывается через ExecuteRouteTool для воспроизведения.
        
        Args:
            name: имя маршрута
        
        Returns:
            Список команд или None, если маршрут не найден
        """
        route = self.routes.get(name)
        if route:
            return route["commands"]
        return None
    
    def list_routes(self) -> List[str]:
        """
        ВОЗВРАЩАЕТ СПИСОК ВСЕХ СОХРАНЁННЫХ МАРШРУТОВ
        
        Используется для отладки или если LLM хочет узнать,
        какие маршруты уже есть в памяти.
        """
        return list(self.routes.keys())
    
    def delete_route(self, name: str):
        """
        УДАЛЯЕТ МАРШРУТ ИЗ ПАМЯТИ
        
        Args:
            name: имя маршрута
        """
        if name in self.routes:
            del self.routes[name]
            self.save()
            logger.info(f"🗑️ Маршрут '{name}' удалён")
    
    def get_route_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        ВОЗВРАЩАЕТ ИНФОРМАЦИЮ О МАРШРУТЕ (БЕЗ КОМАНД)
        
        Используется, чтобы узнать, когда маршрут был создан
        и сколько в нём шагов, не загружая все команды.
        """
        route = self.routes.get(name)
        if route:
            return {
                "name": name,
                "created": route["created"],
                "steps": route["steps"]
            }
        return None
    
    # ================================================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ================================================================
    
    def save(self):
        """Сохраняет все маршруты в JSON-файл"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.routes, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 Память маршрутов сохранена ({len(self.routes)} маршрутов)")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения маршрутов: {e}")
    
    def load(self):
        """Загружает маршруты из JSON-файла"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.routes = json.load(f)
            logger.info(f"📂 Загружено {len(self.routes)} маршрутов из {self.storage_path}")
        except FileNotFoundError:
            logger.info("🆕 Создана новая память маршрутов")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки маршрутов: {e}")
            self.routes = {}
    
    def clear(self):
        """Очищает всю память маршрутов"""
        self.routes.clear()
        self.save()
        logger.info("🧹 Память маршрутов очищена")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ROUTE MEMORY")
    print("="*60 + "\n")
    
    # Создаём память во временном файле
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        memory = RouteMemory(tmp.name)
    
    # Сохраняем тестовый маршрут
    print("📝 Сохраняем маршрут 'на кухню'...")
    test_route = [
        {"action": "move_forward", "params": {"speed": 300, "duration": 2.0}},
        {"action": "turn_left", "params": {"speed": 400, "duration": 0.8}},
        {"action": "move_forward", "params": {"speed": 250, "duration": 1.5}},
        {"action": "stop", "params": {}}
    ]
    memory.save_route("на кухню", test_route)
    
    # Сохраняем ещё один
    print("📝 Сохраняем маршрут 'в гараж'...")
    garage_route = [
        {"action": "move_backward", "params": {"speed": 200, "duration": 1.0}},
        {"action": "turn_right", "params": {"speed": 400, "duration": 0.8}},
        {"action": "move_forward", "params": {"speed": 350, "duration": 3.0}}
    ]
    memory.save_route("в гараж", garage_route)
    
    # Показываем все маршруты
    print(f"\n📋 Все маршруты: {memory.list_routes()}")
    
    # Информация о маршруте
    info = memory.get_route_info("на кухню")
    print(f"\nℹ️ Информация о маршруте 'на кухню':")
    print(f"   Создан: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['created']))}")
    print(f"   Шагов: {info['steps']}")
    
    # Загружаем и показываем команды
    print(f"\n🚶 Команды маршрута 'на кухню':")
    commands = memory.get_route("на кухню")
    for i, cmd in enumerate(commands):
        print(f"   {i+1}. {cmd['action']} {cmd['params']}")
    
    # Удаляем один маршрут
    print(f"\n🗑️ Удаляем маршрут 'в гараж'...")
    memory.delete_route("в гараж")
    print(f"   Остались: {memory.list_routes()}")
    
    print("\n" + "="*60)
    print("✅ Демонстрация завершена.")
    print("="*60 + "\n")
