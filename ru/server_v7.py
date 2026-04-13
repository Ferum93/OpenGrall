#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ server_v7.py — НЕРВНАЯ СИСТЕМА ГРАЛЛА                                        ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это центральный узел, через который проходят ВСЕ сигналы в системе.        ║
║   Оператор, агент, ESP32, камеры, датчики — все подключаются сюда.           ║
║                                                                              ║
║ ФИЛОСОФИЯ — ОДИН МОЗГ, МНОГО ТЕЛ:                                            ║
║                                                                              ║
║   Гралл — это не конкретный робот. Это ЛИЧНОСТЬ, которая может               ║
║   вселяться в разные тела.                                                   ║
║                                                                              ║
║   • Дома — живёт в роботе-пылесосе                                           ║
║   • В машине — подключается к ESP32 автомобиля                               ║
║   • На работе — управляет манипулятором                                      ║
║   • В смартфоне — просто голосовой ассистент                                 ║
║                                                                              ║
║   Агент (мозг) — один. Сервер — его нервная система. Тела — сменные.         ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ:                                                            ║
║                                                                              ║
║   1. Устройства регистрируются с указанием роли:                             ║
║      {"type": "register", "role": "operator"}                                ║
║                                                                              ║
║   2. Сервер ведёт реестр подключённых устройств.                             ║
║                                                                              ║
║   3. Сообщения маршрутизируются по ролям:                                    ║
║      - operator → esp (команды движения)                                     ║
║      - esp → operator, agent (телеметрия, рефлексы)                          ║
║      - streamer → operator (WebRTC видео)                                    ║
║      - agent → esp (стратегические команды)                                  ║
║                                                                              ║
║   4. Устройства могут подключаться и отключаться в любой момент.             ║
║      Сервер продолжает работать.                                             ║
║                                                                              ║
║ РОЛИ УСТРОЙСТВ:                                                              ║
║   • operator — человек с веб-интерфейсом                                     ║
║   • streamer — источник видео (телефон робота)                               ║
║   • esp — контроллер моторов/сенсоров                                        ║
║   • agent — мозг (LLM)                                                       ║
║   • camera — дополнительная камера                                           ║
║   • sensor — дополнительный датчик                                           ║
║                                                                              ║
║ ЗАПУСК:                                                                      ║
║   python server_v7.py                                                        ║
║                                                                              ║
║   После запуска:                                                             ║
║   - HTTP сервер на порту 5001 (веб-интерфейс)                                ║
║   - WebSocket на порту 5002 (обмен данными)                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import threading
import time
import subprocess
import socket
import struct
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from websockets.server import serve
import websockets
from collections import defaultdict
from typing import Dict, Any, Optional

# ================================================================
# НАСТРОЙКИ
# ================================================================

HTTP_PORT = 5001
WS_PORT = 5002
BATTERY_CHECK_INTERVAL = 60

# ================================================================
# СОСТОЯНИЕ СИСТЕМЫ
# ================================================================

# Состояние железа (то, что могут запросить клиенты)
state = {
    'left_motor': 0,
    'right_motor': 0,
    'servo_angle': 57,
    'light': False,
    'charge_enabled': False,
    'smartphone_battery': 100,
    'smartphone_charging': False,
}

# РЕЕСТР КОМПОНЕНТОВ — СЕРДЦЕ РАСПРЕДЕЛЁННОЙ АРХИТЕКТУРЫ
# Каждое устройство регистрируется с указанием роли и capabilities
components = {
    'operator': {'connected': None, 'capabilities': ['control', 'video']},
    'streamer': {'connected': None, 'capabilities': ['video_source']},
    'esp': {'connected': None, 'capabilities': ['motors', 'servo', 'light']},
    'agent': {'connected': None, 'capabilities': ['llm', 'planning']},
    # В будущем можно добавить:
    # 'camera_back': {'connected': None, 'capabilities': ['video_source']},
    # 'arm': {'connected': None, 'capabilities': ['manipulator']},
    # 'gps': {'connected': None, 'capabilities': ['position']},
}

connections_lock = threading.Lock()
ws_loop = None


# ================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ================================================================

def get_ip():
    """Получить IP адрес сервера"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def get_battery():
    """Получить уровень батареи телефона (Termux)"""
    try:
        result = subprocess.run(['termux-battery-status'], 
                               capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get('percentage', 100), data.get('status') == 'CHARGING'
    except:
        pass
    return 75, False


def battery_monitor():
    """Фоновый мониторинг батареи телефона"""
    while True:
        try:
            level, charging = get_battery()
            state['smartphone_battery'] = level
            state['smartphone_charging'] = charging
            
            battery_msg = {
                "type": "battery",
                "level": level,
                "charging": charging,
                "timestamp": time.time()
            }
            
            with connections_lock:
                for role, info in components.items():
                    ws = info.get('connected')
                    if ws:
                        asyncio.run_coroutine_threadsafe(
                            ws.send(json.dumps(battery_msg)), ws_loop
                        )
            time.sleep(BATTERY_CHECK_INTERVAL)
        except Exception as e:
            print(f"Battery error: {e}")
            time.sleep(10)


# ================================================================
# HTTP СЕРВЕР (ОТДАЁТ ВЕБ-ИНТЕРФЕЙС)
# ================================================================

class CustomHandler(SimpleHTTPRequestHandler):
    """Обработчик статических файлов"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='.', **kwargs)

def run_http():
    """Запуск HTTP сервера"""
    server = HTTPServer(('', HTTP_PORT), CustomHandler)
    print(f"🌐 HTTP сервер на порту {HTTP_PORT}")
    server.serve_forever()


# ================================================================
# WEBSOCKET СЕРВЕР (НЕРВНАЯ СИСТЕМА)
# ================================================================

async def handle_connection(websocket):
    """Обработчик WebSocket соединения"""
    global ws_loop
    if ws_loop is None:
        ws_loop = asyncio.get_running_loop()
    
    role = None
    print(f"🔌 Новое подключение: {websocket.remote_address}")
    
    try:
        # РЕГИСТРАЦИЯ УСТРОЙСТВА
        async for message in websocket:
            data = json.loads(message)
            if data.get('type') == 'register':
                role = data.get('role')
                if role in components:
                    with connections_lock:
                        old = components[role]['connected']
                        if old:
                            await old.close()
                        components[role]['connected'] = websocket
                        components[role]['last_seen'] = time.time()
                    
                    print(f"✅ Зарегистрирован: {role}")
                    await websocket.send(json.dumps({
                        "type": "registered", 
                        "role": role
                    }))
                    
                    # Отправляем начальное состояние
                    if role == 'esp':
                        await websocket.send(json.dumps({
                            "type": "light", 
                            "state": state['light']
                        }))
                    elif role == 'operator':
                        # Отправляем текущий статус всех компонентов
                        status_msg = {
                            "type": "status",
                            "data": {
                                "esp_connected": components['esp']['connected'] is not None,
                                "agent_connected": components['agent']['connected'] is not None,
                                "streamer_connected": components['streamer']['connected'] is not None,
                                "light": state['light'],
                                "servo_angle": state['servo_angle']
                            }
                        }
                        await websocket.send(json.dumps(status_msg))
                    
                    break
                else:
                    await websocket.send(json.dumps({
                        "type": "error", 
                        "message": f"Неизвестная роль: {role}"
                    }))
                    await websocket.close()
                    return
            else:
                await websocket.send(json.dumps({
                    "type": "error", 
                    "message": "Сначала зарегистрируйтесь"
                }))
                await websocket.close()
                return
        
        # ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ СООБЩЕНИЙ
        async for message in websocket:
            try:
                data = json.loads(message)
                await route_message(role, data, websocket)
            except Exception as e:
                print(f"Ошибка обработки сообщения от {role}: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"❌ {role} отключился")
    finally:
        with connections_lock:
            if components.get(role, {}).get('connected') == websocket:
                components[role]['connected'] = None
                print(f"📤 {role} удалён из реестра")


async def route_message(role: str, data: Dict, websocket):
    """
    МАРШРУТИЗАЦИЯ СООБЩЕНИЙ — СЕРДЦЕ НЕРВНОЙ СИСТЕМЫ
    
    Определяет, кто кому может отправлять сообщения.
    """
    msg_type = data.get('type')
    
    # ========== OPERATOR → ESP / AGENT ==========
    if role == 'operator':
        if msg_type == 'motor':
            with connections_lock:
                esp = components.get('esp', {}).get('connected')
            if esp:
                await esp.send(json.dumps(data))
            # Агенту для обучения
            with connections_lock:
                agent = components.get('agent', {}).get('connected')
            if agent:
                await agent.send(json.dumps({
                    "type": "motor_command",
                    "left": data.get('left'),
                    "right": data.get('right')
                }))
        
        elif msg_type == 'servo':
            with connections_lock:
                esp = components.get('esp', {}).get('connected')
            if esp:
                await esp.send(json.dumps(data))
        
        elif msg_type == 'light':
            state['light'] = data.get('state', False)
            with connections_lock:
                esp = components.get('esp', {}).get('connected')
            if esp:
                await esp.send(json.dumps(data))
        
        elif msg_type == 'stop':
            with connections_lock:
                esp = components.get('esp', {}).get('connected')
            if esp:
                await esp.send(json.dumps(data))
        
        elif msg_type == 'webrtc':
            target = data.get('target')
            if target == 'streamer':
                with connections_lock:
                    streamer = components.get('streamer', {}).get('connected')
                if streamer:
                    await streamer.send(json.dumps({
                        "type": "webrtc",
                        "data": data.get('data')
                    }))
        
        elif msg_type == 'query':
            # Голосовой запрос к агенту
            with connections_lock:
                agent = components.get('agent', {}).get('connected')
            if agent:
                await agent.send(json.dumps({
                    "type": "human_query",
                    "text": data.get('text', '')
                }))
        
        elif msg_type == 'get_status':
            # Запрос текущего состояния
            status_msg = {
                "type": "status",
                "data": {
                    "esp_connected": components['esp']['connected'] is not None,
                    "agent_connected": components['agent']['connected'] is not None,
                    "streamer_connected": components['streamer']['connected'] is not None,
                    "light": state['light'],
                    "servo_angle": state['servo_angle']
                }
            }
            await websocket.send(json.dumps(status_msg))
    
    # ========== STREAMER → OPERATOR ==========
    elif role == 'streamer':
        if msg_type == 'webrtc':
            target = data.get('target')
            if target == 'operator':
                with connections_lock:
                    operator = components.get('operator', {}).get('connected')
                if operator:
                    await operator.send(json.dumps({
                        "type": "webrtc",
                        "data": data.get('data')
                    }))
    
    # ========== ESP → OPERATOR / AGENT ==========
    elif role == 'esp':
        if msg_type == 'telemetry':
            with connections_lock:
                for r in ['operator', 'agent']:
                    ws = components.get(r, {}).get('connected')
                    if ws:
                        await ws.send(json.dumps({
                            "type": "esp_telemetry",
                            "data": data.get('data', {})
                        }))
        
        elif msg_type == 'reflex':
            # Уведомление о рефлексе TinyML
            with connections_lock:
                agent = components.get('agent', {}).get('connected')
                if agent:
                    await agent.send(json.dumps(data))
                operator = components.get('operator', {}).get('connected')
                if operator:
                    await operator.send(json.dumps({
                        "type": "toast",
                        "message": f"🚨 Рефлекс: {data.get('reflex', {}).get('type', 'unknown')}"
                    }))
    
    # ========== AGENT → ESP / OPERATOR ==========
    elif role == 'agent':
        if msg_type == 'command':
            target = data.get('target')
            if target == 'esp':
                with connections_lock:
                    esp = components.get('esp', {}).get('connected')
                if esp:
                    await esp.send(json.dumps(data.get('command', {})))
        
        elif msg_type == 'speak':
            with connections_lock:
                operator = components.get('operator', {}).get('connected')
            if operator:
                await operator.send(json.dumps({
                    "type": "agent_message",
                    "text": data.get('text', '')
                }))


async def ws_server():
    """Запуск WebSocket сервера"""
    async with serve(handle_connection, "0.0.0.0", WS_PORT):
        print(f"🔌 WebSocket сервер на порту {WS_PORT}")
        await asyncio.Future()


def run_ws():
    """Запуск WebSocket в отдельном потоке"""
    asyncio.run(ws_server())


# ================================================================
# ЗАПУСК
# ================================================================

if __name__ == "__main__":
    print("="*50)
    print("🤖 СЕРВЕР ГРАЛЛА v7.0 — НЕРВНАЯ СИСТЕМА")
    print("="*50)
    
    ip = get_ip()
    
    # Запускаем HTTP сервер (веб-интерфейс)
    threading.Thread(target=run_http, daemon=True).start()
    
    # Запускаем WebSocket сервер (обмен данными)
    threading.Thread(target=run_ws, daemon=True).start()
    
    # Запускаем мониторинг батареи
    threading.Thread(target=battery_monitor, daemon=True).start()
    
    print(f"\n📡 HTTP: http://{ip}:{HTTP_PORT}")
    print(f"📡 WebSocket: ws://{ip}:{WS_PORT}")
    print("\n📋 Роли устройств:")
    for role, info in components.items():
        caps = ', '.join(info['capabilities'])
        print(f"   • {role} — {caps}")
    print("\n💡 Оператор может подключиться через браузер.")
    print("   Агент, ESP, Streamer — через WebSocket.")
    print("="*50)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Завершение работы")
