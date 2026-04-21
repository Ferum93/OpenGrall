#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ vision/vlm_client.py — КЛИЕНТ ДЛЯ ВИЗУАЛЬНОЙ ЯЗЫКОВОЙ МОДЕЛИ (VLM)           ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «глаза» робота. VLM анализирует кадры с камеры и возвращает            ║
║   структурированное описание сцены: что за помещение, какие объекты,         ║
║   свободен ли путь.                                                          ║
║                                                                              ║
║ КАК ЭТО РАБОТАЕТ В АРХИТЕКТУРЕ:                                              ║
║                                                                              ║
║   VLM работает АСИНХРОННО в фоне (через VLMScanner).                         ║
║   Она НЕ ждёт запросов от LLM. Она постоянно сканирует и обновляет           ║
║   SensorMemory.                                                              ║
║                                                                              ║
║   Когда агенту нужно принять решение, он просто берёт ПОСЛЕДНИЙ СКАН         ║
║   из SensorMemory. Возраст и вес скана сами подскажут, насколько ему         ║
║   можно доверять.                                                            ║
║                                                                              ║
║   НИКТО НИКОГО НЕ ЖДЁТ.                                                      ║
║                                                                              ║
║ РЕКОМЕНДУЕМЫЕ МОДЕЛИ (ДЛЯ СЛАБОГО ЖЕЛЕЗА):                                   ║
║                                                                              ║
║   • Moondream 1.8B — самая лёгкая, работает на смартфоне                     ║
║   • Qwen2-VL-2B — баланс качества и скорости                                 ║
║   • Florence-2 — ещё легче, но менее универсальна                            ║
║                                                                              ║
║   Все модели загружаются в 4-битном режиме (load_in_4bit=True).              ║
║   Это позволяет запускать их даже на устройствах с 4 ГБ RAM.                 ║
║                                                                              ║
║ КАК ИСПОЛЬЗОВАТЬ:                                                            ║
║                                                                              ║
║   from vision.vlm_client import VLMClient                                    ║
║                                                                              ║
║   vlm = VLMClient(model_name="vikhr/vikhr-vl-2b")  # или moondream           ║
║   description = await vlm.analyze_scene(image, prompt)                       ║
║                                                                              ║
║   # ВАЖНО: этот метод вызывается ТОЛЬКО из VLMScanner.                       ║
║   # Агент никогда не вызывает VLM напрямую.                                  ║
║                                                                              ║
║ ТРЕБОВАНИЯ:                                                                  ║
║   • torch, transformers, pillow                                              ║
║   • bitsandbytes (для 4-битного квантования)                                 ║
║   • flash-attn (опционально, ускорение)                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import numpy as np
import logging
import json
import re
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import cv2

logger = logging.getLogger(__name__)


class VLMClient:
    """
    КЛИЕНТ ДЛЯ ВИЗУАЛЬНОЙ ЯЗЫКОВОЙ МОДЕЛИ
    
    Загружает модель один раз при старте и держит в памяти.
    Используется ТОЛЬКО через VLMScanner, никогда не вызывается агентом напрямую.
    """
    
    def __init__(self, 
                 model_name: str = "vikhr/vikhr-vl-2b",
                 device: str = None,
                 load_in_4bit: bool = True,
                 max_new_tokens: int = 50):
        """
        Args:
            model_name: имя модели на Hugging Face
            device: 'cuda', 'cpu', или None (автоопределение)
            load_in_4bit: использовать 4-битное квантование (экономия RAM)
            max_new_tokens: максимальная длина ответа
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Автоопределение устройства
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"🔄 Загружаю VLM: {model_name} на {self.device}...")
        
        # Параметры загрузки
        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        if load_in_4bit and self.device == "cuda":
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            logger.info("   📦 4-битное квантование включено")
        
        # Загружаем модель
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Для CPU явно перемещаем модель
            if self.device == "cpu" and not load_in_4bit:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"✅ VLM загружена ({self.device}, {self._get_model_size():.1f} ГБ RAM)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки VLM: {e}")
            raise
    
    def _get_model_size(self) -> float:
        """Приблизительная оценка размера модели в ГБ"""
        try:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            return param_size / (1024 ** 3)
        except:
            return 0.0
    
    async def analyze_scene(self, image: np.ndarray, prompt: str) -> str:
        """
        АНАЛИЗИРУЕТ СЦЕНУ И ВОЗВРАЩАЕТ ТЕКСТОВОЕ ОПИСАНИЕ
        
        ВАЖНО: Этот метод вызывается ТОЛЬКО из VLMScanner.
        Агент никогда не вызывает VLM напрямую — он берёт готовые сканы из SensorMemory.
        
        Args:
            image: кадр в формате numpy (BGR, как из cv2)
            prompt: текстовый запрос (что нужно описать)
        
        Returns:
            str: описание сцены (желательно в JSON-формате)
        """
        return await self._generate_response(image, prompt)
    
    async def analyze_with_focus(self, image: np.ndarray, target_object: str) -> dict:
        """
        ПРЕЦИЗИОННЫЙ РЕЖИМ VLM — ФОКУСИРОВКА НА КОНКРЕТНОМ ОБЪЕКТЕ
        
        Используется для точного позиционирования при:
        - Захвате объекта манипулятором
        - Режиме "follow me" (отслеживание человека)
        - Возврате на док-станцию (поиск QR-кода или визуальной метки)
        - Инспекции конкретного предмета
        
        ВАЖНО: Этот метод не используется VLMScanner'ом. Он вызывается
        напрямую инструментами, когда нужно точное позиционирование.
        
        Args:
            image: кадр в формате numpy (BGR, как из cv2)
            target_object: описание целевого объекта (например, "красный мяч", "QR-код", "человек")
        
        Returns:
            dict: {
                "found": bool,
                "object": str или None,
                "distance_cm": float или None,
                "offset_x_deg": float или None,  # смещение от центра кадра по горизонтали
                "offset_y_deg": float или None,  # смещение от центра кадра по вертикали
                "orientation": str или None,     # "left", "right", "front", "back", "unknown"
                "confidence": float (0-1),
                "raw_response": str
            }
        """
        
        # Формируем промпт для прецизионного режима
        prompt = f"""В кадре есть объект: {target_object}.
Сфокусируйся ТОЛЬКО на нём. Игнорируй все остальные объекты.

Ответь СТРОГО в JSON-формате:
{{
  "object": "название найденного объекта или null",
  "distance_cm": число (примерное расстояние в сантиметрах) или null,
  "offset_x_deg": число (смещение от центра кадра по горизонтали в градусах, отрицательное - влево, положительное - вправо) или null,
  "offset_y_deg": число (смещение от центра кадра по вертикали в градусах, отрицательное - вниз, положительное - вверх) или null,
  "orientation": "left/right/front/back/unknown" или null,
  "confidence": число от 0 до 1
}}

Если объект НЕ найден, верни:
{{"object": null, "confidence": 0}}

Не пиши НИЧЕГО кроме JSON."""

        try:
            response_text = await self._generate_response(image, prompt)
            
            # Парсим JSON из ответа
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return {
                        "found": data.get("object") is not None and data.get("confidence", 0) > 0.3,
                        "object": data.get("object"),
                        "distance_cm": data.get("distance_cm"),
                        "offset_x_deg": data.get("offset_x_deg"),
                        "offset_y_deg": data.get("offset_y_deg"),
                        "orientation": data.get("orientation"),
                        "confidence": data.get("confidence", 0.0),
                        "raw_response": response_text
                    }
                except json.JSONDecodeError:
                    logger.warning(f"VLM вернула невалидный JSON: {response_text[:100]}...")
            
            # Fallback: не удалось распарсить JSON
            return {
                "found": False,
                "object": None,
                "distance_cm": None,
                "offset_x_deg": None,
                "offset_y_deg": None,
                "orientation": None,
                "confidence": 0.0,
                "raw_response": response_text
            }
            
        except Exception as e:
            logger.error(f"Ошибка в analyze_with_focus: {e}")
            return {
                "found": False,
                "object": None,
                "distance_cm": None,
                "offset_x_deg": None,
                "offset_y_deg": None,
                "orientation": None,
                "confidence": 0.0,
                "raw_response": str(e)
            }
    
    async def _generate_response(self, image: np.ndarray, prompt: str) -> str:
        """
        ВНУТРЕННИЙ МЕТОД — ГЕНЕРАЦИЯ ОТВЕТА VLM
        
        Используется как analyze_scene, так и analyze_with_focus.
        """
        # Конвертируем BGR (OpenCV) в RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Формируем сообщение для модели
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Подготовка входных данных
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Перемещаем на нужное устройство
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Генерация
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=0.0,
            )
        
        # Декодируем только новые токены
        output_text = self.processor.decode(
            generated_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        logger.debug(f"VLM ответ: {output_text[:100]}...")
        return output_text.strip()
    
    def get_stats(self) -> dict:
        """Возвращает информацию о модели"""
        return {
            "model": self.model_name,
            "device": self.device,
            "size_gb": self._get_model_size(),
            "max_tokens": self.max_new_tokens
        }


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ VLM CLIENT")
        print("="*60 + "\n")
        
        # Проверяем, есть ли CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA не доступна, используется CPU (будет медленно)")
        
        print("\n🔄 Загружаем модель...")
        
        try:
            # Пробуем загрузить Moondream (самая лёгкая)
            vlm = VLMClient(model_name="vikhr/vikhr-vl-2b")
            
            stats = vlm.get_stats()
            print(f"\n📊 Статистика модели:")
            print(f"   Модель: {stats['model']}")
            print(f"   Устройство: {stats['device']}")
            print(f"   Размер: {stats['size_gb']:.1f} ГБ")
            
            # Создаём синтетическое изображение для теста
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(dummy_image, (200, 150), (400, 350), (0, 0, 255), -1)
            
            print("\n📸 Анализируем тестовое изображение (обычный режим)...")
            result = await vlm.analyze_scene(
                dummy_image,
                prompt="Что на изображении? Опиши одним предложением."
            )
            print(f"   Ответ: {result}")
            
            print("\n🎯 Анализируем тестовое изображение (прецизионный режим)...")
            focus_result = await vlm.analyze_with_focus(
                dummy_image,
                target_object="красный прямоугольник"
            )
            print(f"   Найден: {focus_result['found']}")
            print(f"   Объект: {focus_result['object']}")
            print(f"   Уверенность: {focus_result['confidence']}")
            print(f"   Координаты: offset_x={focus_result['offset_x_deg']}°, offset_y={focus_result['offset_y_deg']}°")
            print(f"   Дистанция: {focus_result['distance_cm']} см")
            
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("   Возможно, нужно установить зависимости:")
            print("   pip install transformers torch pillow bitsandbytes")
        
        print("\n" + "="*60)
        print("✅ Демонстрация завершена")
        print("="*60 + "\n")
    
    asyncio.run(demo())