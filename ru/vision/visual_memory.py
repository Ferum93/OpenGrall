#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ vision/visual_memory.py - ВИЗУАЛЬНАЯ ПАМЯТЬ ОБЪЕКТОВ                         ║
║                                                                              ║
║ ЧТО ЭТО:                                                                     ║
║   Это «зрительная кора» робота. Хранит образы объектов, которые робот        ║
║   видел раньше, и позволяет находить их снова по изображению или тексту.     ║
║                                                                              ║
║ КАК РАБОТАЕТ:                                                                ║
║                                                                              ║
║   Робот видит объект → запоминает его через save_object():                   ║
║     • SIFT-дескрипторы (ключевые точки) — для точного поиска                 ║
║     • Гистограмма цвета — для быстрого сравнения                             ║
║     • Эмбеддинг (если есть SentenceTransformer) — для семантического поиска  ║
║                                                                              ║
║   Позже робот может:                                                         ║
║     • Найти объект по имени — find_object("ботинок", current_image)          ║
║     • Найти объект по описанию — search_by_text("красный стул")              ║
║                                                                              ║
║ ГИБРИДНЫЙ ПОДХОД — ТРИ МЕТОДА ПОИСКА:                                        ║
║                                                                              ║
║   1. Гистограмма (быстро, но неточно) — первая линия.                        ║
║   2. SIFT (точно, но медленнее) — если гистограмма не дала уверенности.      ║
║   3. Эмбеддинги (семантика) — для поиска по тексту.                          ║
║                                                                              ║
║   Это позволяет роботу узнавать объекты даже при изменении ракурса           ║
║   или освещения, и находить их по описанию («найди то, что похоже на стул»). ║
║                                                                              ║
║ ИСПОЛЬЗОВАНИЕ В АГЕНТЕ:                                                      ║
║                                                                              ║
║   Инструменты, которые используют VisualMemory:                              ║
║     • RememberObjectTool — «запомни этот объект как "ботинок"»               ║
║     • FindObjectTool — «найди "ботинок" в текущем кадре»                     ║
║     • SearchByTextTool — «найди все объекты, похожие на "стул"»              ║
║                                                                              ║
║ ТРЕБОВАНИЯ:                                                                  ║
║   • opencv-python (SIFT, гистограммы)                                        ║
║   • faiss (опционально, для быстрого поиска эмбеддингов)                     ║
║   • sentence-transformers (опционально, для текстового поиска)               ║
║                                                                              ║
║ КАК НАСТРОИТЬ ПОД СЕБЯ:                                                      ║
║                                                                              ║
║   1. Порог гистограммы — изменить 0.7 в find_object()                        ║
║   2. Порог SIFT — изменить 10 в len(good_matches) > 10                       ║
║   3. Порог эмбеддинга — изменить 0.8 в find_object()                         ║
║   4. Добавить свой детектор — переопределить _compute_embedding()            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import cv2
import numpy as np
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ================================================================
# ПРОВЕРКА ЗАВИСИМОСТЕЙ
# ================================================================

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS не доступен. Поиск по тексту будет ограничен.")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ SentenceTransformers не доступен. Семантический поиск отключен.")


class VisualMemory:
    """
    ВИЗУАЛЬНАЯ ПАМЯТЬ — ХРАНИТ ОБРАЗЫ ОБЪЕКТОВ И УМЕЕТ ИХ НАХОДИТЬ
    
    Использует гибридный подход: SIFT для точного поиска,
    гистограммы для быстрого, эмбеддинги для семантики.
    """
    
    def __init__(self, storage_path: str):
        """
        Args:
            storage_path: путь к JSON-файлу для сохранения памяти
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        
        # Хранилище объектов: name → {histogram, descriptors, embedding, ...}
        self.objects = {}
        
        # FAISS индекс для быстрого поиска эмбеддингов
        self.index = None
        self.names = []  # имена объектов в том же порядке, что и в индексе
        self.embedding_dim = 512
        
        # Для асинхронного захвата кадров (используется в RememberObjectTool)
        self.frame_future: Optional[asyncio.Future] = None
        
        # Пытаемся загрузить модель для эмбеддингов
        self.use_embeddings = False
        if TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
                logger.info("✅ Текстовая модель загружена (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.warning(f"❌ Не удалось загрузить текстовую модель: {e}")
        
        self.load()
    
    # ================================================================
    # ВЫЧИСЛЕНИЕ ПРИЗНАКОВ
    # ================================================================
    
    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        ВЫЧИСЛЯЕТ ЦВЕТОВУЮ ГИСТОГРАММУ ИЗОБРАЖЕНИЯ
        
        Используется для быстрого сравнения. Устойчива к небольшим изменениям.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 8 бинов на каждый цветовой канал = 8×8×8 = 512 измерений
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], 
                           [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def _compute_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        ВЫЧИСЛЯЕТ ЭМБЕДДИНГ ИЗОБРАЖЕНИЯ
        
        Используется для семантического поиска. Эмбеддинги близких по смыслу
        объектов (например, разные стулья) будут похожи.
        """
        try:
            # Приводим к единому размеру
            resized = cv2.resize(image, (224, 224))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Гистограмма + статистика как простой эмбеддинг
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
            mean = np.mean(gray) / 255.0
            std = np.std(gray) / 255.0
            
            features = np.concatenate([hist, [mean, std]])
            
            # Приводим к фиксированной размерности
            if len(features) < self.embedding_dim:
                features = np.pad(features, (0, self.embedding_dim - len(features)))
            else:
                features = features[:self.embedding_dim]
            
            return features.astype(np.float32)
        except Exception as e:
            logger.error(f"Ошибка вычисления эмбеддинга: {e}")
            return None
    
    # ================================================================
    # СОХРАНЕНИЕ ОБЪЕКТА
    # ================================================================
    
    def save_object(self, name: str, image: np.ndarray):
        """
        ЗАПОМИНАЕТ ОБЪЕКТ ПОД УКАЗАННЫМ ИМЕНЕМ
        
        Вызывается через RememberObjectTool, когда робот видит что-то важное.
        
        Args:
            name: имя объекта (например, "мой_ботинок")
            image: изображение объекта (BGR, как из cv2)
        """
        logger.info(f"💾 Запоминаю объект: {name}")
        
        # 1. Гистограмма (быстрое сравнение)
        hist = self._compute_histogram(image)
        
        # 2. SIFT-дескрипторы (точное сравнение)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        # 3. Эмбеддинг (семантический поиск)
        embedding = None
        if self.use_embeddings:
            embedding = self._compute_embedding(image)
        
        # Сохраняем всё
        self.objects[name] = {
            "histogram": hist.tolist(),
            "descriptors": descriptors.tolist() if descriptors is not None else None,
            "embedding": embedding.tolist() if embedding is not None else None,
            "timestamp": time.time(),
            "keypoints_count": len(keypoints) if keypoints else 0
        }
        
        # Добавляем в FAISS индекс для быстрого текстового поиска
        if embedding is not None and FAISS_AVAILABLE:
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embedding))
            self.index.add(np.array([embedding], dtype=np.float32))
            self.names.append(name)
        
        self.save()
        logger.info(f"✅ Объект '{name}' сохранён (ключевых точек: {self.objects[name]['keypoints_count']})")
    
    # ================================================================
    # ПОИСК ОБЪЕКТА ПО ИМЕНИ
    # ================================================================
    
    def find_object(self, name: str, current_image: np.ndarray) -> Dict[str, Any]:
        """
        ИЩЕТ ОБЪЕКТ ПО ИМЕНИ В ТЕКУЩЕМ ИЗОБРАЖЕНИИ
        
        Вызывается через FindObjectTool. Пробует три метода:
        1. Гистограмма (если похожесть > 0.7 — сразу успех)
        2. SIFT (если гистограмма не дала уверенности)
        3. Эмбеддинг (если SIFT недоступен)
        
        Returns:
            {
                "found": True/False,
                "method": "histogram" / "sift" / "embedding" / "none",
                "confidence": 0.0-1.0,
                "name": name,
                ...
            }
        """
        if name not in self.objects:
            return {"found": False, "method": "none", "confidence": 0.0}
        
        obj = self.objects[name]
        results = []
        
        # -------------------- 1. ГИСТОГРАММА --------------------
        current_hist = self._compute_histogram(current_image)
        stored_hist = np.array(obj["histogram"], dtype=np.float32)
        hist_sim = cv2.compareHist(stored_hist, current_hist, cv2.HISTCMP_CORREL)
        results.append(("histogram", hist_sim))
        
        # Если гистограмма уверена — сразу возвращаем успех
        if hist_sim > 0.7:
            return {"found": True, "method": "histogram", 
                   "confidence": hist_sim, "name": name}
        
        # -------------------- 2. SIFT --------------------
        if obj["descriptors"] and len(obj["descriptors"]) > 0:
            sift = cv2.SIFT_create()
            kp2, des2 = sift.detectAndCompute(current_image, None)
            
            if des2 is not None and len(des2) > 0:
                # FLANN матчер для быстрого поиска
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                stored_des = np.array(obj["descriptors"], dtype=np.float32)
                matches = flann.knnMatch(stored_des, des2, k=2)
                
                # Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                sift_score = len(good_matches) / max(len(kp2), 1)
                results.append(("sift", sift_score))
                
                # Если хороших совпадений много — объект найден
                if len(good_matches) > 10:
                    return {"found": True, "method": "sift", 
                           "confidence": min(1.0, sift_score * 2), 
                           "name": name, "matches": len(good_matches)}
        
        # -------------------- 3. ЭМБЕДДИНГ --------------------
        if self.use_embeddings and obj.get("embedding"):
            current_emb = self._compute_embedding(current_image)
            if current_emb is not None:
                stored_emb = np.array(obj["embedding"])
                # Косинусное сходство
                emb_sim = np.dot(stored_emb, current_emb) / (
                    np.linalg.norm(stored_emb) * np.linalg.norm(current_emb) + 1e-6
                )
                results.append(("embedding", emb_sim))
                
                if emb_sim > 0.8:
                    return {"found": True, "method": "embedding", 
                           "confidence": emb_sim, "name": name}
        
        # -------------------- НЕ НАЙДЕНО --------------------
        best_method, best_score = max(results, key=lambda x: x[1], default=("none", 0))
        return {"found": False, "method": best_method, 
                "confidence": best_score, "name": name, "all_results": results}
    
    # ================================================================
    # ПОИСК ПО ТЕКСТУ
    # ================================================================
    
    def search_by_text(self, query: str) -> List[Dict[str, Any]]:
        """
        ИЩЕТ ОБЪЕКТЫ ПО ТЕКСТОВОМУ ОПИСАНИЮ
        
        Вызывается через SearchByTextTool. Использует эмбеддинги
        для семантического поиска. Если эмбеддинги недоступны —
        fallback на поиск по имени.
        
        Args:
            query: текстовое описание ("красный стул", "ботинок")
        
        Returns:
            Список найденных объектов с confidence
        """
        # Fallback: поиск по имени
        if not self.use_embeddings or not FAISS_AVAILABLE:
            results = []
            for name in self.objects.keys():
                if query.lower() in name.lower():
                    results.append({
                        "name": name, 
                        "method": "name_match", 
                        "confidence": 0.8, 
                        "data": self.objects[name]
                    })
            return results
        
        # Семантический поиск через эмбеддинги
        query_embedding = self.text_model.encode(query)
        
        if self.index and self.index.ntotal > 0:
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k=5
            )
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.names):
                    name = self.names[idx]
                    confidence = 1.0 / (1.0 + distances[0][i])
                    results.append({
                        "name": name, 
                        "method": "text_embedding", 
                        "confidence": confidence, 
                        "data": self.objects[name]
                    })
            return results
        
        return []
    
    # ================================================================
    # УПРАВЛЕНИЕ ПАМЯТЬЮ
    # ================================================================
    
    def get_all_objects(self) -> List[str]:
        """Возвращает список всех сохранённых объектов"""
        return list(self.objects.keys())
    
    def delete_object(self, name: str):
        """Удаляет объект из памяти"""
        if name in self.objects:
            del self.objects[name]
            self._rebuild_index()
            self.save()
            logger.info(f"🗑️ Объект '{name}' удалён")
    
    def _rebuild_index(self):
        """Перестраивает FAISS индекс после удаления объекта"""
        if not FAISS_AVAILABLE:
            return
        
        self.index = None
        self.names = []
        
        for name, obj in self.objects.items():
            if obj.get("embedding"):
                if self.index is None:
                    self.index = faiss.IndexFlatL2(len(obj["embedding"]))
                self.index.add(np.array([obj["embedding"]], dtype=np.float32))
                self.names.append(name)
    
    def save(self):
        """Сохраняет память в JSON-файл"""
        data = {
            "objects": self.objects, 
            "names": self.names, 
            "version": "2.0"
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"💾 Визуальная память сохранена ({len(self.objects)} объектов)")
    
    def load(self):
        """Загружает память из JSON-файла"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.objects = data.get("objects", {})
                self.names = data.get("names", [])
                self._rebuild_index()
                logger.info(f"📂 Загружено {len(self.objects)} объектов из {self.storage_path}")
        except FileNotFoundError:
            logger.info("🆕 Создана новая визуальная память")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки визуальной памяти: {e}")


# ================================================================
# ДЕМОНСТРАЦИЯ
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ VISUAL MEMORY")
    print("="*60 + "\n")
    
    # Создаём память в временном файле
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        memory = VisualMemory(tmp.name)
    
    print("📸 Для демонстрации нужен реальный кадр с камеры.")
    print("   Здесь показана работа с заглушкой.\n")
    
    # Создаём синтетическое изображение
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(dummy_image, (200, 150), (400, 350), (0, 0, 255), -1)
    
    print("💾 Сохраняем объект 'красный_квадрат'...")
    memory.save_object("красный_квадрат", dummy_image)
    
    print("\n📋 Все объекты в памяти:")
    for name in memory.get_all_objects():
        print(f"   - {name}")
    
    print("\n🔍 Ищем 'красный_квадрат' в том же изображении...")
    result = memory.find_object("красный_квадрат", dummy_image)
    print(f"   Найден: {result['found']}")
    print(f"   Метод: {result['method']}")
    print(f"   Уверенность: {result['confidence']:.2f}")
    
    print("\n🔎 Ищем по тексту 'квадрат'...")
    text_results = memory.search_by_text("квадрат")
    if text_results:
        for r in text_results:
            print(f"   - {r['name']} (confidence: {r['confidence']:.2f})")
    else:
        print("   Ничего не найдено (эмбеддинги отключены)")
    
    print("\n" + "="*60)
    print("✅ Демонстрация завершена.")
    print("="*60 + "\n")
