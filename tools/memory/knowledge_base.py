#!/usr/bin/env python3
"""
Векторная база знаний с FAISS
Хранит общие факты и знания, не связанные с конкретным пользователем
"""

import os
import json
import uuid
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
from core.tool_system import SharedModelManager


@dataclass
class KnowledgeFact:
    """Факт в базе знаний"""
    id: str
    content: str
    timestamp: str  # YYYY-MM-DD HH:MM:SS
    embedding: Optional[np.ndarray] = None
    source: str = "conversation"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Не сериализуем embedding в JSON
        if 'embedding' in data:
            del data['embedding']
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeFact':
        return cls(**data)


class KnowledgeBase:
    """Векторная база знаний с FAISS"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-0.6B",
                 db_path: str = "knowledge_db",
                 embedding_dim: int = None):
        
        self.model_name = model_name
        self.db_path = db_path
        self.facts: Dict[str, KnowledgeFact] = {}
        
        print(f"🧠 Инициализация Knowledge Base с shared моделью...")
        
        # Получаем shared модель для эмбеддингов
        model_components = SharedModelManager.get_model(model_name)
        self.tokenizer = model_components['tokenizer']
        self.model = model_components['model']
        self.device = model_components['device']
        
        # Определяем размерность эмбеддинга
        if embedding_dim is None:
            # Создаем тестовый эмбеддинг для определения размерности
            test_embedding = self._create_embedding("test")
            self.embedding_dim = test_embedding.shape[0]
            print(f"📏 Автоматически определена размерность эмбеддинга: {self.embedding_dim}")
        else:
            self.embedding_dim = embedding_dim
        
        # Инициализируем FAISS индекс с правильной размерностью
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product для косинусного сходства
        
        # Загружаем существующую БД
        self._load_database()
        
        print("✅ Knowledge Base готова")
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Создание эмбеддинга для текста"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Для causal LM используем hidden states или logits
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Используем последний hidden state
                embeddings = outputs.hidden_states[-1].mean(dim=1)
            elif hasattr(outputs, 'last_hidden_state'):
                # Стандартный подход для encoder моделей
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback: используем logits как эмбеддинги
                embeddings = outputs.logits.mean(dim=1)
                
            # Нормализуем для косинусного сходства
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().astype('float32')[0]
    
    def _save_database(self):
        """Сохранение базы данных"""
        os.makedirs(self.db_path, exist_ok=True)
        
        # Сохраняем факты
        facts_data = {fact_id: fact.to_dict() for fact_id, fact in self.facts.items()}
        with open(os.path.join(self.db_path, "facts.json"), "w", encoding="utf-8") as f:
            json.dump(facts_data, f, ensure_ascii=False, indent=2)
        
        # Сохраняем FAISS индекс
        if self.index.ntotal > 0:
            faiss.write_index(self.index, os.path.join(self.db_path, "faiss.index"))
        
        # Сохраняем маппинг ID для FAISS
        fact_ids = list(self.facts.keys())
        with open(os.path.join(self.db_path, "fact_ids.pkl"), "wb") as f:
            pickle.dump(fact_ids, f)
        
        print(f"💾 База данных сохранена: {len(self.facts)} фактов")
    
    def _load_database(self):
        """Загрузка базы данных"""
        if not os.path.exists(self.db_path):
            print("📝 Создается новая база знаний")
            return
        
        try:
            # Загружаем факты
            facts_path = os.path.join(self.db_path, "facts.json")
            if os.path.exists(facts_path):
                with open(facts_path, "r", encoding="utf-8") as f:
                    facts_data = json.load(f)
                
                self.facts = {
                    fact_id: KnowledgeFact.from_dict(fact_data)
                    for fact_id, fact_data in facts_data.items()
                }
            
            # Загружаем FAISS индекс
            index_path = os.path.join(self.db_path, "faiss.index")
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            
            # Загружаем маппинг ID
            ids_path = os.path.join(self.db_path, "fact_ids.pkl")
            if os.path.exists(ids_path):
                with open(ids_path, "rb") as f:
                    self.fact_ids = pickle.load(f)
            else:
                self.fact_ids = list(self.facts.keys())
            
            print(f"📚 Загружена база знаний: {len(self.facts)} фактов")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки БД: {e}. Создается новая.")
            self.facts = {}
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.fact_ids = []
    
    def add_fact(self, content: str) -> str:
        """Добавление факта в базу знаний"""
        
        # Проверяем на дубликаты
        existing_facts = self.search_facts(content, limit=1, threshold=0.95)
        if existing_facts and existing_facts[0][1] > 0.95:
            return f"❌ Похожий факт уже существует: {existing_facts[0][0].content[:50]}..."
        
        # Создаем новый факт
        fact = KnowledgeFact(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        # Создаем эмбеддинг
        fact.embedding = self._create_embedding(content)
        
        # Добавляем в словарь
        self.facts[fact.id] = fact
        
        # Добавляем в FAISS индекс
        if fact.embedding is not None:
            self.index.add(fact.embedding.reshape(1, -1))
        
        # Обновляем маппинг ID
        if not hasattr(self, 'fact_ids'):
            self.fact_ids = []
        self.fact_ids.append(fact.id)
        
        # Сохраняем БД
        #self._save_database()
        
        return f"✅ Добавлен факт: {content[:50]}... (ID: {fact.id[:8]})"
    
    def search_facts(self, 
                    query: str, 
                    limit: int = 5, 
                    threshold: float = 0.7) -> List[Tuple[KnowledgeFact, float]]:
        """Поиск фактов по семантическому сходству"""
        
        if self.index.ntotal == 0:
            return []
        
        # Создаем эмбеддинг для поискового запроса
        query_embedding = self._create_embedding(query)
        
        # Поиск в FAISS
        scores, indices = self.index.search(query_embedding.reshape(1, -1), min(limit * 2, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.fact_ids):
                continue
                
            fact_id = self.fact_ids[idx]
            if fact_id not in self.facts:
                continue
                
            fact = self.facts[fact_id]
            
            # Фильтрация по порогу
            if score >= threshold:
                results.append((fact, float(score)))
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def delete_fact(self, fact_id: str) -> str:
        """Удаление факта"""
        if fact_id not in self.facts:
            return f"❌ Факт с ID {fact_id} не найден"
        
        fact = self.facts[fact_id]
        del self.facts[fact_id]
        
        # Для удаления из FAISS нужно пересоздать индекс
        self._rebuild_index()
        
        return f"✅ Удален факт: {fact.content[:50]}..."
    
    def _rebuild_index(self):
        """Пересоздание FAISS индекса"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.fact_ids = []
        
        for fact_id, fact in self.facts.items():
            if fact.embedding is not None:
                self.index.add(fact.embedding.reshape(1, -1))
                self.fact_ids.append(fact_id)
        
        #self._save_database()
    
    def get_knowledge_context(self, query: str, max_facts: int = 3) -> str:
        """Получение контекста знаний для промпта"""
        relevant_facts = self.search_facts(query, limit=max_facts, threshold=0.6)
        
        if not relevant_facts:
            return ""
        
        context_parts = ["=== РЕЛЕВАНТНЫЕ ЗНАНИЯ ==="]
        for fact, score in relevant_facts:
            context_parts.append(
                f"{fact.content} (релевантность: {score:.2f})"
            )
        context_parts.append("=== КОНЕЦ ЗНАНИЙ ===")
        
        return "\n".join(context_parts)
    
    def auto_add_facts(self, text: str) -> str:
        """Автоматическое добавление фактов из текста"""
        facts = self.extract_facts_from_text(text)
        added_count = 0
        
        for fact in facts:
            try:
                result = self.add_fact(fact)
                if "✅" in result:
                    added_count += 1
            except Exception as e:
                print(f"⚠️ Ошибка добавления факта: {e}")
        
        if added_count > 0:
            return f"🧠 Автоматически добавлено {added_count} фактов в базу знаний"
        else:
            return ""
    
    def extract_facts_from_text(self, text: str) -> List[str]:
        """Извлечение потенциальных фактов из текста"""
        # Простая эвристика для извлечения фактов
        # В реальной системе можно использовать NER или более сложные методы
        
        potential_facts = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Слишком короткие
                continue
            
            # Ищем предложения с датами, местами, определениями
            indicators = [
                'это', 'является', 'означает', 'находится', 'произошло',
                'был создан', 'изобретен', 'основан', 'расположен'
            ]
            
            # Проверяем на персональную информацию (исключаем)
            personal_indicators = [
                'я', 'мне', 'мой', 'моя', 'мое', 'мои', 'меня', 'мной'
            ]
            
            sentence_lower = sentence.lower()
            
            # Исключаем персональную информацию
            if any(indicator in sentence_lower for indicator in personal_indicators):
                continue
            
            # Ищем факты
            if (any(indicator in sentence_lower for indicator in indicators) or
                any(char.isdigit() for char in sentence) or  # Содержит числа/даты
                len(sentence) > 30):  # Достаточно информативное предложение
                
                potential_facts.append(sentence)
        
        return potential_facts[:3]  # Максимум 3 факта за раз
    
    def get_stats(self) -> str:
        """Статистика базы знаний"""
        if not self.facts:
            return "📊 База знаний пуста"
        
        stats = [
            f"📊 Статистика базы знаний:",
            f"  📚 Всего фактов: {len(self.facts)}",
            f"  💾 FAISS индекс: {self.index.ntotal} векторов",
            f"  🕒 Последнее обновление: {max(fact.timestamp for fact in self.facts.values())[:19]}"
        ]
        
        return "\n".join(stats)


def create_knowledge_tools(knowledge_base: KnowledgeBase) -> Dict[str, Dict[str, Any]]:
    """Создание инструментов для работы с базой знаний"""
    
    tools = {}
    
    # 1. Добавление факта
    tools["knowledge_add_fact"] = {
        "name": "knowledge_add_fact",
        "description": "Добавляет общий факт в базу знаний (не связанный с конкретным пользователем)",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Факт для добавления в базу знаний"
                }
            },
            "required": ["content"]
        },
        "function": knowledge_base.add_fact
    }
    
    # 2. Поиск фактов
    tools["knowledge_search"] = {
        "name": "knowledge_search",
        "description": "Ищет релевантные факты в базе знаний",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Максимальное количество результатов"
                }
            },
            "required": ["query"]
        },
        "function": lambda query, limit=5: _search_facts_formatted(
            knowledge_base, query, limit
        )
    }
    
    # 3. Статистика
    tools["knowledge_stats"] = {
        "name": "knowledge_stats",
        "description": "Получает статистику базы знаний",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": knowledge_base.get_stats
    }
    
    return tools


def _search_facts_formatted(knowledge_base: KnowledgeBase, 
                          query: str, 
                          limit: int = 5) -> str:
    """Форматированный поиск фактов"""
    
    results = knowledge_base.search_facts(query, limit)
    
    if not results:
        return f"❌ Фактов по запросу '{query}' не найдено"
    
    formatted_results = [f"🔍 Найдено {len(results)} фактов по запросу '{query}':"]
    
    for i, (fact, score) in enumerate(results, 1):
        formatted_results.append(
            f"{i}. {fact.content} "
            f"(релевантность: {score:.2f}, добавлен: {fact.timestamp[:10]})"
        )
    
    return "\n".join(formatted_results)