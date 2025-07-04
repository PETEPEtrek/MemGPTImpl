#!/usr/bin/env python3
"""
FIFO очередь диалогов с суммаризацией и архивированием

Каждый блок содержит запрос пользователя и ответ агента.
При переполнении происходит суммаризация старых блоков и архивирование в FAISS.
"""

import json
import uuid
import torch
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import os
import sys

# Добавляем путь к родительской директории для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.tool_system import SharedModelManager


@dataclass
class DialogBlock:
    """Блок диалога - запрос пользователя + ответ агента"""
    id: str
    user_message: str
    assistant_response: str
    timestamp: str
    tokens_estimate: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogBlock':
        return cls(**data)
    
    def get_full_text(self) -> str:
        """Получить полный текст диалога для промпта"""
        return f"Пользователь: {self.user_message}\nАссистент: {self.assistant_response}"
    
    def get_summary_text(self) -> str:
        """Получить краткий текст для суммаризации"""
        return f"[{self.timestamp[:10]}] Пользователь спросил о {self.user_message[:50]}..., ассистент ответил: {self.assistant_response[:100]}..."


@dataclass 
class SummaryBlock:
    """Суммаризованный блок диалогов"""
    id: str
    summary_content: str
    original_blocks_count: int
    timestamp_range: str  # "2024-01-01 to 2024-01-15"
    tokens_estimate: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummaryBlock':
        return cls(**data)
    
    def get_full_text(self) -> str:
        """Получить текст суммаризации для промпта"""
        return f"[СУММАРИЗАЦИЯ {self.original_blocks_count} диалогов {self.timestamp_range}]: {self.summary_content}"


class DialogFIFO:
    """FIFO очередь диалогов с суммаризацией и архивированием"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-0.6B",
                 max_prompt_tokens: int = 3000,
                 storage_file: str = "dialog_fifo.json"):
        
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self.storage_file = storage_file
        
        # Очередь: summary_block (если есть) + dialog_blocks
        self.summary_block: Optional[SummaryBlock] = None
        self.dialog_blocks: List[DialogBlock] = []
        
        # Подключаемся к shared модели
        print(f"🤖 DialogFIFO подключается к shared модели {model_name}...")
        model_components = SharedModelManager.get_model(model_name)
        self.tokenizer = model_components['tokenizer']
        self.model = model_components['model']
        self.device = model_components['device']
        
        # Загружаем сохраненные данные
        #self.load_data()
        
        print("✅ DialogFIFO готов")
    
    def _estimate_tokens(self, text: str) -> int:
        """Оценка количества токенов в тексте"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _get_current_tokens(self) -> int:
        """Получить текущее количество токенов в очереди"""
        total = 0
        
        if self.summary_block:
            total += self.summary_block.tokens_estimate
        
        for block in self.dialog_blocks:
            total += block.tokens_estimate
        
        return total
    
    def _generate_summary(self, blocks: List[DialogBlock]) -> str:
        """Генерация суммаризации для списка блоков"""
        if not blocks:
            return ""
        
        # Создаем текст для суммаризации
        dialogs_text = "\n\n".join([block.get_summary_text() for block in blocks])
        
        summary_prompt = f"""Суммаризируй следующие диалоги в 2-3 предложения, сохранив ключевую информацию:

{dialogs_text}

Краткая суммаризация:"""
        
        try:
            inputs = self.tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            summary = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return summary
            
        except Exception as e:
            print(f"⚠️ Ошибка суммаризации: {e}")
            # Fallback суммаризация
            return f"Обсуждались темы: {', '.join([block.user_message[:30] for block in blocks[:3]])}..."
    
    def _archive_blocks_to_faiss(self, blocks: List[DialogBlock]) -> bool:
        """Архивирование блоков в FAISS базу знаний"""
        try:
            # Импортируем KnowledgeBase
            from tools.knowledge.knowledge_base import KnowledgeBase
            
            # Создаем или загружаем базу знаний
            kb = KnowledgeBase(self.model_name)
            
            # Архивируем каждый блок как отдельный факт
            archived_count = 0
            for block in blocks:
                fact_content = f"Диалог {block.timestamp[:10]}: Пользователь: {block.user_message} | Ответ: {block.assistant_response}"
                
                result = kb.add_fact(content=fact_content)
                
                if "✅" in result:
                    archived_count += 1
            
            print(f"📦 Архивировано {archived_count} диалогов в FAISS")
            return archived_count > 0
            
        except Exception as e:
            print(f"⚠️ Ошибка архивирования в FAISS: {e}")
            return False
    
    def _perform_overflow_management(self):
        """Управление переполнением очереди"""
        if not self.dialog_blocks:
            return
        
        # Вычисляем количество блоков для суммаризации (50%)
        blocks_count = len(self.dialog_blocks)
        summarize_count = max(1, blocks_count // 2)
        
        # Берем старейшие блоки для суммаризации
        blocks_to_summarize = self.dialog_blocks[:summarize_count]
        blocks_to_keep = self.dialog_blocks[summarize_count:]
        
        print(f"🔄 Суммаризация {summarize_count} из {blocks_count} диалогов...")
        
        # Архивируем в FAISS
        #self._archive_blocks_to_faiss(blocks_to_summarize)
        
        # Генерируем суммаризацию
        new_summary = self._generate_summary(blocks_to_summarize)
        
        if not new_summary:
            print("⚠️ Не удалось создать суммаризацию")
            return
        
        # Определяем временной диапазон
        timestamps = [block.timestamp for block in blocks_to_summarize]
        time_range = f"{timestamps[0][:10]} to {timestamps[-1][:10]}"
        
        if self.summary_block is None:
            # Создаем новый суммаризованный блок
            self.summary_block = SummaryBlock(
                id=str(uuid.uuid4()),
                summary_content=new_summary,
                original_blocks_count=summarize_count,
                timestamp_range=time_range,
                tokens_estimate=self._estimate_tokens(new_summary)
            )
            print(f"✅ Создан новый блок суммаризации ({summarize_count} диалогов)")
        else:
            # Добавляем к существующему блоку суммаризации
            extended_summary = f"{self.summary_block.summary_content} | {new_summary}"
            
            self.summary_block.summary_content = extended_summary
            self.summary_block.original_blocks_count += summarize_count
            self.summary_block.timestamp_range = f"{self.summary_block.timestamp_range.split(' to ')[0]} to {time_range.split(' to ')[1]}"
            self.summary_block.tokens_estimate = self._estimate_tokens(extended_summary)
            
            print(f"✅ Расширен блок суммаризации (теперь {self.summary_block.original_blocks_count} диалогов)")
        
        # Обновляем очередь
        self.dialog_blocks = blocks_to_keep
        
        # Сохраняем изменения
        self.save_data()
    
    def add_dialog(self, user_message: str, assistant_response: str) -> str:
        """Добавление нового диалога в очередь"""
        
        # Создаем новый блок диалога
        dialog_block = DialogBlock(
            id=str(uuid.uuid4()),
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now().isoformat(),
            tokens_estimate=self._estimate_tokens(f"{user_message} {assistant_response}")
        )
        
        # Добавляем в очередь
        self.dialog_blocks.append(dialog_block)
        
        # Проверяем переполнение
        if self._get_current_tokens() > self.max_prompt_tokens:
            self._perform_overflow_management()
        
        # Сохраняем
        #self.save_data()
        
        return f"✅ Диалог добавлен в FIFO очередь (ID: {dialog_block.id[:8]})"
    
    def get_dialog_context(self) -> str:
        """Получение контекста диалогов для промпта"""
        if not self.summary_block and not self.dialog_blocks:
            return ""
        
        context_parts = ["=== ИСТОРИЯ ДИАЛОГОВ ==="]
        
        # Добавляем суммаризацию (если есть)
        if self.summary_block:
            context_parts.append(self.summary_block.get_full_text())
            context_parts.append("")  # Пустая строка для разделения
        
        # Добавляем текущие диалоги
        for block in self.dialog_blocks:
            context_parts.append(block.get_full_text())
            context_parts.append("")  # Пустая строка между диалогами
        
        context_parts.append("=== КОНЕЦ ИСТОРИИ ===")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> str:
        """Статистика FIFO очереди"""
        current_tokens = self._get_current_tokens()
        
        stats = [
            "📊 Статистика FIFO очереди диалогов:",
            f"  💬 Активных диалогов: {len(self.dialog_blocks)}",
            f"  🔤 Токенов в очереди: {current_tokens}/{self.max_prompt_tokens}",
            f"  📈 Заполненность: {(current_tokens/self.max_prompt_tokens)*100:.1f}%"
        ]
        
        if self.summary_block:
            stats.extend([
                f"  📋 Суммаризация: {self.summary_block.original_blocks_count} диалогов ({self.summary_block.timestamp_range})",
                f"  🔤 Токенов в суммаризации: {self.summary_block.tokens_estimate}"
            ])
        else:
            stats.append("  📋 Суммаризация: отсутствует")
        
        return "\n".join(stats)
    
    def clear_fifo(self) -> str:
        """Очистка FIFO очереди"""
        self.summary_block = None
        self.dialog_blocks = []
        self.save_data()
        return "🧹 FIFO очередь диалогов очищена"
    
    def save_data(self):
        """Сохранение данных в файл"""
        data = {
            "summary_block": self.summary_block.to_dict() if self.summary_block else None,
            "dialog_blocks": [block.to_dict() for block in self.dialog_blocks],
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения FIFO: {e}")
    
    def load_data(self):
        """Загрузка данных из файла"""
        if not os.path.exists(self.storage_file):
            return
        
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Загружаем суммаризацию
            if data.get("summary_block"):
                self.summary_block = SummaryBlock.from_dict(data["summary_block"])
            
            # Загружаем диалоги
            self.dialog_blocks = [
                DialogBlock.from_dict(block_data) 
                for block_data in data.get("dialog_blocks", [])
            ]
            
            print(f"📂 Загружена FIFO очередь: {len(self.dialog_blocks)} диалогов")
            if self.summary_block:
                print(f"   📋 + суммаризация {self.summary_block.original_blocks_count} архивных диалогов")
            
        except Exception as e:
            print(f"⚠️ Ошибка загрузки FIFO: {e}")
    
    def export_fifo(self) -> str:
        """Экспорт FIFO в JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary_block": self.summary_block.to_dict() if self.summary_block else None,
            "total_dialogs": len(self.dialog_blocks),
            "current_tokens": self._get_current_tokens(),
            "dialog_blocks": [block.to_dict() for block in self.dialog_blocks]
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)


def create_dialog_fifo_tools(dialog_fifo: DialogFIFO) -> Dict[str, Dict[str, Any]]:
    """Создание инструментов для управления FIFO очередью диалогов"""
    
    tools = {}
    
    # 1. Получение статистики FIFO
    tools["fifo_stats"] = {
        "name": "fifo_stats",
        "description": "Показать статистику FIFO очереди диалогов",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: dialog_fifo.get_stats()
    }
    
    # 2. Получение контекста диалогов
    tools["fifo_context"] = {
        "name": "fifo_context", 
        "description": "Получить контекст истории диалогов для промпта",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: dialog_fifo.get_dialog_context()
    }
    
    # 3. Очистка FIFO
    tools["fifo_clear"] = {
        "name": "fifo_clear",
        "description": "Очистить FIFO очередь диалогов",
        "parameters": {
            "type": "object", 
            "properties": {},
            "required": []
        },
        "function": lambda: dialog_fifo.clear_fifo()
    }
    
    # 4. Экспорт FIFO
    tools["fifo_export"] = {
        "name": "fifo_export",
        "description": "Экспортировать FIFO очередь в JSON формат",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: dialog_fifo.export_fifo()
    }
    
    return tools


# Тестирование
def main():
    """Тестирование DialogFIFO"""
    print("🧪 Тестирование DialogFIFO")
    print("=" * 50)
    
    # Создаем FIFO с малым лимитом для демонстрации
    fifo = DialogFIFO(max_prompt_tokens=500)
    
    # Тестовые диалоги
    test_dialogs = [
        ("Привет! Как дела?", "Привет! У меня все отлично, спасибо! А как у тебя дела?"),
        ("Расскажи про Python", "Python - это высокоуровневый язык программирования, созданный Гвидо ван Россумом. Он известен своей простотой и читаемостью кода."),
        ("Что такое машинное обучение?", "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам учиться на данных без явного программирования."),
        ("Как работают нейронные сети?", "Нейронные сети состоят из узлов (нейронов), соединенных весами. Они обучаются, корректируя веса на основе ошибок предсказаний."),
        ("Объясни алгоритм градиентного спуска", "Градиентный спуск - это оптимизационный алгоритм, который итеративно движется в направлении, противоположном градиенту функции потерь."),
    ]
    
    # Добавляем диалоги
    for i, (user_msg, assistant_msg) in enumerate(test_dialogs, 1):
        print(f"\n{i}. Добавление диалога...")
        result = fifo.add_dialog(user_msg, assistant_msg)
        print(f"   {result}")
        print(f"   {fifo.get_stats()}")
    
    # Показываем финальный контекст
    print(f"\n📋 Финальный контекст диалогов:")
    print(fifo.get_dialog_context())
    
    print(f"\n✅ Тестирование завершено!")


if __name__ == "__main__":
    main() 