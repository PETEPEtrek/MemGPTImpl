#!/usr/bin/env python3
"""
Core Memory System - по аналогии с MemGPT
Два основных блока памяти: Human и Persona
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class CoreMemoryBlock:
    """Блок Core Memory"""
    name: str
    content: str
    last_updated: str
    max_length: int = 2000
    
    def update_content(self, new_content: str) -> bool:
        """Обновление содержимого блока"""
        if len(new_content) <= self.max_length:
            self.content = new_content
            self.last_updated = datetime.now().isoformat()
            return True
        return False
    
    def append_content(self, additional_content: str) -> bool:
        """Добавление содержимого к существующему"""
        new_content = f"{self.content}\n{additional_content}".strip()
        return self.update_content(new_content)
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            "name": self.name,
            "content": self.content,
            "last_updated": self.last_updated,
            "max_length": self.max_length
        }


class CoreMemory:
    """
    Система Core Memory с двумя основными блоками:
    - human: информация о пользователе
    - persona: информация о чат-боте
    """
    
    def __init__(self, storage_file: str = "core_memory.json"):
        self.storage_file = storage_file
        
        # Инициализация блоков
        self.human = CoreMemoryBlock(
            name="human",
            content="",
            last_updated=datetime.now().isoformat(),
            max_length=2000
        )
        
        self.persona = CoreMemoryBlock(
            name="persona", 
            content="Я умный ИИ-ассистент с памятью. Я помогаю пользователям, запоминаю важную информацию и веду естественные диалоги.",
            last_updated=datetime.now().isoformat(),
            max_length=1500
        )
        
        # Загружаем существующие данные
        #self.load_from_file()
    
    def add_to_human(self, information: str) -> str:
        """
        Добавить информацию о пользователе в блок human
        
        Args:
            information: Информация для добавления
            
        Returns:
            Статус операции
        """
        try:
            if self.human.append_content(information):
                #self.save_to_file()
                return f"✅ Информация добавлена в human блок: {information[:50]}..."
            else:
                return f"❌ Не удалось добавить - превышен лимит {self.human.max_length} символов"
        except Exception as e:
            return f"❌ Ошибка добавления в human блок: {e}"
    
    def add_to_persona(self, information: str) -> str:
        """
        Добавить информацию о боте в блок persona
        
        Args:
            information: Информация для добавления
            
        Returns:
            Статус операции
        """
        try:
            if self.persona.append_content(information):
                #self.save_to_file()
                return f"✅ Информация добавлена в persona блок: {information[:50]}..."
            else:
                return f"❌ Не удалось добавить - превышен лимит {self.persona.max_length} символов"
        except Exception as e:
            return f"❌ Ошибка добавления в persona блок: {e}"
    
    def edit_human(self, new_content: str) -> str:
        """
        Полностью переписать блок human
        
        Args:
            new_content: Новое содержимое блока
            
        Returns:
            Статус операции
        """
        try:
            if self.human.update_content(new_content):
                #self.save_to_file()
                return f"✅ Human блок полностью обновлен ({len(new_content)} символов)"
            else:
                return f"❌ Не удалось обновить - превышен лимит {self.human.max_length} символов"
        except Exception as e:
            return f"❌ Ошибка обновления human блока: {e}"
    
    def edit_persona(self, new_content: str) -> str:
        """
        Полностью переписать блок persona
        
        Args:
            new_content: Новое содержимое блока
            
        Returns:
            Статус операции
        """
        try:
            if self.persona.update_content(new_content):
                #self.save_to_file()
                return f"✅ Persona блок полностью обновлен ({len(new_content)} символов)"
            else:
                return f"❌ Не удалось обновить - превышен лимит {self.persona.max_length} символов"
        except Exception as e:
            return f"❌ Ошибка обновления persona блока: {e}"
    
    def get_human_info(self) -> str:
        """Получить информацию из human блока"""
        return self.human.content if self.human.content else "Нет информации о пользователе"
    
    def get_persona_info(self) -> str:
        """Получить информацию из persona блока"""
        return self.persona.content
    
    def get_core_memory_context(self) -> str:
        """
        Получить контекст Core Memory для вставки в промпт
        
        Returns:
            Форматированный контекст для промпта
        """
        context_parts = ["=== CORE MEMORY ==="]
        
        # Human блок
        if self.human.content.strip():
            context_parts.append(f"Human: {self.human.content}")
        else:
            context_parts.append("Human: <нет информации о пользователе>")
        
        # Persona блок  
        context_parts.append(f"Persona: {self.persona.content}")
        
        context_parts.append("=== END CORE MEMORY ===")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> str:
        """Получить статистику Core Memory"""
        human_len = len(self.human.content)
        persona_len = len(self.persona.content)
        
        stats = [
            "📊 Core Memory статистика:",
            f"  👤 Human: {human_len}/{self.human.max_length} символов",
            f"  🤖 Persona: {persona_len}/{self.persona.max_length} символов",
            f"  📅 Human обновлен: {self.human.last_updated[:19]}",
            f"  📅 Persona обновлен: {self.persona.last_updated[:19]}"
        ]
        
        return "\n".join(stats)
    
    def clear_human(self) -> str:
        """Очистить human блок"""
        try:
            self.human.update_content("")
            #self.save_to_file()
            return "✅ Human блок очищен"
        except Exception as e:
            return f"❌ Ошибка очистки human блока: {e}"
    
    def clear_persona(self) -> str:
        """Очистить persona блок"""
        try:
            self.persona.update_content("")
            #self.save_to_file()
            return "✅ Persona блок очищен"
        except Exception as e:
            return f"❌ Ошибка очистки persona блока: {e}"
    
    def save_to_file(self) -> bool:
        """Сохранить Core Memory в файл"""
        try:
            data = {
                "human": self.human.to_dict(),
                "persona": self.persona.to_dict(),
                "saved_at": datetime.now().isoformat()
            }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения Core Memory: {e}")
            return False
    
    def load_from_file(self) -> bool:
        """Загрузить Core Memory из файла"""
        try:
            if not os.path.exists(self.storage_file):
                return False
            
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Восстанавливаем human блок
            if "human" in data:
                human_data = data["human"]
                self.human = CoreMemoryBlock(
                    name=human_data.get("name", "human"),
                    content=human_data.get("content", ""),
                    last_updated=human_data.get("last_updated", datetime.now().isoformat()),
                    max_length=human_data.get("max_length", 2000)
                )
            
            # Восстанавливаем persona блок
            if "persona" in data:
                persona_data = data["persona"]
                self.persona = CoreMemoryBlock(
                    name=persona_data.get("name", "persona"),
                    content=persona_data.get("content", ""),
                    last_updated=persona_data.get("last_updated", datetime.now().isoformat()),
                    max_length=persona_data.get("max_length", 1500)
                )
            
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки Core Memory: {e}")
            return False
    
    def export_memory(self) -> str:
        """Экспорт всей Core Memory в JSON"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "human": self.human.to_dict(),
            "persona": self.persona.to_dict()
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def import_memory(self, json_data: str) -> str:
        """Импорт Core Memory из JSON"""
        try:
            data = json.loads(json_data)
            
            if "human" in data:
                human_data = data["human"]
                if self.human.update_content(human_data.get("content", "")):
                    pass
                else:
                    return "❌ Human блок слишком большой для импорта"
            
            if "persona" in data:
                persona_data = data["persona"]
                if self.persona.update_content(persona_data.get("content", "")):
                    pass
                else:
                    return "❌ Persona блок слишком большой для импорта"
            
            #self.save_to_file()
            return "✅ Core Memory импортирована успешно"
            
        except Exception as e:
            return f"❌ Ошибка импорта Core Memory: {e}"


def create_core_memory_tools(core_memory: CoreMemory) -> Dict[str, Dict]:
    """
    Создает tool-функции для работы с Core Memory
    
    Args:
        core_memory: Экземпляр CoreMemory
        
    Returns:
        Словарь с конфигурацией инструментов
    """
    
    tools = {}
    
    # Human блок tools
    tools["core_memory_add_human"] = {
        "name": "core_memory_add_human",
        "description": "Добавить информацию о пользователе в Core Memory (human блок)",
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "Информация о пользователе для добавления"
                }
            },
            "required": ["information"]
        },
        "function": core_memory.add_to_human
    }
    
    tools["core_memory_edit_human"] = {
        "name": "core_memory_edit_human", 
        "description": "Полностью переписать информацию о пользователе в Core Memory (human блок)",
        "parameters": {
            "type": "object",
            "properties": {
                "new_content": {
                    "type": "string",
                    "description": "Новое содержимое human блока"
                }
            },
            "required": ["new_content"]
        },
        "function": core_memory.edit_human
    }
    
    # Persona блок tools
    tools["core_memory_add_persona"] = {
        "name": "core_memory_add_persona",
        "description": "Добавить информацию о чат боте в Core Memory (persona блок)",
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "Информация о боте для добавления"
                }
            },
            "required": ["information"]
        },
        "function": core_memory.add_to_persona
    }
    
    tools["core_memory_edit_persona"] = {
        "name": "core_memory_edit_persona",
        "description": "Полностью переписать информацию о чат боте в Core Memory (persona блок)",
        "parameters": {
            "type": "object",
            "properties": {
                "new_content": {
                    "type": "string", 
                    "description": "Новое содержимое persona блока"
                }
            },
            "required": ["new_content"]
        },
        "function": core_memory.edit_persona
    }
    
    # Вспомогательные tools
    tools["core_memory_get_human"] = {
        "name": "core_memory_get_human",
        "description": "Получить информацию о пользователе из Core Memory",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: core_memory.get_human_info()
    }
    
    tools["core_memory_get_persona"] = {
        "name": "core_memory_get_persona",
        "description": "Получить информацию о чат боте из Core Memory",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: core_memory.get_persona_info()
    }
    
    tools["core_memory_stats"] = {
        "name": "core_memory_stats",
        "description": "Показать статистику Core Memory",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "function": lambda: core_memory.get_stats()
    }
    
    return tools