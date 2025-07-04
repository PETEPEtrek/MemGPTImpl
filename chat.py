import sys
import os
from datetime import datetime

# Фиксируем OpenMP ошибку на macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tool_system import NativeToolSystem
from tools.memory.core_memory import CoreMemory, create_core_memory_tools
from tools.memory.knowledge_base import KnowledgeBase, create_knowledge_tools
from tools.memory.dialog_fifo import DialogFIFO, create_dialog_fifo_tools


class TripleMemoryChat:
    """Чат с тройной системой памяти"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        self.core_memory = CoreMemory()
        self.knowledge_base = KnowledgeBase(model_name)
        self.dialog_fifo = DialogFIFO(model_name, max_prompt_tokens=2500)
        self.tool_system = NativeToolSystem(model_name)

        self._register_tools()
    
    def _register_tools(self):
        """Регистрация всех инструментов"""
        # Core Memory
        for tool_name, tool_config in create_core_memory_tools(self.core_memory).items():
            self.tool_system.register_tool(**tool_config)
        
        # Knowledge Base  
        for tool_name, tool_config in create_knowledge_tools(self.knowledge_base).items():
            self.tool_system.register_tool(**tool_config)
        
        # Dialog FIFO
        for tool_name, tool_config in create_dialog_fifo_tools(self.dialog_fifo).items():
            self.tool_system.register_tool(**tool_config)
    
    def _create_prompt(self, user_message: str) -> str:
        """Создание промпта с контекстом всех систем памяти"""
        
        core_context = self.core_memory.get_core_memory_context()
        dialog_context = self.dialog_fifo.get_dialog_context()
        
        return f"""Ты умный ИИ-ассистент для общения с пользователем:

У тебя есть инструменты для работы с памятью:
- Core Memory: 
  * core_memory_add_human - добавить информацию связанную с пользователем
  * core_memory_edit_human - обновить информацию связанную с пользователем (надо изменить старый факт/старые факты на новый/новые, то есть "Имеет B2 уровень английского" -> "Имеет C1 уровень английского")
  * core_memory_add_persona - добавить информацию о персоне чат бота
  * core_memory_edit_persona - обновить информацию о персоне чат бота
- Knowledge Base: для общих фактов и знаний (не личных)
  * knowledge_add_fact - добавить факт в базу знаний (новый факт)
  * knowledge_edit_fact - обновить факт в базу знаний (надо изменить старый факт/старые факты на новый/новые, то есть "Эйфелева башня была построена в 1889 году" -> "Эйфелева башня была построена в 1887 году")
  * knowledge_search - поиск фактов в базе знаний (поиск по ключевым словам)
- Dialog FIFO: автоматически управляет историей диалогов

ОБЯЗАТЕЛЬНО используй инструменты памяти:
- Если информация о пользователе или связана с ним → ВСЕГДА используй core_memory_add_human или core_memory_edit_human
- Если информация пользователя связана с тобой (чат ботом) → ВСЕГДА используй core_memory_add_persona или core_memory_edit_persona  
- Если упоминаются общие факты (не личные) → используй knowledge_add_fact или knowledge_edit_fact

АЛГОРИТМ:
1. Проанализируй сообщение пользователя на наличие новой информации
2. Если есть информация о пользователе/тебе/общие факты → ОБЯЗАТЕЛЬНО вызови соответствующий инструмент ПЕРЕД ответом
3. Только после использования инструментов дай текстовый ответ
4. Может потребовать вызов нескольких разных инструментов одновременно

ПРИМЕРЫ:
- "меня зовут Петр" → core_memory_add_human("Имя: Петр")
- "тебя зовут Макс" → core_memory_add_persona("Меня зовут Макс")
- "Python создан в 1991" → knowledge_add_fact("Python создан в 1991 году")
- "поднимался на Эверест, это самая высокая гора" → 
  1) core_memory_add_human("Поднимался на Эверест в прошлом месяце")
  2) knowledge_add_fact("Эверест - самая высокая гора в мире")

ПРАВИЛО РАЗДЕЛЕНИЯ:
- Что делал/думает/любит ПОЛЬЗОВАТЕЛЬ → core_memory_*_human
- Что делаешь/думаешь ТЫ (бот) → core_memory_*_persona  
- Общие ФАКТЫ О МИРЕ → knowledge_add_fact

Используй дюбую найденную информацию в твоем промпте или найденную в базе знаний связанную с запросом пользователя.

{core_context}

{dialog_context}

Пользователь: {user_message}

Ответ:"""
    
    def process_message(self, user_message: str) -> str:
        """Обработка сообщения"""
        try:
            # Создаем промпт с контекстом
            prompt = self._create_prompt(user_message)
            
            # Получаем ответ
            response = self.tool_system.generate_response(
                prompt,
                max_tokens=800,
                temperature=0.7
            )
            self.dialog_fifo.add_dialog(user_message, response)
            #self._log_memory_state()
            
            return response
            
        except Exception as e:
            return f"❌ Ошибка: {e}"
    
    def _log_memory_state(self):
        """Логирование состояния всех систем памяти"""
        print("\n" + "─" * 60)
        print("📊 СОСТОЯНИЕ ПАМЯТИ:")
        print("─" * 60)
        
        # Core Memory - показываем содержимое
        print("🧠 CORE MEMORY:")
        print(f"   👤 Human: {self.core_memory.human.content if self.core_memory.human.content.strip() else '(пусто)'}")
        print(f"   🤖 Persona: {self.core_memory.persona.content if self.core_memory.persona.content.strip() else '(пусто)'}")
        
        # Knowledge Base
        kb_stats = self.knowledge_base.get_stats()
        print("📚 " + kb_stats.split('\n')[0])  # Только заголовок
        for line in kb_stats.split('\n')[1:]:
            if line.strip():
                print("   " + line)
                
        # Dialog FIFO
        fifo_stats = self.dialog_fifo.get_stats()
        print("💬 " + fifo_stats.split('\n')[0])  # Только заголовок
        for line in fifo_stats.split('\n')[1:]:
            if line.strip():
                print("   " + line)
        
        print("─" * 60)
    
    def show_stats(self):
        """Показать статистику всех систем"""
        print("\n📊 СТАТИСТИКА ПАМЯТИ:")
        print("=" * 50)
        print(self.core_memory.get_stats())
        print("\n" + self.knowledge_base.get_stats())
        print("\n" + self.dialog_fifo.get_stats())
    
    def run(self):
        
        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/exit":
                    print("👋 До свидания!")
                    break
                
                print("🤖 Бот: ", end="")
                response = self.process_message(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")


def main():
    """Главная функция"""
    try:
        chat = TripleMemoryChat()
        chat.run()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main() 