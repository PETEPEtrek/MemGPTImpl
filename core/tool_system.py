#!/usr/bin/env python3
import json
import inspect
import re
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SharedModelManager:
    """Менеджер для совместного использования модели между компонентами"""
    
    _instances = {}
    
    @classmethod
    def get_model(cls, model_name: str):
        """Получить shared экземпляр модели"""
        if model_name not in cls._instances:
            print(f"🤖 Загружаю shared модель {model_name}...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Выбираем устройство
            if torch.backends.mps.is_available():
                device = "mps"
                print("🚀 Используется Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                print("⚠️ Используется CPU")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                trust_remote_code=True,
                device_map=None
            ).to(device)
            
            model.eval()
            
            cls._instances[model_name] = {
                'tokenizer': tokenizer,
                'model': model,
                'device': device
            }
            print("✅ Shared модель загружена")
        
        return cls._instances[model_name]
    
    @classmethod
    def clear_cache(cls):
        """Очистить кеш моделей"""
        cls._instances.clear()


@dataclass
class ToolFunction:
    """Описание функции-инструмента"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для промпта"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class NativeToolSystem:
    """Нативная система инструментов для Qwen модели"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.tools: Dict[str, ToolFunction] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Получаем shared модель и токенизатор
        print(f"🤖 Подключаюсь к shared модели {model_name}...")
        model_components = SharedModelManager.get_model(model_name)
        self.tokenizer = model_components['tokenizer']
        self.model = model_components['model']
        self.device = model_components['device']
        print("✅ NativeToolSystem готов")
    
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     parameters: Dict[str, Any], 
                     function: Callable) -> None:
        """
        Регистрация инструмента
        
        Args:
            name: Имя функции
            description: Описание функции
            parameters: JSON Schema параметров
            function: Сама функция для выполнения
        """
        tool = ToolFunction(name, description, parameters, function)
        self.tools[name] = tool
        print(f"🔧 Зарегистрирован инструмент: {name}")
    
    def tool(self, name: str, description: str, parameters: Dict[str, Any]):
        """Декоратор для регистрации инструментов"""
        def decorator(func: Callable):
            self.register_tool(name, description, parameters, func)
            return func
        return decorator
    
    def auto_register_tool(self, func: Callable) -> None:
        """Автоматическая регистрация функции как инструмента"""
        name = func.__name__
        description = func.__doc__ or f"Функция {name}"
        
        # Извлекаем параметры из signature
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_type = "string"  # По умолчанию
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            
            parameters["properties"][param_name] = {
                "type": param_type,
                "description": f"Параметр {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        self.register_tool(name, description, parameters, func)
    
    def _format_tools_for_prompt(self) -> str:
        """Форматирование инструментов для промпта"""
        if not self.tools:
            return ""
        
        tools_json = [tool.to_dict() for tool in self.tools.values()]
        return f"\n\nДоступные инструменты:\n{json.dumps(tools_json, ensure_ascii=False, indent=2)}"
    
    def _create_system_prompt(self) -> str:
        """Создание системного промпта"""
        system_prompt = """Ты полезный ИИ-ассистент с доступом к инструментам. 

Когда тебе нужно использовать инструмент, отвечай в следующем формате:
<tool_call>
{"name": "имя_функции", "arguments": {"параметр": "значение"}}
</tool_call>

После использования инструмента ты получишь результат и сможешь дать финальный ответ пользователю.
Если инструменты не нужны, отвечай как обычно."""
        
        tools_info = self._format_tools_for_prompt()
        return system_prompt + tools_info
    
    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Извлечение вызовов инструментов из ответа модели"""
        tool_calls = []
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                print(f"⚠️ Ошибка парсинга tool call: {match}")
        
        return tool_calls
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Выполнение вызова инструмента"""
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        if tool_name not in self.tools:
            return f"❌ Инструмент '{tool_name}' не найден"
        
        try:
            tool = self.tools[tool_name]
            result = tool.function(**arguments)
            return f"✅ Результат {tool_name}: {result}"
        except Exception as e:
            return f"❌ Ошибка выполнения {tool_name}: {str(e)}"
    
    def generate_response(self, 
                         user_input: str, 
                         max_tokens: int = 800,
                         temperature: float = 0.7) -> str:
        """Генерация ответа с поддержкой инструментов"""
        
        # Добавляем сообщение пользователя в историю
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Создаем промпт
        messages = [
            {"role": "system", "content": self._create_system_prompt()}
        ] + self.conversation_history
        
        # Форматируем через chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Генерируем ответ
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Декодируем ответ
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Проверяем на tool calls
        tool_calls = self._extract_tool_calls(generated_text)
        
        if tool_calls:
            # Выполняем инструменты
            tool_results = []
            for tool_call in tool_calls:
                result = self._execute_tool_call(tool_call)
                tool_results.append(result)
            
            # Добавляем результаты в историю
            self.conversation_history.append({
                "role": "assistant",
                "content": generated_text
            })
            
            tool_results_text = "\n".join(tool_results)
            self.conversation_history.append({
                "role": "system",
                "content": f"Результаты выполнения инструментов:\n{tool_results_text}"
            })
            
            # Генерируем финальный ответ
            messages = [
                {"role": "system", "content": self._create_system_prompt()}
            ] + self.conversation_history + [
                {"role": "user", "content": "Дай финальный ответ на основе результатов инструментов."}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            final_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response
        
        else:
            # Обычный ответ без инструментов
            self.conversation_history.append({
                "role": "assistant",
                "content": generated_text
            })
            return generated_text
    
    def clear_history(self):
        """Очистка истории разговора"""
        self.conversation_history = []
        print("🧹 История разговора очищена")
    
    def get_registered_tools(self) -> List[str]:
        """Получение списка зарегистрированных инструментов"""
        return list(self.tools.keys())