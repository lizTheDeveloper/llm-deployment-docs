#!/usr/bin/env python3
"""
Lab 8: FastAPI with Tool Calling (Simplified)
Demonstrates how to implement tool/function calling in an OpenAI-compatible API
Note: Simplified version for Mac M3 - uses a small model
"""

import json
import re

def main():
    print("=" * 60)
    print("Lab 8: FastAPI Tool Calling (Simplified)")
    print("=" * 60)
    
    # Import libraries
    print("\n1️⃣ Importing libraries...")
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import List, Optional, Dict, Any
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ All libraries imported successfully")
    except ImportError as e:
        print(f"Error importing libraries: {e}")
        return False
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model
    print("\n2️⃣ Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device != "mps" else "cpu",
            low_cpu_mem_usage=True
        )
        if device == "mps":
            model = model.to(device)
        
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Define tool/function
    print("\n3️⃣ Defining example tool...")
    
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        """
        Simulates getting weather for a location
        """
        # Simulated weather data
        weather_data = {
            "san francisco": {"temp": 18, "condition": "partly cloudy"},
            "new york": {"temp": 22, "condition": "sunny"},
            "london": {"temp": 15, "condition": "rainy"},
            "tokyo": {"temp": 20, "condition": "clear"},
        }
        
        location_key = location.lower()
        if location_key in weather_data:
            data = weather_data[location_key]
            temp = data["temp"]
            if unit.lower() == "fahrenheit":
                temp = (temp * 9/5) + 32
            return f"The weather in {location} is {data['condition']} with a temperature of {temp}° {unit}."
        else:
            return f"Weather data not available for {location}."
    
    print("✓ Tool defined: get_current_weather")
    
    # Define API schema
    print("\n4️⃣ Defining API schema...")
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ToolFunction(BaseModel):
        name: str
        description: str
        parameters: Dict[str, Any]
    
    class Tool(BaseModel):
        type: str = "function"
        function: ToolFunction
    
    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 256
        tools: Optional[List[Tool]] = None
        tool_choice: Optional[str] = "auto"
    
    class ToolCall(BaseModel):
        id: str
        type: str = "function"
        function: Dict[str, Any]
    
    class ChatResponseMessage(BaseModel):
        role: str
        content: Optional[str] = None
        tool_calls: Optional[List[ToolCall]] = None
    
    class Choice(BaseModel):
        index: int
        message: ChatResponseMessage
        finish_reason: str
    
    class ChatResponse(BaseModel):
        id: str
        object: str
        choices: List[Choice]
        model: str
    
    print("✓ API schema defined")
    
    # Create FastAPI app
    print("\n5️⃣ Creating FastAPI application with tool calling...")
    
    app = FastAPI(title="OpenAI-Compatible API with Tool Calling")
    
    # Simple tool call parser (for demonstration)
    def parse_tool_call(text: str, available_tools: List[Tool]) -> Optional[Dict]:
        """
        Simple heuristic to detect if the model wants to call a tool
        In production, you'd train the model to output structured tool calls
        """
        text_lower = text.lower()
        
        # Check if the response mentions weather and a location
        if "weather" in text_lower:
            # Try to extract location
            for tool in available_tools:
                if tool.function.name == "get_current_weather":
                    # Simple location extraction
                    cities = ["san francisco", "new york", "london", "tokyo"]
                    for city in cities:
                        if city in text_lower:
                            return {
                                "name": "get_current_weather",
                                "arguments": json.dumps({"location": city.title(), "unit": "celsius"})
                            }
        
        return None
    
    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def create_chat_completion(req: ChatRequest):
        user_message = req.messages[-1].content if req.messages else ""
        
        # If tools are provided, include them in the prompt
        if req.tools:
            tools_description = "\n\nAvailable tools:\n"
            for tool in req.tools:
                tools_description += f"- {tool.function.name}: {tool.function.description}\n"
            
            prompt = f"<|system|>\nYou are a helpful assistant. {tools_description}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
        else:
            prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
        
        # Generate initial response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in full_text:
            assistant_response = full_text.split("<|assistant|>")[-1].strip()
        else:
            assistant_response = full_text
        
        # Check if we should call a tool
        if req.tools:
            tool_call_info = parse_tool_call(assistant_response, req.tools)
            
            if tool_call_info:
                # Execute the tool
                if tool_call_info["name"] == "get_current_weather":
                    args = json.loads(tool_call_info["arguments"])
                    tool_result = get_current_weather(**args)
                    
                    # Generate final response with tool result
                    follow_up_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_message}\n\nTool result: {tool_result}</s>\n<|assistant|>\n"
                    
                    inputs = tokenizer(follow_up_prompt, return_tensors="pt").to(device)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=req.max_tokens,
                            temperature=req.temperature,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "<|assistant|>" in full_text:
                        final_response = full_text.split("<|assistant|>")[-1].strip()
                    else:
                        final_response = full_text
                    
                    assistant_response = final_response
        
        # Create response
        response = ChatResponse(
            id="chatcmpl-tool-1",
            object="chat.completion",
            choices=[Choice(
                index=0,
                message=ChatResponseMessage(
                    role="assistant",
                    content=assistant_response
                ),
                finish_reason="stop"
            )],
            model=req.model
        )
        
        return response
    
    @app.get("/")
    async def root():
        return {"message": "OpenAI-Compatible API with Tool Calling", "model": model_name}
    
    print("✓ FastAPI application created")
    
    # Test the API
    print("\n6️⃣ Testing tool calling...")
    
    import asyncio
    
    # Define test tools
    test_tools = [
        Tool(
            type="function",
            function=ToolFunction(
                name="get_current_weather",
                description="Get the current weather for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            )
        )
    ]
    
    test_request = ChatRequest(
        model="qwen2.5",
        messages=[ChatMessage(role="user", content="What's the weather in San Francisco?")],
        tools=test_tools,
        tool_choice="auto"
    )
    
    async def test_api():
        try:
            response = await create_chat_completion(test_request)
            print(f"\nTest Request: What's the weather in San Francisco?")
            print(f"Response: {response.choices[0].message.content[:300]}...")
            print("\n✓ API test successful!")
            return True
        except Exception as e:
            print(f"\nError testing API: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run test
    success = asyncio.run(test_api())
    
    if not success:
        return False
    
    # Instructions
    print("\n" + "=" * 60)
    print("Lab 8 Complete!")
    print("=" * 60)
    print("\nKey Concepts Demonstrated:")
    print("- Tool/function schema definition")
    print("- Parsing model output for tool calls")
    print("- Executing tools and feeding results back")
    print("- Multi-turn conversation with tool results")
    print("\nNote: Production systems would use:")
    print("- Fine-tuned models for structured tool calling")
    print("- Proper JSON parsing and validation")
    print("- Error handling and retry logic")
    print("- Authentication and rate limiting")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

