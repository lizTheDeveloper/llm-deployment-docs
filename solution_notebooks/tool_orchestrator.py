
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import json
import uuid

app = FastAPI(title="LLM Tool Calling Orchestrator")

# Tool definitions
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Simulate weather lookup - in production, call real weather API"""
    import random
    temp = random.randint(15, 30)
    conditions = random.choice(["sunny", "cloudy", "rainy", "partly cloudy"])
    return f"The weather in {location} is {temp}°{unit[0].upper()} and {conditions}."

def calculate_math(expression: str) -> str:
    """Safely evaluate a math expression"""
    try:
        allowed = set("0123456789+-*/() .")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather,
    "calculate_math": calculate_math
}

# Request/Response models
class Message(BaseModel):
    role: str
    content: str

class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

@app.post("/v1/chat/completions")
async def chat_with_tools(request: ChatRequest):
    """
    Orchestration layer: handles tool calling logic
    Calls vLLM for inference, executes tools, returns final response
    """
    vllm_url = "http://localhost:8000/v1/chat/completions"

    # Build prompt with tool descriptions if tools provided
    messages = list(request.messages)

    if request.tools:
        tool_desc = "\n\nAvailable tools:\n"
        for tool in request.tools:
            func = tool.function
            tool_desc += f"- {func['name']}: {func['description']}\n"
            tool_desc += f"  Parameters: {json.dumps(func['parameters'])}\n"
        tool_desc += "\nTo use a tool, respond with JSON: {\"tool\": \"tool_name\", \"arguments\": {...}}\n"

        # Add tool instructions to system message
        if messages[0].role == "system":
            messages[0].content += tool_desc
        else:
            messages.insert(0, Message(role="system", content=f"You are a helpful assistant.{tool_desc}"))

    # Call vLLM for initial inference
    async with httpx.AsyncClient(timeout=30.0) as client:
        vllm_request = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        response = await client.post(vllm_url, json=vllm_request)
        result = response.json()

        assistant_response = result["choices"][0]["message"]["content"]

        # Check if model wants to call a tool
        tool_call = None
        if request.tools:
            try:
                import re
                tool_match = re.search(r'\{[\s\S]*?"tool"[\s\S]*?\}', assistant_response)
                if tool_match:
                    tool_call = json.loads(tool_match.group(0))
            except:
                pass

        # If tool call detected, execute and get final response
        if tool_call and tool_call.get("tool") in AVAILABLE_TOOLS:
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("arguments", {})

            # Execute tool
            tool_function = AVAILABLE_TOOLS[tool_name]
            tool_result = tool_function(**tool_args)

            # Add tool result to conversation
            messages.append(Message(role="assistant", content=assistant_response))
            messages.append(Message(role="function", content=f"Tool result: {tool_result}"))

            # Get final response from vLLM
            vllm_request["messages"] = [{"role": m.role, "content": m.content} for m in messages]
            response = await client.post(vllm_url, json=vllm_request)
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": result.get("created", 0),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop"
            }],
            "usage": result.get("usage", {})
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
