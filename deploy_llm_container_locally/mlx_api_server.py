#!/usr/bin/env python3
"""
OpenAI-compatible API server using MLX-LM for Apple Silicon.
Much faster than vLLM CPU mode on M1/M2/M3 Macs.
"""

import argparse
import time
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from mlx_lm import load, generate


app = FastAPI(title="MLX-LM OpenAI-Compatible API")

# Global model storage
model_config = {}


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "mlx"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_config["model_name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Convert messages to prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant:"

        # Generate response using MLX
        # Note: MLX-LM's generate() doesn't accept temperature in current version
        response_text = generate(
            model_config["model"],
            model_config["tokenizer"],
            prompt=prompt,
            max_tokens=request.max_tokens,
            verbose=False
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    try:
        # Note: MLX-LM's generate() doesn't accept temperature in current version
        response_text = generate(
            model_config["model"],
            model_config["tokenizer"],
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            verbose=False
        )

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.prompt.split()) + len(response_text.split())
            }
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="MLX model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Loading MLX model: {args.model}")
    print("This will use Apple Metal GPU acceleration...")
    
    # Load model using MLX
    model, tokenizer = load(args.model)
    
    model_config["model"] = model
    model_config["tokenizer"] = tokenizer
    model_config["model_name"] = args.model

    print(f"Model loaded successfully!")
    print(f"Starting server on {args.host}:{args.port}")
    print(f"OpenAI-compatible endpoints:")
    print(f"  - http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  - http://{args.host}:{args.port}/v1/completions")
    print(f"  - http://{args.host}:{args.port}/v1/models")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

