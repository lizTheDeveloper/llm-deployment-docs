#!/usr/bin/env python3
"""
Lab 7: Deploying LLMs with FastAPI (OpenAI-Compatible)
Test script for creating an OpenAI-compatible API
"""

def main():
    print("=" * 60)
    print("Lab 7: FastAPI OpenAI-Compatible Deployment")
    print("=" * 60)
    
    # Import libraries
    print("\n1️⃣ Importing libraries...")
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import List, Optional
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import uvicorn
        print("✓ All libraries imported successfully")
    except ImportError as e:
        print(f"Error importing libraries: {e}")
        print("Please install: pip install fastapi uvicorn pydantic torch transformers")
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
    
    # Define Pydantic models
    print("\n3️⃣ Defining API schema...")
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 256
        stream: Optional[bool] = False
    
    class Choice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str
    
    class ChatResponse(BaseModel):
        id: str
        object: str
        choices: List[Choice]
        model: str
    
    print("✓ API schema defined")
    
    # Create FastAPI app
    print("\n4️⃣ Creating FastAPI application...")
    
    app = FastAPI(title="OpenAI-Compatible LLM API (Qwen2.5)")
    
    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def create_chat_completion(req: ChatRequest):
        # Extract last user message
        user_message = req.messages[-1].content if req.messages else ""
        
        # Format prompt for chat model
        prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
        
        # Tokenize and generate
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
        
        # Decode response
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "<|assistant|>" in full_text:
            assistant_response = full_text.split("<|assistant|>")[-1].strip()
        else:
            assistant_response = full_text
        
        # Create response
        response = ChatResponse(
            id="chatcmpl-1",
            object="chat.completion",
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=assistant_response),
                finish_reason="stop"
            )],
            model=req.model
        )
        
        return response
    
    @app.get("/")
    async def root():
        return {"message": "OpenAI-Compatible LLM API", "model": model_name}
    
    print("✓ FastAPI application created")
    print("\n5️⃣ Testing the API...")
    
    # Test with a sample request
    import asyncio
    
    test_request = ChatRequest(
        model="qwen2.5",
        messages=[ChatMessage(role="user", content="What is Python?")]
    )
    
    async def test_api():
        try:
            response = await create_chat_completion(test_request)
            print(f"\nTest Request: What is Python?")
            print(f"Response: {response.choices[0].message.content[:200]}...")
            print("\n✓ API test successful!")
            return True
        except Exception as e:
            print(f"\nError testing API: {e}")
            return False
    
    # Run test
    success = asyncio.run(test_api())
    
    if not success:
        return False
    
    # Instructions for running the server
    print("\n" + "=" * 60)
    print("Lab 7 Complete!")
    print("=" * 60)
    print("\nTo run the server:")
    print("  python -c 'from lab7_fastapi_deployment import app; import uvicorn; uvicorn.run(app, host=\"0.0.0.0\", port=8000)'")
    print("\nTo test with OpenAI client:")
    print("  from openai import OpenAI")
    print("  client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')")
    print("  response = client.chat.completions.create(")
    print("      model='qwen2.5',")
    print("      messages=[{'role': 'user', 'content': 'Hello!'}]")
    print("  )")
    
    return True

# Make app accessible for uvicorn
app = None

def create_app():
    """Create and return the FastAPI app for uvicorn"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import List, Optional
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "mps" else "cpu",
        low_cpu_mem_usage=True
    )
    if device == "mps":
        model = model.to(device)
    
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ChatRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 256
        stream: Optional[bool] = False
    
    class Choice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str
    
    class ChatResponse(BaseModel):
        id: str
        object: str
        choices: List[Choice]
        model: str
    
    app = FastAPI(title="OpenAI-Compatible LLM API")
    
    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def create_chat_completion(req: ChatRequest):
        user_message = req.messages[-1].content if req.messages else ""
        prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
        
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
        
        response = ChatResponse(
            id="chatcmpl-1",
            object="chat.completion",
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=assistant_response),
                finish_reason="stop"
            )],
            model=req.model
        )
        return response
    
    @app.get("/")
    async def root():
        return {"message": "OpenAI-Compatible LLM API", "model": model_name}
    
    return app

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

