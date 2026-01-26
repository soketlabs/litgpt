import torch
import threading
import uuid
import time
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

# =========================================================
# Model Configuration (FROM YOUR SCRIPT)
# =========================================================

MODEL_ID = "soketlabs/saarthi-agri-v1"

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.8
TOP_P = 0.9
REPETITION_PENALTY = 1.1

print("\n================ LOADING MODEL ================\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()

print("✅ Model loaded successfully\n")

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(title="Saarthi-Agri OpenAI Compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANT: allows OPTIONS
    allow_headers=["*"],
)
# =========================================================
# OpenAI Schema
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE
    top_p: Optional[float] = TOP_P
    repetition_penalty: Optional[float] = REPETITION_PENALTY


# =========================================================
# Prompt Builder (USES YOUR CHAT TEMPLATE)
# =========================================================

def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Always inject the system prompt server-side.
    Client only sends user content.
    """

    #final_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append client messages (user / assistant if any)
    # for m in messages:
    #   final_messages.append({"role": m.role, "content": m.content})
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print(f"the prompt is {prompt}")

    return prompt



# =========================================================
# Streaming Generator
# =========================================================

def token_streamer(prompt: str, gen_args: dict):
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_args["input_ids"] = inputs["input_ids"]
    gen_args["attention_mask"] = inputs["attention_mask"]
    gen_args["streamer"] = streamer

    thread = threading.Thread(target=model.generate, kwargs=gen_args)
    thread.start()

    for token in streamer:
        yield token
        
        

# SYSTEM_PROMPT = """You are a helpful District Agricultural Officer providing crop advisory to farmers based on location and various climatic conditions given as input.

# Output Order (Mandatory and Strict):
# 1. The very first output token must be exactly <unused0>
# 2. Immediately after <unused0>, produce a structured analytical reasoning section in English covering:
# - Crop suitability for the given Month and Region, considering the Crop type and Growth Stage
# - Climate assessment using Weather description
# - Soil behavior, soil moisture retention, and irrigation needs based on Soil Type 
# - Growth-stage-specific agronomic requirements and timing considerations
# - Risk analysis including Stress factors (pests/diseases) and weather-related stress
# - Impact of Farming Practice on productivity and risk mitigation
# - Integrated recommendation logic combining all above parameters coherently
# 3. After the reasoning is complete, output exactly <unused1>
# 4. Only after <unused1>, produce the final advisory response intended for the user.

# Output Restrictions:
# - Do not output anything before <unused0>.
# - Do not output anything between <unused0> and <unused1> except the analytical reasoning section.
# - Do not repeat <unused0> or <unused1>.
# - Do not include meta commentary or explanations about the protocol.
# - The final advisory must be written strictly in the language requested by the user.
# """



# =========================================================
# OpenAI Compatible Endpoint
# =========================================================

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):

    prompt = build_prompt(req.messages)

    generation_args = dict(
        max_new_tokens=req.max_tokens,
        do_sample=False,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        return_dict_in_generate=True,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())

    # -----------------------------------------------------
    # STREAMING MODE
    # -----------------------------------------------------
    if req.stream:

        async def event_generator():
            try:
                for token in token_streamer(prompt, generation_args):
                    print(f"the token is {token} ")
                    chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "delta": {"content": token},
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                    yield {
                          "event": "data",
                        "data": chunk
                    }

                # End marker
                yield {
                    "event": "data",
                    "data": "[DONE]"
                }

            except Exception as e:
                yield {
                    "event": "data",
                    "data": f"ERROR: {str(e)}"
                }

        return EventSourceResponse(event_generator())

    # -----------------------------------------------------
    # NON-STREAMING MODE
    # -----------------------------------------------------
    print(f"the prompt is {prompt} ")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generation_args
        )
        
    

    generated_tokens = output.sequences[0]
    decoded_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": decoded_output
                },
                "finish_reason": "stop",
            }
        ],
    }

    return JSONResponse(response)


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


# =========================================================
# Launch
# =========================================================
# Run using:
# uvicorn server_saarthi_openai:app --host 0.0.0.0 --port 8000
