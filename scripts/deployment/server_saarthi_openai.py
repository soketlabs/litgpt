import torch
import threading
import asyncio
import uuid
import time
import queue as queue_module
from typing import List, Optional, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# =========================================================
# Model Configuration (FROM YOUR SCRIPT)
# =========================================================

MODEL_ID = "soketlabs/sarthi-agri-v1"

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
# Cancellation Support
# =========================================================

# Maps request_id -> threading.Event; set the event to cancel generation.
active_generations: Dict[str, threading.Event] = {}


class StopOnEvent(StoppingCriteria):
    """Custom stopping criteria that halts model.generate() when the event is set."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_event.is_set()


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
# (streaming is now handled inline in the endpoint)
# =========================================================
        
        

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
async def chat_completions(req: ChatCompletionRequest, request: Request):

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
    # STREAMING MODE  (with cancellation support)
    # -----------------------------------------------------
    if req.stream:
        stop_event = threading.Event()
        active_generations[request_id] = stop_event

        async def event_generator():
            # Set up streamer + generation thread
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            gen_args = dict(generation_args)
            gen_args["input_ids"] = inputs["input_ids"]
            gen_args["attention_mask"] = inputs["attention_mask"]
            gen_args["streamer"] = streamer
            gen_args["stopping_criteria"] = StoppingCriteriaList(
                [StopOnEvent(stop_event)]
            )

            thread = threading.Thread(target=model.generate, kwargs=gen_args)
            thread.start()

            loop = asyncio.get_event_loop()

            try:
                while True:
                    # --- check if the client has gone away ---
                    if await request.is_disconnected():
                        print(f"[cancel] Client disconnected: {request_id}")
                        stop_event.set()
                        break

                    # --- non-blocking read from the streamer queue ---
                    try:
                        token = await loop.run_in_executor(
                            None,
                            lambda: streamer.text_queue.get(timeout=0.5),
                        )
                    except queue_module.Empty:
                        # No token yet; loop back and re-check disconnect
                        continue

                    # End-of-stream sentinel
                    if token is streamer.stop_signal:
                        break

                    if stop_event.is_set():
                        break

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
                    yield {"event": "data", "data": chunk}

                # Send [DONE] only if we weren't cancelled
                if not stop_event.is_set():
                    yield {"event": "data", "data": "[DONE]"}

            except asyncio.CancelledError:
                print(f"[cancel] Async cancelled: {request_id}")
                stop_event.set()

            except Exception as e:
                yield {"event": "data", "data": f"ERROR: {str(e)}"}

            finally:
                # Always ensure the generation thread is stopped & cleaned up
                stop_event.set()
                active_generations.pop(request_id, None)
                thread.join(timeout=10)
                print(f"[cleanup] Generation thread done: {request_id}")

        return EventSourceResponse(event_generator())

    # -----------------------------------------------------
    # NON-STREAMING MODE  (with cancellation support)
    # -----------------------------------------------------
    stop_event = threading.Event()
    active_generations[request_id] = stop_event

    try:
        print(f"the prompt is {prompt} ")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_args["stopping_criteria"] = StoppingCriteriaList(
            [StopOnEvent(stop_event)]
        )

        # Run generation in a thread so the event loop stays responsive
        def _generate():
            with torch.no_grad():
                return model.generate(**inputs, **generation_args)

        output = await asyncio.get_event_loop().run_in_executor(None, _generate)

        # If the client disconnected while we were generating, bail out
        if await request.is_disconnected():
            return JSONResponse(
                {"error": "Client disconnected"}, status_code=499
            )

        generated_tokens = output.sequences[0]
        decoded_output = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        finish = "stop" if not stop_event.is_set() else "cancelled"
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
                        "content": decoded_output,
                    },
                    "finish_reason": finish,
                }
            ],
        }
        return JSONResponse(response)

    finally:
        active_generations.pop(request_id, None)


# =========================================================
# Explicit Cancel Endpoint
# =========================================================


class CancelRequest(BaseModel):
    request_id: str


@app.post("/v1/chat/completions/cancel")
async def cancel_generation(req: CancelRequest):
    """
    Explicitly cancel an in-flight generation by its request_id.
    The request_id is returned in every streaming chunk's "id" field.
    """
    stop_event = active_generations.get(req.request_id)
    if stop_event is None:
        return JSONResponse(
            {"error": "Request not found or already completed"},
            status_code=404,
        )
    stop_event.set()
    return JSONResponse({"status": "cancelled", "request_id": req.request_id})


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
