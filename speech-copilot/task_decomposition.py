# qwen_audio_tool_runner.py

# This framework supports audio tool calling and Qwen-Audio reasoning.
# It allows you to call tools directly or run Qwen-Audio reasoning on a local .wav file.

from fastapi import FastAPI, Form
from typing import Optional
from pydantic import BaseModel
import uvicorn
import os
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from huggingface_hub import list_models

app = FastAPI()

# Define a tool description structure
class Tool(BaseModel):
    name: str
    task: str
    hf_tag: str
    default_model: str

# Predefined list of available tools
TOOL_REGISTRY = {
    "emotion_tool": Tool(
        name="emotion_tool",
        task="audio-classification",
        hf_tag="emotion",
        default_model="speechbrain/emotion-recognition-wav2vec2"
    ),
    "speaker_tool": Tool(
        name="speaker_tool",
        task="audio-classification",
        hf_tag="speaker-identification",
        default_model="superb/hubert-large-superb-sid"
    )
}

# Search Hugging Face for related models (optional override)
def search_hf_models(tool: Tool):
    models = list_models(filter=tool.task, tags=[tool.hf_tag], sort="downloads", direction=-1)
    return [m.modelId for m in models[:3]]

# Run a selected tool on given audio
def run_tool(tool: Tool, audio_path: str, model_id: Optional[str] = None):
    model_to_use = model_id or tool.default_model
    pipe = pipeline(task=tool.task, model=model_to_use)
    result = pipe(audio_path)
    return result

# Load Qwen-Audio model and processor once at startup
qwen_model = AutoModelForSpeechSeq2Seq.from_pretrained("Qwen/Qwen-Audio-Chat")
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio-Chat")

def local_qwen_audio_infer(audio_path: str, user_prompt: str):
    # Load and preprocess the audio
    inputs = qwen_processor(text=user_prompt, audio=audio_path, return_tensors="pt")
    with torch.no_grad():
        output = qwen_model.generate(**inputs, max_new_tokens=256)
    result = qwen_processor.batch_decode(output, skip_special_tokens=True)
    return result[0]

# Local file-based endpoint for Qwen-Audio reasoning
@app.get("/ask_file")
def ask_with_audio_path(audio_path: str = Form(...), prompt: str = Form(...)):
    if not os.path.isfile(audio_path):
        return {"error": "Audio file not found."}
    response = local_qwen_audio_infer(audio_path, prompt)
    return {"qwen_response": response}

# File path version of run_tool
@app.get("/run_tool_file")
def run_tool_from_path(audio_path: str = Form(...), tool_name: str = Form(...), override_model: Optional[str] = Form(None)):
    if not os.path.isfile(audio_path):
        return {"error": "Audio file not found."}
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Tool {tool_name} not found"}
    tool = TOOL_REGISTRY[tool_name]
    result = run_tool(tool, audio_path, override_model)
    return {"result": result}

# Sample endpoint to list available tools
@app.get("/tools")
def list_tools():
    return {k: v.dict() for k, v in TOOL_REGISTRY.items()}

if __name__ == "__main__":
    uvicorn.run("qwen_audio_tool_runner:app", host="0.0.0.0", port=8080, reload=True)
