#!/usr/bin/env python3
"""
AI Model Training Backend Server
Supports real-time training with WebSocket updates
Optimized for various model architectures including Causal LM and Seq2Seq
Author: Your Name
License: MIT
"""

import asyncio
import json
import logging
import os
import sys
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import psutil

# GPU monitoring - optional, won't fail if not available
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPUtil not installed. GPU monitoring disabled.")

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# PEFT imports - optional for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not installed. LoRA/QLoRA training disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Training Server",
    description="Local AI model fine-tuning platform with real-time monitoring",
    version="1.0.0"
)

# Global training state
training_state = {
    "active": False,
    "progress": 0,
    "epoch": 0,
    "loss": 0,
    "step": 0,
    "total_steps": 0,
    "start_time": None,
    "model": None,
    "trainer": None
}

MODEL_CONFIGS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "causal", "size": "1.1B", "recommended_batch_size": 4
    },
    "gemma:2b": {
        "name": "google/gemma-2b",
        "type": "causal", "size": "2B", "recommended_batch_size": 2
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "type": "causal", "size": "2.7B", "recommended_batch_size": 1
    },
    "stable-code-3b": {
        "name": "stabilityai/stable-code-3b-4k",
        "type": "causal", "size": "3B", "recommended_batch_size": 1
    },
    "pythia-1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "type": "causal", "size": "1.4B", "recommended_batch_size": 2
    },
    "starcoder2:3b": {
        "name": "bigcode/starcoder2-3b",
        "type": "causal", "size": "3B", "recommended_batch_size": 1
    },
    "codet5p-770m": {
        "name": "Salesforce/codet5p-770M-py",
        "type": "seq2seq", "size": "770M", "recommended_batch_size": 2
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "type": "causal", "size": "355M", "recommended_batch_size": 4
    },
    "replit-code-v1-3b": {
        "name": "replit/replit-code-v1-3b",
        "type": "causal", "size": "3B", "recommended_batch_size": 1
    },
    "open_llama_3b_v2": {
        "name": "openlm-research/open_llama_3b_v2",
        "type": "causal", "size": "3B", "recommended_batch_size": 1
    },
    "falcon-1b": {
        "name": "tiiuae/falcon-rw-1b",
        "type": "causal", "size": "1B", "recommended_batch_size": 2
    },
    "tinydolphin": {
        "name": "cognitivecomputations/TinyDolphin-2.8-1.1b",
        "type": "causal", "size": "1.1B", "recommended_batch_size": 4
    }
}

class TrainingConfig(BaseModel):
    model: str
    method: str = "lora"
    dataset: str
    batch_size: int = 1
    learning_rate: float = 2e-5
    epochs: int = 1
    max_length: int = 512
    fp16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./models/finetuned"

class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("connection open")
    
    def disconnect(self, websocket: WebSocket):
        logger.info("connection closed")
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_update(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

manager = WebSocketManager()

class TrainingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            asyncio.create_task(manager.send_update({
                "type": "metrics", "epoch": state.epoch, "loss": logs.get("loss", 0),
                "step": state.global_step, "total_steps": state.max_steps,
                "learning_rate": logs.get("learning_rate", 0)
            }))
            training_state.update({
                "epoch": state.epoch, "loss": logs.get("loss", 0), "step": state.global_step,
                "total_steps": state.max_steps,
                "progress": (state.global_step / state.max_steps * 100) if state.max_steps > 0 else 0
            })
    
    def on_epoch_end(self, args, state, control, **kwargs):
        asyncio.create_task(manager.send_update({
            "type": "log", "level": "info", "message": f"Completed epoch {int(state.epoch)}"
        }))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

@app.post("/api/train")
async def start_training(config: TrainingConfig):
    if training_state["active"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    training_state.update({
        "active": True, "progress": 0, "epoch": 0, "loss": 0, "step": 0,
        "total_steps": 0, "start_time": datetime.now()
    })
    asyncio.create_task(run_training(config))
    return {"status": "started", "config": config.dict()}

@app.post("/api/stop")
async def stop_training():
    if not training_state["active"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    if training_state["trainer"]:
        training_state["trainer"].args.should_stop = True
    training_state["active"] = False
    return {"status": "stopped"}

@app.get("/api/status")
async def get_status():
    gpu_info = get_gpu_stats()
    return {
        "training": training_state["active"], "progress": training_state["progress"],
        "epoch": training_state["epoch"], "loss": training_state["loss"],
        "step": training_state["step"], "total_steps": training_state["total_steps"],
        "gpu": gpu_info,
        "system": {
            "cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent
        }
    }

def get_gpu_stats():
    if GPU_AVAILABLE:
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                "name": gpu.name, "memory_used": f"{gpu.memoryUsed:.1f}",
                "memory_total": f"{gpu.memoryTotal:.1f}", "utilization": gpu.load * 100,
                "temperature": gpu.temperature
            }
        except: pass
    return {"name": "GPU not detected", "memory_used": "0.0", "memory_total": "0.0", "utilization": 0, "temperature": 0}

def get_model_architecture(model_name: str):
    if any(x in model_name.lower() for x in ['t5', 'flan', 'ul2']):
        return "seq2seq", AutoModelForSeq2SeqLM
    return "causal", AutoModelForCausalLM

async def run_training(config: TrainingConfig):
    try:
        await manager.send_update({"type": "log", "level": "info", "message": "Initializing training..."})
        if not os.path.exists(config.dataset):
            raise FileNotFoundError(f"Dataset not found: {config.dataset}")

        model_config_data = MODEL_CONFIGS.get(config.model)
        if not model_config_data:
            raise ValueError(f"Model key '{config.model}' not found in MODEL_CONFIGS.")
        
        full_model_name = model_config_data["name"]
        model_type, model_class = get_model_architecture(full_model_name)
        
        await manager.send_update({"type": "log", "level": "info", "message": f"Model type detected: {model_type} for {full_model_name}"})
        
        tokenizer = AutoTokenizer.from_pretrained(full_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {"torch_dtype": torch.float16 if config.fp16 else torch.float32}
        
        if config.method in ["lora", "qlora"] and PEFT_AVAILABLE:
            if config.method == "qlora":
                try:
                    import bitsandbytes
                    model_kwargs["load_in_8bit"] = True
                except ImportError:
                    await manager.send_update({"type": "log", "level": "warning", "message": "bitsandbytes not installed. Using LoRA."})
                    config.method = "lora"
            
            model = model_class.from_pretrained(full_model_name, **model_kwargs)
            
            model = prepare_model_for_kbit_training(model)

            task_type = TaskType.SEQ_2_SEQ_LM if model_type == "seq2seq" else TaskType.CAUSAL_LM
            peft_config = LoraConfig(
                task_type=task_type, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            await manager.send_update({"type": "log", "level": "info", "message": f"Using {config.method.upper()}"})
        else:
            model = model_class.from_pretrained(full_model_name, **model_kwargs)
            await manager.send_update({"type": "log", "level": "info", "message": "Using full fine-tuning"})

        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        await manager.send_update({"type": "log", "level": "info", "message": f"Model loaded: {full_model_name}"})
        
        dataset = load_dataset("json", data_files=config.dataset)
        
        def tokenize_function(examples):
            key_to_use = None
            for key in ["text", "content", "response", "instruction", "output"]:
                if key in examples:
                    key_to_use = key
                    break
            
            if key_to_use is None:
                raise ValueError("Dataset does not contain any of the expected text columns: text, content, response, instruction, output")

            texts_to_tokenize = [str(text) if text is not None else "" for text in examples[key_to_use]]

            return tokenizer(
                texts_to_tokenize,
                truncation=True,
                padding="max_length",
                max_length=config.max_length
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
        
        await manager.send_update({"type": "log", "level": "info", "message": f"Dataset loaded and tokenized: {len(tokenized_dataset['train'])} samples"})
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=config.output_dir, num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4, warmup_steps=50, logging_steps=10,
            save_steps=500, save_total_limit=2, fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing, learning_rate=config.learning_rate,
            report_to="tensorboard", logging_dir=f"{config.output_dir}/logs",
            remove_unused_columns=False, optim="adamw_torch"
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) if model_type == "causal" else DataCollatorForSeq2Seq(tokenizer, model=model)
        
        trainer = Trainer(
            model=model, args=training_args, train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer, data_collator=data_collator, callbacks=[TrainingCallback()]
        )
        
        training_state["trainer"] = trainer
        training_state["total_steps"] = (len(tokenized_dataset["train"]) // (config.batch_size * training_args.gradient_accumulation_steps)) * config.epochs
        
        await manager.send_update({"type": "log", "level": "info", "message": "Starting training..."})
        
        train_result = trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        with open(os.path.join(config.output_dir, "training_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        await manager.send_update({"type": "log", "level": "success", "message": f"Training completed! Model saved to {config.output_dir}"})
        await manager.send_update({"type": "log", "level": "info", "message": f"Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}"})
        
    except FileNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        await manager.send_update({"type": "log", "level": "error", "message": f"Dataset not found: {str(e)}"})
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory")
        await manager.send_update({"type": "log", "level": "error", "message": "GPU out of memory! Try reducing batch_size or max_length."})
    except Exception as e:
        logger.error(f"Training error: {e}")
        await manager.send_update({"type": "log", "level": "error", "message": f"Training failed: {str(e)}"})
    finally:
        training_state["active"] = False
        training_state["trainer"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("models/finetuned", exist_ok=True)
    
    if not os.path.exists("static/index.html"):
        print("⚠️  Warning: static/index.html not found!")
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {vram_gb:.1f} GB")
    else:
        print("⚠️  CUDA not available. Training will be slow on CPU!")
    
    uvicorn.run("training_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
