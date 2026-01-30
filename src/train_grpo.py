import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from reward_gemini import gemini_judge_reward
from prompts import SYSTEM_PROMPT, JUDGE_PROMPT

load_dotenv()
os.environ["JUDGE_PROMPT"] = JUDGE_PROMPT 

def main():
    model_name = os.getenv("MODEL_NAME", "microsoft/phi-4")
    data_path = os.getenv("TRAIN_DATA", "data/train.json") 
    output_dir = os.getenv("OUTPUT_DIR", "runs/phi4-grpo-lora")

    # W&B
    os.environ.setdefault("WANDB_PROJECT", os.getenv("WANDB_PROJECT", "grpo-laudite"))
    if os.getenv("WANDB_ENTITY"):
        os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY", "")
    if os.getenv("WANDB_RUN_NAME"):
        os.environ["WANDB_NAME"] = os.getenv("WANDB_RUN_NAME", "")

    ds = load_dataset("json", data_files=data_path, split="train")

    def flatten(ex):
        extra = ex.get("extra", {}) or {}
        return {
            "id": ex.get("id"),
            "input": ex.get("input", ""),
            "label": ex.get("label", ""),
            "initial_text": extra.get("initial_text", ""),
            "user_message": extra.get("user_message", ""),
            "user_context": extra.get("user_context", ""),
            "preview_report_user_example": extra.get("preview_report_user_example", ""),
            "default_context": extra.get("default_context", ""),
        }

    ds = ds.map(flatten, remove_columns=ds.column_names)

    def build_prompt(ex):
        return {"prompt": SYSTEM_PROMPT + "\n\n" + ex["input"]}

    ds = ds.map(build_prompt)

    print(f"--- Carregando modelo {model_name} (Modo PyTorch Nativo - Sem vLLM) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carregamos com SDPA para ser rápido na H100
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,           
        attn_implementation="sdpa",     
        device_map="cuda"              
    )
    
    print("--- Modelo carregado. Iniciando Trainer... ---")

    peft_config = LoraConfig(
        r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=os.getenv("LORA_TARGET", "all-linear"),
    )

    args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "8")),
        learning_rate=float(os.getenv("LR", "1e-5")),
        num_train_epochs=float(os.getenv("EPOCHS", "1")),
        logging_steps=int(os.getenv("LOGGING_STEPS", "1")),
        save_steps=int(os.getenv("SAVE_STEPS", "10")),
        bf16=True, 
        report_to=["wandb"],
        num_generations=int(os.getenv("NUM_GENERATIONS", "4")),
        max_prompt_length=int(os.getenv("MAX_PROMPT_LEN", "4096")),
        max_completion_length=int(os.getenv("MAX_COMPLETION_LEN", "1536")),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        top_p=float(os.getenv("TOP_P", "0.95")),
        
        
        use_vllm=False,
        
        
        disable_dropout=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=gemini_judge_reward,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()