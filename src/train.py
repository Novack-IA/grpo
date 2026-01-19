import os
import torch
import wandb
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from transformers import TrainerCallback
from dotenv import load_dotenv

# Importando seus componentes existentes
from rewards.rewards import LauditeRewardManager
from models.evaluator import LauditeEvaluator
from models.prompts import SYSTEM_PROMPT

# Carregar chaves de API (Gemini e WandB)
load_dotenv()

class LauditeEvalCallback(TrainerCallback):
    """
    Callback para executar o LauditeEvaluator periodicamente e logar no WandB.
    """
    def __init__(self, evaluator, val_dataset_path, eval_steps_interval=3):
        self.evaluator = evaluator
        self.val_dataset_path = val_dataset_path
        self.eval_steps_interval = eval_steps_interval
        self.checkpoint_count = 0

    def on_save(self, args, state, control, **kwargs):
        # O state.global_step e o número de saves ajudam a controlar o intervalo
        self.checkpoint_count += 1
        
        if self.checkpoint_count % self.eval_steps_interval == 0:
            print(f"\n[🔬 Checkpoint {state.global_step}] Iniciando Avaliação de Validação...")
            
            # Utiliza o evaluator.py que você já construiu
            metrics_df = self.evaluator.evaluate_dataset(
                dataset_path=self.val_dataset_path,
                output_path=f"eval_results_step_{state.global_step}.json"
            )
            
            # Log das métricas clínicas no WandB
            wandb.log({
                "clinical/judge_score": metrics_df['score_judge'].mean(),
                "clinical/similarity_score": metrics_df['score_similarity'].mean(),
                "clinical/xml_syntax_score": metrics_df['score_syntax'].mean(),
                "clinical/weighted_total": metrics_df['score_final_weighted'].mean(),
                "train/global_step": state.global_step
            })

def train():
    # 1. Configuração do WandB
    wandb.init(
        project="phi4-grpo-laudite",
        name="h100-cluster-run-v1",
        config={
            "base_model": "microsoft/phi-4",
            "learning_rate": 5e-6,
            "num_generations": 8,
            "gpu": "H100"
        }
    )

    # 2. Inicializar o Gestor de Recompensas
    # O device "cuda" em H100 garantirá que o Jina Embedding voe
    reward_manager = LauditeRewardManager(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        device="cuda"
    )

    # 3. Configuração de Treino (Otimizada para H100)
    training_args = GRPOConfig(
        output_dir="./checkpoints",
        learning_rate=5e-6,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        num_generations=8, # 'G' do GRPO: amostras por prompt
        max_prompt_length=1024,
        max_completion_length=1024,
        beta=0.04,
        bf16=True, # Nativo em H100 para máxima performance
        logging_steps=1,
        save_steps=100, # Frequência de checkpoint
        report_to="wandb",
        # O GRPO do TRL já loga automaticamente as funções de recompensa
    )

    # 4. Preparação do Dataset conforme seu formato
    def format_for_grpo(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["input"]}
            ],
            "label": example["label"], # Usado pela sua função de similaridade
            "extra": example["extra"]  # Usado pelo seu juiz (Gemini)
        }

    train_dataset = load_dataset(
        "json", 
        data_files="src/data/data/processed/train_dataset.json", 
        field="dataset"
    )["train"].map(format_for_grpo)

    # 5. Inicializar o Trainer
    # Passamos o dicionário 'extra' e 'label' para as funções de recompensa
    trainer = GRPOTrainer(
        model="microsoft/phi-4",
        reward_funcs=[
            reward_manager.xml_format_reward,        #
            reward_manager.semantic_similarity_reward, #
            reward_manager.judge_reward              #
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules="all-linear", # H100 aguenta treinar todas as projeções lineares
            task_type="CAUSAL_LM"
        )
    )

    # 6. Adicionar o Callback de Avaliação Clínica
    evaluator = LauditeEvaluator(
        model=trainer.model, 
        tokenizer=trainer.tokenizer, 
        reward_manager=reward_manager
    )
    
    trainer.add_callback(
        LauditeEvalCallback(
            evaluator=evaluator, 
            val_dataset_path="src/data/data/processed/val_dataset.json",
            eval_steps_interval=3 # A cada 3 checkpoints salvos
        )
    )

    print("Iniciando Treino em Cluster H100...")
    trainer.train()
    
    trainer.save_model("./phi4-laudite-final")

if __name__ == "__main__":
    train()