#!/bin/bash
#SBATCH --job-name=phi4_grpo_laudite
#SBATCH --partition=gpu_h100             # Nome da partição de H100 do seu cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # Reserva 1 GPU H100
#SBATCH --cpus-per-task=16              # CPUs para processamento de dados e embedding
#SBATCH --mem=128G                      # Memória RAM do host
#SBATCH --time=24:00:00                 # Tempo máximo de execução
#SBATCH --output=logs/train_%j.out      # Arquivo de log

# Carrega módulos necessários (comum em clusters universitários/corporativos)
module load apptainer
module load cuda/12.8                   # Versão recomendada para H100

# Variáveis de ambiente
export WANDB_API_KEY="SUA_CHAVE_AQUI"
export GEMINI_API_KEY="SUA_CHAVE_AQUI"

# Executa o treinamento dentro do container
# O parâmetro --nv é obrigatório para o Apptainer acessar a GPU do host
apptainer exec --nv \
    --bind .:/workspace \
    laudite_phi4.sif \
    python /workspace/src/train.py