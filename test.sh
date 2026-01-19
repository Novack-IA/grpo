#!/bin/bash
#SBATCH --job-name=phi4_inference_test
#SBATCH --partition=gpu_h100            
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00                 
#SBATCH --output=logs/test_%j.out

# Carregar módulos do cluster
module load apptainer
module load cuda/12.8

# Variáveis de ambiente necessárias para o LauditeRewardManager e monitoramento
export GEMINI_API_KEY="SUA_CHAVE_AQUI"

# Execução do teste dentro do container
# O script run_test.py requer obrigatoriamente o --test_file
apptainer exec --nv \
    --bind .:/workspace \
    laudite_phi4.sif \
    python /workspace/src/models/run_test.py \
    --base_model "microsoft/phi-4" \
    --test_file "/workspace/src/data/data/processed/test_dataset.json" \
    --output_file "/workspace/results_test_phi4.json" \
    --batch_size 4