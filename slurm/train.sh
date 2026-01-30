#!/bin/bash
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --job-name=phi4-grpo-native
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- Configuração de Caminhos ---
export USER_ROOT="/raid/user_gustavonovack"
export CACHE_ROOT="${USER_ROOT}/cache"
export PROJ_ROOT="${USER_ROOT}/grpo"
export IMAGE_PATH="${USER_ROOT}/images/vllm-openai_latest.sif"

export HF_HOME="${CACHE_ROOT}/hf_home"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip_cache"
export TORCH_HOME="${CACHE_ROOT}/torch"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

mkdir -p $HF_HOME $PIP_CACHE_DIR $TORCH_HOME $APPTAINER_CACHEDIR

echo "Iniciando Job $SLURM_JOB_ID no node $SLURMD_NODENAME"

srun apptainer exec \
    --nv \
    --bind "${USER_ROOT}:${USER_ROOT}" \
    --bind "/tmp:/tmp" \
    --env HF_HOME="${HF_HOME}" \
    --env PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
    "${IMAGE_PATH}" \
    bash -c "
        cd ${PROJ_ROOT}
        export PATH=\$HOME/.local/bin:\$PATH
        
        # Instala deps
        python3 -m pip install --user --upgrade pip > /dev/null
        # Adicionei 'certifi' explicitamente aqui
        python3 -m pip install --user trl peft wandb python-dotenv google-genai datasets bitsandbytes certifi > /dev/null

        export PYTHONPATH=${PROJ_ROOT}/src:\$PYTHONPATH
        
        # --- A CORREÇÃO DO SSL ---
        echo '--- Configurando SSL para o Gemini ---'
        # Descobre onde o certifi instalou o arquivo .pem e exporta para as variáveis de ambiente
        export SSL_CERT_FILE=\$(python3 -c 'import certifi; print(certifi.where())')
        export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
        export WEBSOCKET_CLIENT_CA_BUNDLE=\$SSL_CERT_FILE
        # -------------------------
        
        echo '--- Iniciando Treino ---'
        python3 src/train_grpo.py
    "