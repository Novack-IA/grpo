import os
import argparse
from model_utils import load_model_and_tokenizer
from evaluator import LauditeEvaluator
from rewards import LauditeRewardManager
from dotenv import load_dotenv

# Carrega API KEY do .env
load_dotenv() 

def main():
    parser = argparse.ArgumentParser(description="Script de Inferência e Avaliação do Phi-4 Laudite")
    
    parser.add_argument("--base_model", type=str, default="microsoft/phi-4", help="Modelo base")
    parser.add_argument("--adapter_path", type=str, default=None, help="Caminho do checkpoint LoRA (ex: ./checkpoints/checkpoint-500)")
    parser.add_argument("--test_file", type=str, required=True, help="Arquivo JSON de teste (test.json)")
    parser.add_argument("--output_file", type=str, default="results_evaluation.json", help="Onde salvar os resultados")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size para inferência")
    
    args = parser.parse_args()

    # 1. Carregar Modelo
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

    # 2. Inicializar Gerenciador de Recompensas (Judge)
    # Certifique-se que GEMINI_API_KEY está no ambiente
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("ERRO: GEMINI_API_KEY não encontrada nas variáveis de ambiente.")

    reward_manager = LauditeRewardManager(gemini_api_key=api_key)

    # 3. Inicializar Avaliador
    evaluator = LauditeEvaluator(
        model=model, 
        tokenizer=tokenizer, 
        reward_manager=reward_manager,
        batch_size=args.batch_size
    )

    # 4. Rodar Avaliação
    evaluator.evaluate_dataset(args.test_file, args.output_file)

if __name__ == "__main__":
    main()