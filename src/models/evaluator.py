import torch
import json
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
from prompts import SYSTEM_PROMPT
from rewards import LauditeRewardManager

class LauditeEvaluator:
    def __init__(self, model, tokenizer, reward_manager: LauditeRewardManager, batch_size=4):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_manager = reward_manager
        self.batch_size = batch_size

    def _prepare_prompts(self, inputs: List[str]) -> List[str]:
        """Encapsula o input do dataset com o System Prompt do Phi-4."""
        formatted_prompts = []
        for inp in inputs:
            # Formato ChatML ou Raw do Phi-4. 
            # Ajuste conforme o template que você usou no treino.
            # Exemplo genérico user/assistant:
            text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{inp}<|end|>\n<|assistant|>\n"
            formatted_prompts.append(text)
        return formatted_prompts

    def evaluate_dataset(self, dataset_path: str, output_path: str = None):
        """
        Roda inferência e avaliação em um arquivo JSON de dataset.
        Retorna um DataFrame com os resultados detalhados.
        """
        # 1. Carregar Dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        items = data['dataset'] # Acessa a lista dentro da chave 'dataset'
        results = []

        print(f"🚀 Iniciando avaliação de {len(items)} itens...")

        # Processar em Batches
        for i in tqdm(range(0, len(items), self.batch_size)):
            batch_items = items[i : i + self.batch_size]
            
            # Preparar dados para o Reward Manager e Modelo
            inputs = [item['input'] for item in batch_items]
            labels = [item['label'] for item in batch_items]
            extras = [item['extra'] for item in batch_items] # Dicts com metadados
            
            # 2. Inferência (Geração)
            prompts = self._prepare_prompts(inputs)
            
            inputs_tokenized = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            with torch.no_grad():
                outputs_ids = self.model.generate(
                    **inputs_tokenized,
                    max_new_tokens=1024,
                    temperature=0.1, # Baixa temperatura para teste
                    do_sample=True
                )
            
            # Decodificar e remover o prompt de entrada da saída
            completions = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
            # Hack simples para remover o prompt se o skip_special_tokens não limpar tudo do chat template
            clean_completions = []
            for prompt, comp in zip(prompts, completions):
                # Remove a parte do prompt se ela aparecer repetida (comum em alguns setups)
                # No Phi-4 idealmente usamos apenas o texto gerado após a última tag
                clean_completions.append(comp.split("<|assistant|>")[-1].strip())

            # 3. Avaliação (Chama o Judge + Embedding + Syntax)
            # Reutilizamos as funções do reward_manager, mas agora queremos os valores brutos
            
            # Nota do Juiz (Gemini)
            judge_scores = self.reward_manager.judge_reward(prompts, clean_completions, labels, extras)
            
            # Nota Sintática (XML)
            syntax_scores = self.reward_manager.xml_format_reward(clean_completions)
            
            # Nota Semântica (Embedding)
            sim_scores = self.reward_manager.semantic_similarity_reward(prompts, clean_completions, labels)

            # 4. Compilar Resultados do Batch
            for j, item in enumerate(batch_items):
                # Extrai o texto final limpo (sem tags XML) para salvar no log
                final_text_content = self.reward_manager._extract_report(clean_completions[j])
                
                result_entry = {
                    "id": item.get('id'),
                    "input": item['input'],
                    "label": labels[j],
                    "generated_full": clean_completions[j],
                    "generated_content": final_text_content, # O laudo limpo
                    "score_judge": judge_scores[j],
                    "score_syntax": syntax_scores[j],
                    "score_similarity": sim_scores[j],
                    # Média ponderada simples para uma visão geral
                    "score_final_weighted": (judge_scores[j] * 0.7) + (syntax_scores[j] * 0.15) + (sim_scores[j] * 0.15)
                }
                results.append(result_entry)

        # 5. Análise Final
        df = pd.DataFrame(results)
        print("\n=== 📊 Relatório de Avaliação ===")
        print(f"Média Judge (Gemini): {df['score_judge'].mean():.4f}")
        print(f"Média Similaridade:   {df['score_similarity'].mean():.4f}")
        print(f"Aderência ao Formato: {df['score_syntax'].mean():.4f}")
        
        # Salvar
        if output_path:
            df.to_json(output_path, orient='records', indent=2, force_ascii=False)
            print(f"💾 Resultados salvos em: {output_path}")

        return df