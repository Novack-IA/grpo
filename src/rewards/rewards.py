import re
import json
import asyncio
import torch
from typing import List, Dict
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

class LauditeRewardManager:
    def __init__(self, gemini_api_key: str, device: str = "cuda"):
        # Configuração do Gemini Judge
        genai.configure(api_key=gemini_api_key)
        self.judge_model = genai.GenerativeModel("gemini-2.5-pro")
        self.generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0  # Determinístico
        )
        
        # Modelo de Embedding para Similaridade Semântica (Local)
        # jina-embeddings-v2-base-pt: 8k context window, SOTA em PT-BR
        print("Carregando modelo de embedding para recompensa local...")
        self.embed_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-pt", trust_remote_code=True)
        self.embed_model.to(device)
        
        # Regex para capturar o conteúdo dentro de <final-text>
        # A flag DOTALL (.) permite capturar quebras de linha
        self.xml_pattern = re.compile(r"<final-text>(.*?)</final-text>", re.DOTALL)
        self.checklist_pattern = re.compile(r"<previous-errors>(.*?)</previous-errors>", re.DOTALL)

    def _extract_report(self, text: str) -> str:
        """Extrai apenas o texto do laudo, ignorando o checklist e as tags XML."""
        match = self.xml_pattern.search(text)
        if match:
            return match.group(1).strip()
        return "" # Retorna vazio se falhar no parse

    def _extract_checklist(self, text: str) -> str:
        """Extrai o checklist para validação se necessário."""
        match = self.checklist_pattern.search(text)
        if match:
            return match.group(1).strip()
        return ""

    # =========================================================================
    # 1. RECOMPENSA SINTÁTICA (XML & ESTRUTURA)
    # =========================================================================
    def xml_format_reward(self, completions: List[str], **kwargs) -> List[float]:
        """
        Garante que o Phi-4 respeite estritamente o formato XML de saída.
        Crucial para que o sistema em produção consiga fazer o parse.
        """
        rewards = []
        for completion in completions:
            score = 0.0
            
            # Verifica presença das tags de abertura e fechamento
            has_open_final = "<final-text>" in completion
            has_close_final = "</final-text>" in completion
            has_open_errors = "<previous-errors>" in completion
            has_close_errors = "</previous-errors>" in completion
            
            if has_open_final and has_close_final:
                score += 0.5
            
            if has_open_errors and has_close_errors:
                score += 0.3
                
            # Verifica se extração funciona
            content = self._extract_report(completion)
            if content and len(content) > 10: # Evita tags vazias
                score += 0.2
                
            rewards.append(score)
        return rewards

    # =========================================================================
    # 2. RECOMPENSA DE SIMILARIDADE SEMÂNTICA (EMBEDDING LOCAL)
    # =========================================================================
    def semantic_similarity_reward(self, prompts: List[str], completions: List[str], label: List[str], **kwargs) -> List[float]:
        """
        Calcula a similaridade de cosseno entre o laudo gerado e o label (Gold Standard).
        Usa Jina V2 para suportar contextos longos.
        """
        # Extrai apenas o laudo limpo da geração do modelo
        clean_completions = [self._extract_report(c) for c in completions]
        
        # Se a extração falhar (string vazia), penalidade ou score 0.
        # Aqui atribuímos 0 para não quebrar o batch, mas o xml_format_reward já penalizou.
        
        # Gera embeddings em batch (muito mais rápido)
        with torch.no_grad():
            gen_embeddings = self.embed_model.encode(clean_completions, convert_to_tensor=True, normalize_embeddings=True)
            ref_embeddings = self.embed_model.encode(label, convert_to_tensor=True, normalize_embeddings=True)
            
        # Calcula Cosseno Similaridade (dot product de vetores normalizados)
        # O resultado é um tensor [batch_size], convertemos para lista
        scores = torch.sum(gen_embeddings * ref_embeddings, dim=1).cpu().tolist()
        
        # Normalização de segurança (clipping entre 0 e 1, embora cosine já seja -1 a 1)
        return [max(0.0, score) for score in scores]

    # =========================================================================
    # 3. RECOMPENSA DO JUIZ (GEMINI 2.5 PRO)
    # =========================================================================
    async def _query_gemini(self, prompt_template: str, inputs: Dict, completion: str, label: str) -> float:
        """Lógica unitária de chamada ao Gemini."""
        
        clean_completion = self._extract_report(completion)
        
        # Se o modelo não gerou o XML correto, não gastamos API e damos nota 0
        if not clean_completion:
            return 0.0

        # Montagem do Prompt Dinâmico
        # ATENÇÃO: Mapeando os campos do seu dataset 'extra' para o prompt
        # inputs aqui refere-se ao dicionário 'extra' daquela linha específica
        prompt = prompt_template.replace("<INITIAL-TEXT>", inputs.get("initial_text", ""))
        prompt = prompt.replace("<USER-MESSAGE>", inputs.get("user_message", ""))
        prompt = prompt.replace("<LABEL-RESPONSE>", label)
        prompt = prompt.replace("<LLM-RESPONSE>", clean_completion)
        
        # Campos opcionais
        prompt = prompt.replace("<USER-CONTEXT>", inputs.get("user_context", "") or "N/A")
        prompt = prompt.replace("<PREVIEW-REPORT-USER-EXAMPLE>", inputs.get("preview_report_user_example", "") or "N/A")
        prompt = prompt.replace("<DEFAULT-CONTEXT>", inputs.get("default_context", "") or "N/A")

        try:
            # Retry simples interno para erros de rede transientes
            for attempt in range(2):
                try:
                    response = await self.judge_model.generate_content_async(
                        prompt, 
                        generation_config=self.generation_config
                    )
                    data = json.loads(response.text)
                    
                    # Lógica de Pontuação
                    summary = data.get("final_summary", {})
                    rubric = data.get("scoring_rubric", {})
                    
                    # Checagem de Alucinação (Punição Severa)
                    safety = rubric.get("safety_check", {}).get("achieved_points", {}).get("no_hallucination", {}).get("points", 0)
                    if safety < 0:
                        return -2.0 # Punição forte relativa (GRPO normaliza, então -2 é bem ruim)

                    # Tenta pegar o total score direto
                    if summary.get("total_score") is not None:
                        # Normaliza de 0-10 para 0-1 (ajuda na estabilidade do gradiente)
                        return float(summary["total_score"]) / 10.0
                    
                    # Fallback: Soma manual
                    score = 0
                    score += rubric["conclusion_comparison"]["achieved_points"]["content_match"]["points"]
                    score += rubric["conclusion_comparison"]["achieved_points"]["conciseness_and_clarity_match"]["points"]
                    score += rubric["report_body_comparison"]["achieved_points"]["accuracy_match"]["points"]
                    score += rubric["report_body_comparison"]["achieved_points"]["localization_match"]["points"]
                    score += rubric["formatting_comparison"]["achieved_points"]["preserves_markdown"]["points"]
                    
                    return float(score) / 10.0
                    
                except Exception as e:
                    if attempt == 1: print(f"Erro no Gemini (tentativa {attempt}): {e}")
                    await asyncio.sleep(1) # Backoff simples
            
            return 0.0 # Falha na API = Sem recompensa (neutro)
            
        except Exception:
            return 0.0

    def judge_reward(self, prompts: List[str], completions: List[str], label: List[str], extra: List[Dict], **kwargs) -> List[float]:
        """
        Wrapper síncrono que dispara avaliações paralelas para o batch.
        TRL passa colunas do dataset como listas (kwargs['extra'] virá como lista de dicts).
        """
        
        from models.prompts import JUDGE_PROMPT 
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        for i, completion in enumerate(completions):
            # extra[i] contém os metadados daquela amostra (initial_text, user_message, etc)
            tasks.append(self._query_gemini(JUDGE_PROMPT, extra[i], completion, label[i]))
            
        scores = loop.run_until_complete(asyncio.gather(*tasks))
        return list(scores)