import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model_and_tokenizer(base_model_path, lora_adapter_path=None, device="cuda"):
    """
    Carrega o modelo Phi-4 e opcionalmente acopla o adaptador LoRA.
    """
    print(f"🔄 Carregando modelo base: {base_model_path}...")
    
    # Configuração para economizar VRAM se necessário (4bit/8bit)
    # Se tiver VRAM de sobra, remova o load_in_4bit
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True
    )
    # Phi-4 geralmente não tem pad_token definido por padrão
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_adapter_path:
        print(f"🔗 Acoplando adaptador LoRA: {lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload() # Opcional: funde pesos para inferência mais rápida

    model.eval() # Modo de avaliação
    return model, tokenizer