import json
import random
from pathlib import Path

def process_and_export(file_path, output_dir="data/processed"):
    # 1. Carregar o dataset original garantindo UTF-8
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Criar diretórios de saída
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    src_path = output_path.parent # Para salvar o prompts.py na raiz de src

    # 2. Extração e Limpeza de Encoding para os Prompts
    def fix_encoding(text):
        if not text: return ""
        # Caso o texto tenha sido lido como Latin-1 mas seja UTF-8, isso corrige
        try:
            return text.encode('latin-1').decode('utf-8')
        except:
            return text

    sys_prompt = fix_encoding(raw_data['prompts']['generator'])
    eval_prompt = fix_encoding(raw_data['prompts']['evaluator'])

    with open(src_path / "prompts.py", "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n\n")
        f.write(f'SYSTEM_PROMPT = """{sys_prompt}"""\n\n')
        f.write(f'JUDGE_PROMPT = """{eval_prompt}"""\n')

    # 4. Preparar o Dataset (Mantendo a estrutura original por ID)
    examples = raw_data['dataset']
    random.seed(42)
    random.shuffle(examples)

    total = len(examples)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:]
    }

    # 5. Salvar os arquivos mantendo a chave "dataset"
    for name, items in splits.items():
        output_struct = {"dataset": items}
        file_name = output_path / f"{name}_dataset.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(output_struct, f, indent=4, ensure_ascii=False)

    return total, len(splits['train']), len(splits['val']), len(splits['test'])

if __name__ == "__main__":
    # Ajuste o nome do arquivo conforme ele estiver na sua home
    file_name = "grpo_dataset.json" 
    try:
        total, tr, vl, ts = process_and_export(file_name)
        print(f"✅ Processamento concluído com sucesso!")
        print(f"Total: {total} | Treino: {tr} | Val: {vl} | Teste: {ts}")
        print(f"Prompts salvos em: src/prompts.py")
        print(f"Datasets salvos em: data/processed/")
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo {file_name} não foi encontrado.")