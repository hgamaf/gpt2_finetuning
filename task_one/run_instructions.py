import os
import json
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import torch
from datetime import datetime

def main():
    # Configurações
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    
    # Carregando dataset
    print("Carregando dataset de teste...")
    df_test = pd.read_parquet("../samples/test_dataset.parquet")
    print(f"Dataset carregado com {len(df_test)} exemplos")
    
    # Carregando modelo
    print("Carregando modelo GPT-2 small...")
    generator = pipeline(
        'text-generation',
        model="ComCom/gpt2-small",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Modelo carregado com sucesso!")
    
    # Gerando respostas
    print("\nGerando respostas para as instruções...")
    results = []
    
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        try:
            # Gerando resposta
            response = generator(
                row['instruction'],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                truncation=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Armazenando resultado
            result = {
                'instruction': row['instruction'],
                'input': row['input'],
                'original_output': row['output'],
                'generated_output': response[0]['generated_text'],
                'index': idx
            }
            results.append(result)
            
            # Salvando resultados parciais a cada 100 exemplos
            if (idx + 1) % 100 == 0:
                with open(f"results/generated_responses_{timestamp}_partial_{idx+1}.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nSalvando resultados parciais até o exemplo {idx+1}")
                
        except Exception as e:
            print(f"\nErro ao processar exemplo {idx}: {str(e)}")
            continue
    
    # Salvando resultados finais
    output_file = f"results/generated_responses_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcesso concluído!")
    print(f"Total de exemplos processados: {len(results)}")
    print(f"Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()