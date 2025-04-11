import os
import json
import pandas as pd
from openai import OpenAI
from typing import Dict, List
import logging
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseEvaluator:
    def __init__(self, api_key: str):
        """Inicializa o avaliador com a chave da API da OpenAI."""
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """Você é um avaliador especializado em comparar respostas geradas por modelos de linguagem.
        Sua tarefa é avaliar se a resposta gerada está de acordo com a resposta original, considerando:
        1. Semântica e significado
        2. Completude da informação
        3. Clareza e coesão
        4. Relevância para a instrução
        
        Responda com uma avaliação detalhada e uma nota de 0 a 10."""
        
    def evaluate_response(self, instruction: str, original_output: str, generated_output: str) -> Dict:
        """Avalia uma única resposta gerada."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""
                    Instrução: {instruction}
                    
                    Resposta Original: {original_output}
                    
                    Resposta Gerada: {generated_output}
                    
                    Por favor, avalie a resposta gerada em comparação com a original."""
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            evaluation = response.choices[0].message.content
            return {
                "evaluation": evaluation,
                "instruction": instruction,
                "original_output": original_output,
                "generated_output": generated_output
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar resposta: {e}")
            return None

    def evaluate_batch(self, 
                      instructions: List[str], 
                      original_outputs: List[str], 
                      generated_outputs: List[str],
                      output_file: str = "evaluation_results.json") -> None:
        """Avalia um lote de respostas e salva os resultados."""
        results = []
        
        for i in tqdm(range(len(instructions)), desc="Avaliando respostas"):
            result = self.evaluate_response(
                instructions[i],
                original_outputs[i],
                generated_outputs[i]
            )
            if result:
                results.append(result)
                
        # Salvar resultados
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Resultados salvos em {output_file}")

def main():
    # Carregar o dataset de teste
    test_dataset_path = "../../samples/test_dataset.parquet"
    test_df = pd.read_parquet(test_dataset_path)
    
    # Carregar as respostas geradas
    generated_responses_path = "generated_responses.json"
    with open(generated_responses_path, 'r') as f:
        generated_responses = json.load(f)
    
    # Inicializar avaliador
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida")
        
    evaluator = ResponseEvaluator(api_key)
    
    # Avaliar respostas
    evaluator.evaluate_batch(
        test_df['instruction'].tolist(),
        test_df['output'].tolist(),
        generated_responses,
        "evaluation_results.json"
    )

if __name__ == "__main__":
    main() 