import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_generated_responses(results_dir="results"):
    """Carrega os resultados gerados mais recentes"""
    # Lista todos os arquivos JSON na pasta results
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and not 'partial' in f]
    if not json_files:
        raise FileNotFoundError("Nenhum arquivo de resultados encontrado")
    
    # Pega o arquivo mais recente
    latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    print(f"Carregando resultados de: {latest_file}")
    
    with open(os.path.join(results_dir, latest_file), 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_bleu_score(reference, candidate):
    """Calcula o BLEU score entre duas sentenças"""
    reference = [reference.split()]
    candidate = candidate.split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

def calculate_rouge_scores(reference, candidate):
    """Calcula os scores ROUGE entre duas sentenças"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }
    except:
        return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}

def calculate_cosine_similarity(reference, candidate, model):
    """Calcula similaridade de cosseno usando embeddings"""
    embeddings = model.encode([reference, candidate])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def analyze_lengths(responses):
    """Analisa os comprimentos das respostas"""
    original_lengths = [len(r['original_output'].split()) for r in responses]
    generated_lengths = [len(r['generated_output'].split()) for r in responses]
    
    return {
        'original': {
            'mean': np.mean(original_lengths),
            'median': np.median(original_lengths),
            'std': np.std(original_lengths),
            'min': min(original_lengths),
            'max': max(original_lengths)
        },
        'generated': {
            'mean': np.mean(generated_lengths),
            'median': np.median(generated_lengths),
            'std': np.std(generated_lengths),
            'min': min(generated_lengths),
            'max': max(generated_lengths)
        }
    }

def generate_report(responses, metrics, output_dir):
    """Gera relatório completo da avaliação"""
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_examples': len(responses),
        'metrics': metrics,
        'examples': {
            'best_bleu': max(responses, key=lambda x: x['metrics']['bleu']),
            'worst_bleu': min(responses, key=lambda x: x['metrics']['bleu']),
            'best_rouge': max(responses, key=lambda x: x['metrics']['rouge']['rouge-l']),
            'worst_rouge': min(responses, key=lambda x: x['metrics']['rouge']['rouge-l'])
        }
    }
    
    # Salvando relatório
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def main():
    # Criando diretório de saída
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregando resultados
    print("Carregando resultados gerados...")
    responses = load_generated_responses()
    
    # Carregando modelo para embeddings
    print("Carregando modelo para embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculando métricas para cada resposta
    print("\nCalculando métricas...")
    for response in tqdm(responses):
        metrics = {
            'bleu': calculate_bleu_score(response['original_output'], response['generated_output']),
            'rouge': calculate_rouge_scores(response['original_output'], response['generated_output']),
            'cosine_similarity': calculate_cosine_similarity(
                response['original_output'],
                response['generated_output'],
                model
            )
        }
        response['metrics'] = metrics
    
    # Análise de comprimentos
    print("\nAnalisando comprimentos...")
    length_stats = analyze_lengths(responses)
    
    # Gerando relatório final
    print("\nGerando relatório...")
    metrics_summary = {
        'bleu': {
            'mean': np.mean([r['metrics']['bleu'] for r in responses]),
            'median': np.median([r['metrics']['bleu'] for r in responses]),
            'std': np.std([r['metrics']['bleu'] for r in responses])
        },
        'rouge': {
            'rouge-1': {
                'mean': np.mean([r['metrics']['rouge']['rouge-1'] for r in responses]),
                'median': np.median([r['metrics']['rouge']['rouge-1'] for r in responses])
            },
            'rouge-2': {
                'mean': np.mean([r['metrics']['rouge']['rouge-2'] for r in responses]),
                'median': np.median([r['metrics']['rouge']['rouge-2'] for r in responses])
            },
            'rouge-l': {
                'mean': np.mean([r['metrics']['rouge']['rouge-l'] for r in responses]),
                'median': np.median([r['metrics']['rouge']['rouge-l'] for r in responses])
            }
        },
        'cosine_similarity': {
            'mean': np.mean([r['metrics']['cosine_similarity'] for r in responses]),
            'median': np.median([r['metrics']['cosine_similarity'] for r in responses]),
            'std': np.std([r['metrics']['cosine_similarity'] for r in responses])
        },
        'length_stats': length_stats
    }
    
    report = generate_report(responses, metrics_summary, output_dir)
    
    print("\nAvaliação concluída!")
    print(f"Relatório salvo em: {os.path.join(output_dir, 'evaluation_report.json')}")

if __name__ == "__main__":
    main() 