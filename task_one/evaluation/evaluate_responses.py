import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_generated_responses(results_dir="../results"):
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

def interpret_metrics(metrics):
    """Interpreta as métricas e fornece uma avaliação qualitativa"""
    def interpret_bleu(score):
        if score >= 0.6:
            return "BOM", "Alta similaridade com a resposta original"
        elif score >= 0.3:
            return "MÉDIO", "Similaridade moderada com a resposta original"
        else:
            return "RUIM", "Baixa similaridade com a resposta original"
    
    def interpret_rouge(score):
        if score >= 0.7:
            return "BOM", "Alta sobreposição com a resposta original"
        elif score >= 0.4:
            return "MÉDIO", "Sobreposição moderada com a resposta original"
        else:
            return "RUIM", "Baixa sobreposição com a resposta original"
    
    def interpret_cosine(similarity):
        if similarity >= 0.8:
            return "BOM", "Alta similaridade semântica com a resposta original"
        elif similarity >= 0.5:
            return "MÉDIO", "Similaridade semântica moderada com a resposta original"
        else:
            return "RUIM", "Baixa similaridade semântica com a resposta original"
    
    def interpret_length_ratio(original_mean, generated_mean):
        ratio = generated_mean / original_mean
        if 0.8 <= ratio <= 1.2:
            return "BOM", "Comprimento similar ao original"
        elif 0.5 <= ratio <= 1.5:
            return "MÉDIO", "Comprimento moderadamente diferente do original"
        else:
            return "RUIM", "Comprimento muito diferente do original"
    
    return {
        'bleu': {
            'score': metrics['bleu']['mean'],
            'interpretation': interpret_bleu(metrics['bleu']['mean'])
        },
        'rouge': {
            'rouge-1': {
                'score': metrics['rouge']['rouge-1']['mean'],
                'interpretation': interpret_rouge(metrics['rouge']['rouge-1']['mean'])
            },
            'rouge-2': {
                'score': metrics['rouge']['rouge-2']['mean'],
                'interpretation': interpret_rouge(metrics['rouge']['rouge-2']['mean'])
            },
            'rouge-l': {
                'score': metrics['rouge']['rouge-l']['mean'],
                'interpretation': interpret_rouge(metrics['rouge']['rouge-l']['mean'])
            }
        },
        'cosine_similarity': {
            'score': metrics['cosine_similarity']['mean'],
            'interpretation': interpret_cosine(metrics['cosine_similarity']['mean'])
        },
        'length_ratio': {
            'score': metrics['length_stats']['generated']['mean'] / metrics['length_stats']['original']['mean'],
            'interpretation': interpret_length_ratio(
                metrics['length_stats']['original']['mean'],
                metrics['length_stats']['generated']['mean']
            )
        }
    }

def generate_report(responses, metrics, output_dir):
    """Gera relatório completo da avaliação"""
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_examples': len(responses),
        'metrics': metrics,
        'interpretation': interpret_metrics(metrics),
        'examples': {
            'best_bleu': {
                'instruction': responses[max(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['instruction'],
                'original_output': responses[max(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['original_output'],
                'generated_output': responses[max(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['generated_output'],
                'metrics': responses[max(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['metrics']
            },
            'worst_bleu': {
                'instruction': responses[min(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['instruction'],
                'original_output': responses[min(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['original_output'],
                'generated_output': responses[min(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['generated_output'],
                'metrics': responses[min(range(len(responses)), key=lambda i: responses[i]['metrics']['bleu'])]['metrics']
            }
        }
    }
    
    # Salvando relatório
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    
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
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculando métricas para cada resposta
    print("\nCalculando métricas...")
    for response in tqdm(responses):
        metrics = {
            'bleu': calculate_bleu_score(response['original_output'], response['generated_output']),
            'rouge': calculate_rouge_scores(response['original_output'], response['generated_output']),
            'cosine_similarity': calculate_cosine_similarity(
                response['original_output'],
                response['generated_output'],
                sentence_model
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