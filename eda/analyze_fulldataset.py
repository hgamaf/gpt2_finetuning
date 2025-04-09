from datasets import load_dataset
import os
import pandas as pd
from collections import Counter
import re
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

# Download necessário para o NLTK
nltk.download('punkt')

# Criando diretório para salvar as análises
os.makedirs("stats", exist_ok=True)

def load_and_split_dataset():
    """Carrega o dataset Alpaca e divide em treino e teste"""
    # Criando diretório para salvar os datasets
    os.makedirs("../samples", exist_ok=True)
    
    # Carregando dataset completo
    dataset = load_dataset("tatsu-lab/alpaca")
    df_full = pd.DataFrame(dataset['train'])
    
    # Criando colunas para estratificação
    df_full['instruction_length'] = df_full['instruction'].str.len()
    df_full['has_input'] = df_full['input'].str.len() > 0
    df_full['length_cat'] = pd.qcut(df_full['instruction_length'], q=4, labels=False)
    
    # Dividindo o dataset em treino e teste
    df_train, df_test = train_test_split(
        df_full,
        test_size=3000,
        train_size=3000,
        random_state=42,
        stratify=df_full[['length_cat', 'has_input']]
    )
    
    # Salvando datasets em parquet
    df_train.to_parquet("../samples/train_dataset.parquet", index=False)
    df_test.to_parquet("../samples/test_dataset.parquet", index=False)
    
    # Salvando estatísticas da amostragem
    with open("stats/sampling_stats.txt", "w") as f:
        f.write("=== Estatísticas da Amostragem ===\n\n")
        
        # Estatísticas do dataset completo
        f.write("Dataset Completo:\n")
        f.write(f"Total de exemplos: {len(df_full):,}\n")
        f.write(f"Exemplos com input: {df_full['has_input'].sum():,} ({df_full['has_input'].mean()*100:.1f}%)\n")
        f.write(f"Exemplos sem input: {(~df_full['has_input']).sum():,} ({(~df_full['has_input']).mean()*100:.1f}%)\n\n")
        
        # Estatísticas do treino
        f.write("Dataset de Treino:\n")
        f.write(f"Total de exemplos: {len(df_train):,}\n")
        f.write(f"Exemplos com input: {df_train['has_input'].sum():,} ({df_train['has_input'].mean()*100:.1f}%)\n")
        f.write(f"Exemplos sem input: {(~df_train['has_input']).sum():,} ({(~df_train['has_input']).mean()*100:.1f}%)\n\n")
        
        # Estatísticas do teste
        f.write("Dataset de Teste:\n")
        f.write(f"Total de exemplos: {len(df_test):,}\n")
        f.write(f"Exemplos com input: {df_test['has_input'].sum():,} ({df_test['has_input'].mean()*100:.1f}%)\n")
        f.write(f"Exemplos sem input: {(~df_test['has_input']).sum():,} ({(~df_test['has_input']).mean()*100:.1f}%)\n\n")
        
        # Distribuição por quartil
        f.write("Distribuição por Quartil de Comprimento:\n")
        for i in range(4):
            f.write(f"\nQuartil {i+1}:\n")
            f.write(f"Dataset completo: {len(df_full[df_full['length_cat'] == i]):,} ({len(df_full[df_full['length_cat'] == i])/len(df_full)*100:.1f}%)\n")
            f.write(f"Treino: {len(df_train[df_train['length_cat'] == i]):,} ({len(df_train[df_train['length_cat'] == i])/len(df_train)*100:.1f}%)\n")
            f.write(f"Teste: {len(df_test[df_test['length_cat'] == i]):,} ({len(df_test[df_test['length_cat'] == i])/len(df_test)*100:.1f}%)\n")
    
    return df_full, df_train, df_test

def analyze_text_lengths(df, dataset_name):
    """Análise de comprimentos de texto"""
    stats = {
        "instruction": {
            "mean": df['instruction'].str.len().mean(),
            "median": df['instruction'].str.len().median(),
            "std": df['instruction'].str.len().std(),
            "min": df['instruction'].str.len().min(),
            "max": df['instruction'].str.len().max()
        },
        "output": {
            "mean": df['output'].str.len().mean(),
            "median": df['output'].str.len().median(),
            "std": df['output'].str.len().std(),
            "min": df['output'].str.len().min(),
            "max": df['output'].str.len().max()
        },
        "input": {
            "mean": df['input'].str.len().mean(),
            "median": df['input'].str.len().median(),
            "std": df['input'].str.len().std(),
            "min": df['input'].str.len().min(),
            "max": df['input'].str.len().max()
        }
    }
    return stats

def analyze_input_presence(df, dataset_name):
    """Análise da presença de inputs"""
    input_stats = {
        "total_examples": len(df),
        "with_input": len(df[df['input'].str.len() > 0]),
        "without_input": len(df[df['input'].str.len() == 0]),
        "input_percentage": (len(df[df['input'].str.len() > 0]) / len(df)) * 100
    }
    return input_stats

def analyze_instruction_types(df, dataset_name):
    """Análise de tipos de instruções"""
    df['instruction_verb'] = df['instruction'].apply(
        lambda x: word_tokenize(x.lower())[0] if len(word_tokenize(x)) > 0 else ''
    )
    verb_counts = Counter(df['instruction_verb'])
    top_verbs = dict(verb_counts.most_common(5))  # Top 5 verbos
    return top_verbs

def analyze_readability(df, dataset_name):
    """Análise de legibilidade usando Flesch Reading Ease"""
    df['flesch_score'] = df['instruction'].apply(flesch_reading_ease)
    readability_stats = {
        "mean_score": df['flesch_score'].mean(),
        "median_score": df['flesch_score'].median(),
        "std_score": df['flesch_score'].std(),
        "min_score": df['flesch_score'].min(),
        "max_score": df['flesch_score'].max()
    }
    return readability_stats

def analyze_topics(df, dataset_name, n_topics=5):
    """Análise de tópicos usando LDA"""
    # Preparando o texto para análise
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(df['instruction'])
    
    # Aplicando LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Extraindo palavras mais importantes para cada tópico
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-6:-1]  # Top 5 palavras
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            "topic_id": topic_idx,
            "top_words": top_words
        })
    
    return topics

def save_analysis_results(all_stats, dataset_name):
    """Salvando resultados da análise"""
    with open(f'stats/analysis_results_{dataset_name}.txt', 'w') as f:
        f.write(f"=== Análise do Dataset Alpaca - {dataset_name} ===\n\n")
        
        f.write("1. Análise de Comprimentos de Texto:\n")
        for field, stats in all_stats['text_lengths'].items():
            f.write(f"\n{field.upper()}:\n")
            for stat, value in stats.items():
                f.write(f"{stat}: {value:.2f}\n")
        
        f.write("\n2. Análise de Inputs:\n")
        for stat, value in all_stats['input_presence'].items():
            f.write(f"{stat}: {value}\n")
        
        f.write("\n3. Top 5 Verbos nas Instruções:\n")
        for verb, count in all_stats['instruction_types'].items():
            f.write(f"{verb}: {count}\n")
        
        f.write("\n4. Análise de Legibilidade (Flesch Reading Ease):\n")
        for stat, value in all_stats['readability'].items():
            f.write(f"{stat}: {value:.2f}\n")
        
        f.write("\n5. Análise de Tópicos (LDA):\n")
        for topic in all_stats['topics']:
            f.write(f"\nTópico {topic['topic_id']}:\n")
            f.write(f"Palavras mais importantes: {', '.join(topic['top_words'])}\n")

def print_detailed_stats(df, dataset_name):
    """Imprime estatísticas detalhadas sobre o dataset"""
    print(f"\n=== Estatísticas Detalhadas do Dataset - {dataset_name} ===")
    print(f"\nTotal de exemplos: {len(df):,}")
    
    # Comprimentos de texto
    print("\nComprimentos de Texto:")
    print(f"Instruções - Média: {df['instruction'].str.len().mean():.2f} caracteres")
    print(f"Instruções - Mediana: {df['instruction'].str.len().median():.2f} caracteres")
    print(f"Instruções - Mínimo: {df['instruction'].str.len().min()} caracteres")
    print(f"Instruções - Máximo: {df['instruction'].str.len().max()} caracteres")
    
    print(f"\nRespostas - Média: {df['output'].str.len().mean():.2f} caracteres")
    print(f"Respostas - Mediana: {df['output'].str.len().median():.2f} caracteres")
    print(f"Respostas - Mínimo: {df['output'].str.len().min()} caracteres")
    print(f"Respostas - Máximo: {df['output'].str.len().max()} caracteres")
    
    # Análise de inputs
    input_stats = analyze_input_presence(df, dataset_name)
    print("\nAnálise de Inputs:")
    print(f"Exemplos com input: {input_stats['with_input']:,} ({input_stats['input_percentage']:.1f}%)")
    print(f"Exemplos sem input: {input_stats['without_input']:,} ({100 - input_stats['input_percentage']:.1f}%)")
    
    # Análise de verbos
    top_verbs = analyze_instruction_types(df, dataset_name)
    print("\nTop 5 Verbos mais Comuns nas Instruções:")
    for verb, count in top_verbs.items():
        print(f"{verb}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Análise de legibilidade
    readability = analyze_readability(df, dataset_name)
    print("\nAnálise de Legibilidade (Flesch Reading Ease):")
    print(f"Média: {readability['mean_score']:.2f}")
    print(f"Mediana: {readability['median_score']:.2f}")
    
    # Análise de tópicos
    topics = analyze_topics(df, dataset_name)
    print("\nAnálise de Tópicos (LDA):")
    for topic in topics:
        print(f"\nTópico {topic['topic_id']}:")
        print(f"Palavras mais importantes: {', '.join(topic['top_words'])}")

if __name__ == "__main__":
    # Carregando e dividindo o dataset
    df_full, df_train, df_test = load_and_split_dataset()
    
    # Executando análises para cada dataset
    datasets = {
        'full': df_full,
        'train': df_train,
        'test': df_test
    }
    
    for dataset_name, df in datasets.items():
        # Executando todas as análises
        all_stats = {
            'text_lengths': analyze_text_lengths(df, dataset_name),
            'input_presence': analyze_input_presence(df, dataset_name),
            'instruction_types': analyze_instruction_types(df, dataset_name),
            'readability': analyze_readability(df, dataset_name),
            'topics': analyze_topics(df, dataset_name)
        }
        
        # Imprimindo estatísticas detalhadas
        print_detailed_stats(df, dataset_name)
        
        # Salvando resultados
        save_analysis_results(all_stats, dataset_name)
    
    print("\nAnálise concluída! Verifique os arquivos em eda/stats/ para estatísticas detalhadas")