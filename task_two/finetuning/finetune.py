# Importações necessárias para o fine-tuning
import os
import pandas as pd
import re
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model
)
from datasets import Dataset
from typing import Dict
import logging
from torch.nn import CrossEntropyLoss
import math



# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCallback(TrainerCallback):
    """Callback para cálculo de métricas com tratamento de dispositivos"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.losses = []
        self.perplexities = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            loss = logs.get("loss", None)
            if loss is not None:
                self.losses.append(loss)
                perplexity = math.exp(loss)
                self.perplexities.append(perplexity)
                logger.info(f"Loss: {loss:.4f} | Perplexity: {perplexity:.4f}")

def compute_metrics(eval_preds):
    """Calcula métricas com conversão explícita para CPU"""
    predictions, labels = eval_preds
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions

    # Converter para tensores CPU
    import torch
    predictions = torch.tensor(predictions).cpu()
    labels = torch.tensor(labels).cpu()

    loss_fct = CrossEntropyLoss()
    predictions = predictions.view(-1, predictions.size(-1))
    labels = labels.view(-1)
    
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    loss = loss_fct(predictions, labels)
    perplexity = math.exp(loss.item())
    
    return {
        "perplexity": perplexity,
        "loss": loss.item()
    }

class AlpacaDataset:
    """Classe de dataset com tratamento robusto de prompts"""
    
    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_prompt(self, example: Dict) -> str:
        has_input = 'input' in example and example['input']
        input_text = example['input']
        
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""
        
        has_input = has_input and input_text.strip()
        
        template = (
            f"Instruction: {example['instruction']}\n"
            f"Input: {input_text}\n" if has_input else ""
            f"Response: {example['output']}"
            f"{self.tokenizer.eos_token}"  # Adição explícita do EOS token
        )
        return template
    
    def tokenize(self, examples: Dict) -> Dict:
        prompts = [
            self.format_prompt({
                'instruction': examples['instruction'][i],
                'input': examples.get('input', [''])[i],
                'output': examples['output'][i]
            }) for i in range(len(examples['instruction']))
        ]
        
        return self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

def clean_text(text: str) -> str:
    """Limpeza preservando case original"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_dataset(dataset: Dataset) -> Dataset:
    def clean_example(example):
        return {
            'instruction': clean_text(example['instruction']),
            'input': clean_text(example.get('input', '')),
            'output': clean_text(example['output'])
        }
    return dataset.map(clean_example)

def load_dataset(path: str) -> Dataset:
    """Carregamento com tratamento de erros"""
    try:
        logger.info(f"Carregando dataset de {path}")
        df = pd.read_parquet(path)
        dataset = Dataset.from_pandas(df)
        return preprocess_dataset(dataset)
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        raise

def create_lora_config() -> LoraConfig:
    """Configuração LoRA otimizada para GPT-2"""
    return LoraConfig(
        r=8,
        lora_alpha=16, # 2*r
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,  # Dropout reduzido
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        fan_in_fan_out=True
    )

def main():
    MODEL_NAME = "gpt2"
    DATASET_PATH = "../../samples/train_dataset.parquet"
    OUTPUT_DIR = "checkpoints"
    LOGGING_DIR = "logs"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # Carregar tokenizer com configurações atualizadas
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Melhor para geração causal

    # Carregar modelo
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Definir requires_grad para todos os parâmetros base como False
    for param in model.parameters():
        param.requires_grad = False

    # Aplicar LoRA 
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # Verificar e imprimir parâmetros treináveis
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Parâmetros treináveis: {trainable_params}")
    print(f"Total de parâmetros: {all_params}")
    print(f"Percentagem de parâmetros treináveis: {100 * trainable_params / all_params:.2f}%")

    # Dataset e pré-processamento
    dataset = load_dataset(DATASET_PATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Processamento de tokens
    processor = AlpacaDataset(tokenizer)
    tokenized_train = dataset["train"].map(
        processor.tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_eval = dataset["test"].map(
        processor.tokenize,
        batched=True,
        remove_columns=dataset["test"].column_names
    )

    # Configuração de treino simplificada e compatível
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,  # Essencial para PEFT
        optim="adamw_torch",
        fp16=False,  # Desativado para evitar problemas
        use_cpu=True,  # Forçar uso de CPU - nova abordagem que substitui no_cuda
        # Remova use_mps_device completamente
        gradient_checkpointing=False,
        dataloader_num_workers=1,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback(tokenizer)]
        #label_names=["labels"]
    )

    trainer.train()

    # Salvamento final
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    main()
