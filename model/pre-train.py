# ----------------------------------
# This is a code example for pretraining the model, describing the hyperparameters used for pretraining.
# ----------------------------------


# Some imports
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Initialise the tokenizer of antibody
tokenizer = RobertaTokenizer.from_pretrained(
    "tokenizer"
)

# Initialise the data collator, which is necessary for batching
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
text_datasets = {
    "train": ['assets/pre-train.txt'],
    "eval": ['assets/pre-val.txt'],
    "test": ['assets/pre-test.txt']
}

dataset = load_dataset("text", data_files=text_datasets)
tokenized_dataset = dataset.map(
    lambda z: tokenizer(
        z["text"],
        padding="max_length",
        truncation=True,
        max_length=150,
        return_special_tokens_mask=True,
    ),
    num_proc=1,
    batched=True,
    remove_columns=["text"],
)
NanoBERTa_config = {
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
}
# Initialise the model
model_config = RobertaConfig(
    vocab_size=NanoBERTa_config.get("vocab_size"),
    hidden_size=NanoBERTa_config.get("hidden_size"),
    max_position_embeddings=NanoBERTa_config.get("max_position_embeddings"),
    num_hidden_layers=NanoBERTa_config.get("num_hidden_layers", 12),
    num_attention_heads=NanoBERTa_config.get("num_attention_heads", 12),
    type_vocab_size=1,
)
model = RobertaForMaskedLM(model_config)
# construct training arguments
args = TrainingArguments(
    output_dir="test",
    overwrite_output_dir=True,
    per_device_train_batch_size=NanoBERTa_config.get("batch_size", 32),
    per_device_eval_batch_size=NanoBERTa_config.get("batch_size", 32),
    max_steps=225000,
    save_steps=2500,
    logging_steps=2500,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    learning_rate=1e-4,
    gradient_accumulation_steps=NanoBERTa_config.get("gradient_accumulation_steps", 1),
    fp16=True,
    evaluation_strategy="steps",
    seed=42  # uses a default seed(42)
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"]
)
# Predict MLM performance on the test dataset
result = trainer.predict(tokenized_dataset['test'])
print(result)
