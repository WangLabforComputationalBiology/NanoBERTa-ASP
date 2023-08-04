from transformers import (
    RobertaTokenizer,
    RobertaForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import (
    Dataset,
    DatasetDict,
    Sequence,
    ClassLabel
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TOKENIZER_DIR = "antibody-tokenizer"
history = []
# Initialise a tokenizer
tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_DIR, max_len=150)
train_df = pd.read_parquet(
    'assets/nanotrain.parquet'
)
val_df = pd.read_parquet(
    'assets/nanoval.parquet'
)
test_df = pd.read_parquet(
    'assets/nanotest.parquet'
)
ab_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[['sequence','paratope_labels']]),
    "validation": Dataset.from_pandas(val_df[['sequence','paratope_labels']]),
    "test": Dataset.from_pandas(test_df[['sequence','paratope_labels']])
})
paratope_class_label = ClassLabel(2, names=['N','P'])
ab_dataset_featurised = ab_dataset.map(
    lambda seq, labels: {
        "sequence": seq,
        "paratope_labels": [paratope_class_label.str2int(sample) for sample in labels]
    },
    input_columns=["sequence", "paratope_labels"], batched=True
)


def preprocess(batch):
    t_inputs = tokenizer(batch['sequence'],
                         padding="max_length")
    batch['input_ids'] = t_inputs.input_ids
    batch['attention_mask'] = t_inputs.attention_mask

    # enumerate
    labels_container = []
    for index, labels in enumerate(batch['paratope_labels']):
        tokenized_input_length = len(batch['input_ids'][index])
        paratope_label_length = len(batch['paratope_labels'][index])

        n_pads_with_eos = max(1, tokenized_input_length - paratope_label_length - 1)

        labels_padded = [-100] + labels + [-100] * n_pads_with_eos

        assert len(labels_padded) == len(batch['input_ids'][index]), \
            f"Lengths don't align, {len(labels_padded)}, {len(batch['input_ids'][index])}, {len(labels)}"

        labels_container.append(labels_padded)

    batch['labels'] = labels_container

    for i, v in enumerate(batch['labels']):
        assert len(batch['input_ids'][i]) == len(v) == len(batch['attention_mask'][i])

    return batch


ab_dataset_tokenized = ab_dataset_featurised.map(
    preprocess,
    batched=True,
    batch_size=8,
    remove_columns=['sequence', 'paratope_labels']
)
label_list = paratope_class_label.names


def compute_metrics(p):

    predictions, labels = p

    prediction_pr = torch.softmax(torch.from_numpy(predictions), dim=2).detach().numpy()[:, :, -1]

    # We run an argmax to get the label
    predictions = np.argmax(predictions, axis=2)

    preds = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    labs = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    probs = [
        [prediction_pr[i][pos] for (pr, (pos, l)) in zip(prediction, enumerate(label)) if l != -100]
        for i, (prediction, label) in enumerate(zip(predictions, labels))
    ]

    # flatten
    preds = sum(preds, [])
    labs = sum(labs, [])
    probs = sum(probs, [])

    mec = {
        "precision": precision_score(labs, preds, pos_label="P"),
        "recall": recall_score(labs, preds, pos_label="P"),
        "f1": f1_score(labs, preds, pos_label="P"),
        "auc": roc_auc_score(labs, probs),
        "aupr": average_precision_score(labs, probs, pos_label="P"),
        "mcc": matthews_corrcoef(labs, preds),
    }
    history.append(mec)

    return mec


batch_size = 32
RUN_ID = "ASP"
SEED = 0
LR = 1e-6

args = TrainingArguments(
    f"{RUN_ID}_{SEED}", # this is the name of the checkpoint folder
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LR, # 1e-6, 5e-6, 1e-5. .... 1e-3
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=100,
    warmup_ratio=0, # 0, 0.05, 0.1 ....
    load_best_model_at_end=True,
    lr_scheduler_type='linear',
    metric_for_best_model='aupr', # name of the metric here should correspond to metrics defined in compute_metrics
    logging_strategy='epoch',
    seed=SEED
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)

# Name of the pre-trained model after you train your MLM
MODEL_DIR = "NanoBERTa-pre"

# We initialise a model using the weights from the pre-trained model
model = RobertaForTokenClassification.from_pretrained(MODEL_DIR, num_labels=2)

trainer = Trainer(
    model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=ab_dataset_tokenized['train'],
    eval_dataset=ab_dataset_tokenized['validation'],
    compute_metrics=compute_metrics
)
trainer.train()
pred = trainer.predict(
    ab_dataset_tokenized['test']
)
a=pred.metrics
print(a)

metrics_list = history

metric_names = ['precision', 'recall', 'f1', 'auc', 'aupr', 'mcc']

# Create a dictionary to store metric values
metric_dict = {}
for metric_name in metric_names:
    metric_dict[metric_name] = []

# Traverse each dictionary and extract the value of each metric
for i in range(len(metrics_list)):
    for metric_name in metric_names:
        metric_value = metrics_list[i].get(metric_name)
        if metric_value is not None:
            metric_dict[metric_name].append(metric_value)

# Draw a Line chart for each indicator
for metric_name, metric_value in metric_dict.items():
    plt.plot(metric_value, label=metric_name)

# Set image titles and axis labels
plt.title('Model Metrics')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')

# add legend
plt.legend()

# Save image as PNG file
plt.savefig('metrics.png')
