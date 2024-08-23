#!/usr/bin/env python
# coding: utf-8

### SARSCOV2 VOC classification model with logs per epoch ###
#########################################
### Importing the necessary libraries ###
#########################################

import pandas as pd
import numpy as np
import torch
from pathlib import Path 
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, Dataset as HFDataset

#########################################
### Define custom dataset class ###
#########################################

class HF_dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

#########################################
### Define metrics computation function ###
#########################################

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1': f1}

#########################################
### Define custom callback for logging metrics ###
#########################################

class LogCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log_file = os.path.join(self.log_dir, f"metrics_epoch_{int(state.epoch)}.log")
            with open(log_file, 'a') as f:
                f.write(f"Epoch {int(state.epoch)}:\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

############################################
### Reading and splitting the data ###
############################################

# Set the main directory path
main_dir = Path("/mmfs1/projects/changhui.yan/DeewanB/DNABert2_rnaseq")

data_path = main_dir / "genome_files/unfiltered_multiple_genomes/RBD_nucleotides_3mil_wo_nonvoc_100k_epi_250bp_50overlap_complementary.csv"

df = pd.read_csv(data_path, dtype={"label_name": str, "sequence": str})

# Ensure labels are encoded as integers
df['label_name'] = pd.factorize(df['label_name'])[0]

# Split data into train and test datasets (80% train, 10% val, 10% test)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label_name'], random_state=42)

# Further split the test dataset into eval and test datasets (50% eval, 50% test of the test split)
df_val, df_test = train_test_split(df_test, test_size=0.5, stratify=df_test['label_name'], random_state=42)

print(f"Training data size: {len(df_train)}")
print(f"Validation data size: {len(df_val)}")
print(f"Test data size: {len(df_test)}")

# Prepare training data
train_sequences, labels_train = df_train["sequence"].tolist(), df_train["label_name"].tolist()

NUM_CLASSES = len(np.unique(labels_train))

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
model = BertForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", num_labels=NUM_CLASSES)

SEQ_MAX_LEN = 150  # Max length of BERT

train_encodings = tokenizer.batch_encode_plus(
    train_sequences,
    max_length=SEQ_MAX_LEN,
    padding=True,  # Pad to max len
    truncation=True,  # Truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # Return PyTorch tensors
)
train_dataset = HF_dataset(
    train_encodings, labels_train
)

# Prepare validation data
val_sequences, labels_val = df_val["sequence"].tolist(), df_val["label_name"].tolist()

val_encodings = tokenizer.batch_encode_plus(
    val_sequences,
    max_length=SEQ_MAX_LEN,
    padding=True,  # Pad to max len
    truncation=True,  # Truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # Return PyTorch tensors
)
val_dataset = HF_dataset(
    val_encodings, labels_val
)

# Prepare test data
test_sequences, labels_test = df_test["sequence"].tolist(), df_test["label_name"].tolist()

test_encodings = tokenizer.batch_encode_plus(
    test_sequences,
    max_length=SEQ_MAX_LEN,
    padding=True,  # Pad to max len
    truncation=True,  # Truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # Return PyTorch tensors
)
test_dataset = HF_dataset(
    test_encodings, labels_test
)

############################################
### Training and evaluating the model ###
############################################

results_dir = main_dir / "3_DNABer2_RBD_3mil_wo_nonvoc_100k_epi_250bp_50overlap_complementary"

results_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 10
BATCH_SIZE = 64  # Reduced batch size for memory management
LEARNING_RATE = 3.0e-05

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=results_dir / "checkpoints",  # Output directory
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=50,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # L2 regularization lambda value
    logging_steps=60,  # Log metrics every 60 steps
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # Use mixed precision training
    gradient_accumulation_steps=4,  # Simulate larger batch size
    learning_rate=LEARNING_RATE,  # Set learning rate
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,  # Compute metrics function is used for evaluation at the end of each epoch
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[LogCallback(results_dir / "checkpoints")],  # Use LogCallback here
)

# Find the latest checkpoint if exists
last_checkpoint = None
if (results_dir / "checkpoints").exists():
    checkpoints = list(sorted((results_dir / "checkpoints").glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[-1])))
    if checkpoints:
        last_checkpoint = str(checkpoints[-1])

# Train the model
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save the model and tokenizer
model_path = results_dir / "model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Evaluate on the test dataset
eval_results = trainer.evaluate(test_dataset)

# Extract metrics from evaluation results
avg_acc = eval_results["eval_accuracy"]
avg_f1 = eval_results["eval_f1"]

print(f"Test accuracy: {avg_acc}")
print(f"Test F1: {avg_f1}")

# Generate plots for training and testing metrics
training_metrics = trainer.state.log_history

epochs = []
train_losses = []
eval_losses = []
eval_accuracies = []
eval_f1s = []

for entry in training_metrics:
    if 'epoch' in entry:
        if 'loss' in entry or 'eval_loss' in entry or 'eval_accuracy' in entry or 'eval_f1' in entry:
            epochs.append(entry['epoch'])
            train_losses.append(entry.get('loss', np.nan))
            eval_losses.append(entry.get('eval_loss', np.nan))
            eval_accuracies.append(entry.get('eval_accuracy', np.nan))
            eval_f1s.append(entry.get('eval_f1', np.nan))

plt.figure(figsize=(14, 10))

# Plot train and eval losses
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, eval_losses, label='Eval Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Eval Loss vs Epochs')
plt.legend()

# Plot eval accuracy and f1 score
plt.subplot(2, 1, 2)
plt.plot(epochs, eval_accuracies, label='Eval Accuracy', marker='o')
plt.plot(epochs, eval_f1s, label='Eval F1 Score', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Eval Accuracy and F1 Score vs Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(results_dir / 'training_metrics.png')
plt.show()
