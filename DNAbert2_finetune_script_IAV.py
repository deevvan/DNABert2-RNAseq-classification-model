#!/usr/bin/env python
# coding: utf-8

### IAV model without kmers ###
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

#########################################
### Define custom dataset class ###
#########################################

class HF_dataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

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
### Define custom callback for early stopping based on accuracy ###
#########################################

class AccuracyThresholdCallback(TrainerCallback):
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get('eval_accuracy', 0) >= self.threshold:
            print(f"Stopping early as accuracy reached {metrics['eval_accuracy']:.4f} which is above the threshold of {self.threshold:.4f}.")
            control.should_training_stop = True
        return control

############################################
### Reading and splitting the data ###
############################################

# Set the main directory path
main_dir = Path("/mmfs1/projects/changhui.yan/DeewanB/DNABert2_rnaseq")

data_path = main_dir / "genome_files/unfiltered_multiple_genomes/WGS_by_VOC_IAV_finetune_5k_epi_250bp_fragments.csv"

df = pd.read_csv(data_path)

# Ensure labels are encoded as integers
df['label_name'] = pd.factorize(df['label_name'])[0]

# Split data into train and test datasets (70% train, 15% val, 15% test)
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['label_name'], random_state=42)

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

SEQ_MAX_LEN = 250  # Max length of BERT

train_encodings = tokenizer.batch_encode_plus(
    train_sequences,
    max_length=SEQ_MAX_LEN,
    padding=True,  # Pad to max len
    truncation=True,  # Truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # Return PyTorch tensors
)
train_dataset = HF_dataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], labels_train
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
    val_encodings["input_ids"], val_encodings["attention_mask"], labels_val
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
    test_encodings["input_ids"], test_encodings["attention_mask"], labels_test
)

############################################
### Training and evaluating the model ###
############################################

results_dir = main_dir / "DNABert2_IAVstrains_finetune_250bp_50overlap_5k_epi"
results_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 10
BATCH_SIZE = 64  # Reduced batch size for memory management
LEARNING_RATE = 9.72339793344816e-6  # 7.178770475697169e-06 # Set learning rate

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=results_dir / "checkpoints",  # Output directory
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
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
    callbacks=[AccuracyThresholdCallback(threshold=1.0)],  # Add the custom callback here
)

trainer.train()

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
