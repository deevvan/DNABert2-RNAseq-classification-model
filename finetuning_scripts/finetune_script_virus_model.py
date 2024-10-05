#!/usr/bin/env python
# coding: utf-8

### Virus model with log epochs & recall precision ###
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from sklearn.preprocessing import label_binarize

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
        # Correctly construct tensors by cloning and detaching
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item
    
#########################################
### Define metrics computation function ###
#########################################

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Calculate precision and recall
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    # Calculate confusion matrix for all classes
    conf_matrix = confusion_matrix(labels, predictions)

    # Convert confusion matrix to list to make it JSON serializable
    conf_matrix_list = conf_matrix.tolist()

    # Return metrics including precision, recall, and confusion matrix as lists
    return {
        'accuracy': accuracy, 
        'f1': f1, 
        'precision': precision,
        'recall': recall,
        'conf_matrix': conf_matrix_list  # Return the confusion matrix as a list
    }

#########################################
### Define custom callback for early stopping based on accuracy ###
#########################################

class AccuracyThresholdCallback(TrainerCallback):
    def __init__(self, threshold=0.9995):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get('eval_accuracy', 0) >= self.threshold:
            print(f"Stopping early as accuracy reached {metrics['eval_accuracy']:.4f} which is above the threshold of {self.threshold:.4f}.")
            control.should_training_stop = True
        return control

#########################################
### Define custom callback for logging metrics ###
#########################################

class LogCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            epoch = int(state.epoch)
            step = state.global_step

            # Collect training metrics for each step
            training_log_file = os.path.join(self.log_dir, f"training_metrics_epoch_{epoch}_step_{step}.log")
            with open(training_log_file, 'w') as f:
                for key, value in logs.items():
                    if key not in ['logits', 'labels']:  # Avoid logging logits and labels
                        f.write(f"{key}: {value}\n")

            # Add to metrics history without logits and labels
            logs_to_save = {k: v for k, v in logs.items() if k not in ['logits', 'labels']}
            logs_to_save['epoch'] = epoch
            logs_to_save['step'] = step
            self.metrics_history.append(logs_to_save)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = int(state.epoch)

            # Write evaluation metrics for each epoch
            eval_log_file = os.path.join(self.log_dir, f"evaluation_metrics_epoch_{epoch}.log")
            with open(eval_log_file, 'w') as f:
                for key, value in metrics.items():
                    if key not in ['logits', 'labels']:  # Avoid logging logits and labels
                        f.write(f"{key}: {value}\n")

    def on_train_end(self, args, state, control, **kwargs):
        # Write all metrics to a final JSON file
        final_metrics_file = os.path.join(self.log_dir, "all_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

############################################
### Reading and splitting the data ###
############################################

# Set the main directory path
main_dir = Path("/path/to/directory/to/save/model/directory/")

# Path where data used to finetune virus model is saved
data_path = main_dir / "finetuning_data_virus_model.csv"

df = pd.read_csv(data_path, dtype={"label_name": str, "sequence": str})

# Factorize the label names to create numeric labels
df['label_number'], label_names = pd.factorize(df['label_name'])

# Split data into train and test datasets (80% train, 10% val, 10% test)
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label_number'], random_state=42)

# Further split the test dataset into eval and test datasets (50% eval, 50% test of the test split)
df_val, df_test = train_test_split(df_test, test_size=0.5, stratify=df_test['label_number'], random_state=42)

print(f"Training data size: {len(df_train)}")
print(f"Validation data size: {len(df_val)}")
print(f"Test data size: {len(df_test)}")

# Prepare training data
train_sequences, labels_train = df_train["sequence"].tolist(), df_train["label_number"].tolist()

NUM_CLASSES = len(np.unique(labels_train))

# Load the DNABERT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
model = BertForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", num_labels=NUM_CLASSES)

# Max length is modified to represent the max length of ART simulated segments used to train the model
SEQ_MAX_LEN = 250  

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
val_sequences, labels_val = df_val["sequence"].tolist(), df_val["label_number"].tolist()

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
test_sequences, labels_test = df_test["sequence"].tolist(), df_test["label_number"].tolist()

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


# Write the DataFrame back to the same CSV file (overwriting the original file with factorized label_numbers)
df.to_csv(data_path, index=False)

############################################
### Training and evaluating the model ###
############################################

# Saving model directory in main_dir path
results_dir = main_dir / "DNABERT2_virus_model"

results_dir.mkdir(parents=True, exist_ok=True)

# Defining training parameters
EPOCHS = 8 # Define number of epochs 
BATCH_SIZE = 32  # Adjust batch size for memory optimization
LEARNING_RATE = 5.0e-05 # Define learning rate

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
    callbacks=[LogCallback(results_dir / "checkpoints"), AccuracyThresholdCallback(threshold=0.9995)]  # Combined callbacks
)

# Find the latest checkpoint if exists to resume training from last checkpoint
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
precision = eval_results["eval_precision"]
recall = eval_results["eval_recall"]
conf_matrix = np.array(eval_results["eval_conf_matrix"])  # Convert back to numpy array

print(f"Test accuracy: {avg_acc}")
print(f"Test F1: {avg_f1}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")

# Evaluate on the test dataset to get logits and labels
predictions = trainer.predict(test_dataset)
logits = predictions.predictions  # Extract logits
true_labels = predictions.label_ids  # Extract true labels

# Save the logits, true labels, and evaluation metrics to a JSON file
results_json = {
    'logits': logits.tolist(),  # Convert to list for JSON serialization
    'true_labels': true_labels.tolist(),  # Convert to list for JSON serialization
    'metrics': {
        'accuracy': avg_acc,
        'f1': avg_f1,
        'precision': precision,
        'recall': recall,
        'conf_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
    }
}

with open(results_dir / 'test_results.json', 'w') as json_file:
    json.dump(results_json, json_file, indent=4)
print(f"Test results saved to {results_dir / 'test_results.json'}")

# Convert logits and labels for ROC AUC calculation
logits = np.array(logits)
true_labels = np.array(true_labels)
true_labels_binarized = label_binarize(true_labels, classes=np.arange(NUM_CLASSES))

fpr = {}
tpr = {}
roc_auc = {}

plt.figure()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], logits[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'{label_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Virus Model')
plt.legend(loc="lower right")
plt.savefig(results_dir / 'virus_model_roc_auc.png')
plt.show()

# Display confusion matrix with label names
plt.figure(figsize=(14, 10))  # Increase figure size to accommodate rotated labels
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Virus Model')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=45, ha='right')  # Rotate y-axis labels
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(results_dir / 'virus_model_confusion_matrix.png')
plt.show()

# Generate plots for training and testing metrics
training_metrics = trainer.state.log_history

epochs = []
train_losses = []
eval_losses = []
eval_accuracies = []
eval_f1s = []
eval_precisions = []
eval_recalls = []

for entry in training_metrics:
    if 'epoch' in entry:
        if 'loss' in entry or 'eval_loss' in entry or 'eval_accuracy' in entry or 'eval_f1' in entry:
            epochs.append(entry['epoch'])
            train_losses.append(entry.get('loss', np.nan))
            eval_losses.append(entry.get('eval_loss', np.nan))
            eval_accuracies.append(entry.get('eval_accuracy', np.nan))
            eval_f1s.append(entry.get('eval_f1', np.nan))
            eval_precisions.append(entry.get('eval_precision', np.nan))
            eval_recalls.append(entry.get('eval_recall', np.nan))

plt.figure(figsize=(14, 10))

# Plot train and eval losses
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, eval_losses, label='Eval Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Eval Loss vs Epochs')
plt.legend()

# Plot eval accuracy, f1 score, precision, and recall
plt.subplot(2, 1, 2)
plt.plot(epochs, eval_accuracies, label='Eval Accuracy', marker='o')
plt.plot(epochs, eval_f1s, label='Eval F1 Score', marker='o')
plt.plot(epochs, eval_precisions, label='Eval Precision', marker='o')
plt.plot(epochs, eval_recalls, label='Eval Recall', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Eval Metrics vs Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(results_dir / 'virus_model_training_metrics.png')
plt.show()
