# python mrna_predictor.py 
#   --fastq_path /path/to/your/input.fastq 
#   --model_dir /path/to/your/model/directory 
#   --csv_dir /path/to/your/csv/directory 
#   --threshold 0.9 
#   --batch_size 64


import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import pandas as pd
import csv
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

#########################################
### Define helper functions ###
#########################################

# Extract sequences from a FASTQ file
def extract_sequences_from_fastq(fastq_file):
    with open(fastq_file, 'r') as file:
        while True:
            header = file.readline().strip()
            if not header:
                break
            sequence = file.readline().strip()
            file.readline().strip()
            file.readline().strip()
            yield sequence

# Extract sequences from a FASTA file
def extract_sequences_from_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence:
                    yield sequence
                    sequence = ""
            else:
                sequence += line
        if sequence:
            yield sequence

# Preprocess a sequence to a specified max length
def preprocess_sequence(sequence, max_length):
    return sequence[:max_length]

# Predict labels for a batch of sequences
def predict_sequences(model, tokenizer, sequences, label_mapping, max_length, threshold, device):
    model.eval()
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence_scores, predicted_labels = torch.max(probs, dim=-1)

    predictions = []
    for i in range(len(sequences)):
        if confidence_scores[i] < threshold:
            label = 'Unknown'
        else:
            label = label_mapping[predicted_labels[i].item()]
        predictions.append((sequences[i], label, confidence_scores[i].item()))
    return predictions

# Calculate statistics for the predicted labels
def calculate_label_statistics(predictions, label_type):
    label_counts = defaultdict(int)
    label_confidence_sums = defaultdict(float)

    for prediction in predictions:
        if label_type == 'virus':
            virus_label = prediction[1]
            virus_conf = prediction[2]
            label_counts[virus_label] += 1
            label_confidence_sums[virus_label] += virus_conf
        elif label_type == 'variant':
            variant_label = prediction[3] if prediction[1] == 'sars_cov_2' else prediction[5]
            variant_conf = prediction[4] if prediction[1] == 'sars_cov_2' else prediction[6]
            if variant_label != 'Unknown':
                label_counts[variant_label] += 1
                label_confidence_sums[variant_label] += variant_conf

    df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    df['Confidence_Sum'] = df['Label'].map(label_confidence_sums)
    df['Avg_Confidence'] = df['Confidence_Sum'] / df['Count']

    median_count = df['Count'].median()
    mad_count = (df['Count'] - median_count).abs().median()
    min_scale = 1

    df['Z_Score'] = (df['Count'] - median_count) / np.maximum(mad_count, min_scale)
    label_statistics = df.set_index('Label').to_dict(orient='index')
    
    return label_statistics

# Initialize the output CSV file with headers
def initialize_output_csv(output_csv_file):
    with open(output_csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Virus_Label', 'Virus_Confidence', 'Variant_COV_Label', 'Variant_COV_Confidence', 'Variant_IAV_Label', 'Variant_IAV_Confidence'])

# Append predictions to the output CSV file
def append_prediction_to_csv(output_csv_file, predictions):
    with open(output_csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)

#########################################
### Main function ###
#########################################

def mrna_predictor(fastq_path, model_dir, csv_dir, threshold, batch_size):
    # Determine base filename and output CSV path
    base_filename = os.path.basename(fastq_path).rsplit('.', 1)[0]
    output_csv_file = os.path.join(model_dir, f"{base_filename}_prediction.csv")
    
    # Define paths to models and label CSV files
    virus_model_path = os.path.join(model_dir, "1_DNABert2_250bp_200overlap_virus_918epi/model")
    virus_labels_csv_path = os.path.join(csv_dir, "WGS_by_virus_finetune_250bp_200overlap_918epi.csv")
    
    cov2_model_path = os.path.join(model_dir, "3_DNABert2_RBD_500k_VOC_labeled_250bp_1bp_overlap_5k_epi/model")
    cov2_labels_csv_path = os.path.join(csv_dir, "RBD_valid_nucleotides_500k_VOC_labeled_5000_epi_250bp_1bp_overlap.csv")

    iav_model_path = os.path.join(model_dir, "2_DNABert2_IAVstrains_finetune_250bp_50overlap_5k_epi/model")
    iav_labels_csv_path = os.path.join(csv_dir, "WGS_by_VOC_IAV_finetune_5k_epi_250bp_fragments.csv")

    # Determine file type and extract sequences accordingly
    if fastq_path.endswith(".fastq") or fastq_path.endswith(".fq"):
        sequence_generator = extract_sequences_from_fastq(fastq_path)
    elif fastq_path.endswith(".fasta") or fastq_path.endswith(".fa"):
        sequence_generator = extract_sequences_from_fasta(fastq_path)
    else:
        raise ValueError("Input file must be in FASTQ or FASTA format.")

    # Load virus model and tokenizer
    virus_model = BertForSequenceClassification.from_pretrained(virus_model_path)
    virus_tokenizer = AutoTokenizer.from_pretrained(virus_model_path)

    # Load virus label CSV and create label mapping
    virus_df = pd.read_csv(virus_labels_csv_path)
    virus_label_mapping = {i: label for i, label in enumerate(virus_df['label_name'].unique())}

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    virus_model.to(device)

    # Initialize the output CSV file
    initialize_output_csv(output_csv_file)

    # Load CoV-2 and IAV models and tokenizers
    cov2_model = BertForSequenceClassification.from_pretrained(cov2_model_path)
    cov2_tokenizer = AutoTokenizer.from_pretrained(cov2_model_path)

    iav_model = BertForSequenceClassification.from_pretrained(iav_model_path)
    iav_tokenizer = AutoTokenizer.from_pretrained(iav_model_path)

    # Load CoV-2 and IAV label CSVs and create label mappings
    cov2_df = pd.read_csv(cov2_labels_csv_path)
    cov2_label_mapping = {i: label for i, label in enumerate(cov2_df['label_name'].unique())}

    iav_df = pd.read_csv(iav_labels_csv_path)
    iav_label_mapping = {i: label for i, label in enumerate(iav_df['label_name'].unique())}

    cov2_model.to(device)
    iav_model.to(device)

    # Count total sequences for progress bar
    total_sequences = sum(1 for _ in sequence_generator)
    sequence_generator = extract_sequences_from_fastq(fastq_path) if fastq_path.endswith(".fastq") or fastq_path.endswith(".fq") else extract_sequences_from_fasta(fastq_path)

    # Predict and process sequences in batches
    with tqdm(total=total_sequences, desc="Predicting variants") as pbar:
        final_predictions = []
        batch_sequences = []
        for sequence in sequence_generator:
            preprocessed_sequence = preprocess_sequence(sequence, 250)
            batch_sequences.append(preprocessed_sequence)

            if len(batch_sequences) == batch_size:
                virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, threshold, device)
                batch_output = []

                for seq, virus_label, virus_conf in virus_predictions:
                    variant_cov_label = 'Unknown'
                    variant_cov_conf = 0.0
                    variant_iav_label = 'Unknown'
                    variant_iav_conf = 0.0

                    if virus_label == 'sars_cov_2':
                        variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, threshold, device)
                        _, variant_cov_label, variant_cov_conf = variant_predictions[0]
                    elif virus_label == 'influenza_a':
                        variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, threshold, device)
                        _, variant_iav_label, variant_iav_conf = variant_predictions[0]
                    
                    batch_output.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                    final_predictions.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                
                append_prediction_to_csv(output_csv_file, batch_output)
                batch_sequences = []
                pbar.update(batch_size)

        # Process any remaining sequences
        if batch_sequences:
            virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, threshold, device)
            batch_output = []

            for seq, virus_label, virus_conf in virus_predictions:
                variant_cov_label = 'Unknown'
                variant_cov_conf = 0.0
                variant_iav_label = 'Unknown'
                variant_iav_conf = 0.0

                if virus_label == 'sars_cov_2':
                    variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, threshold, device)
                    _, variant_cov_label, variant_cov_conf = variant_predictions[0]
                elif virus_label == 'influenza_a':
                    variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, threshold, device)
                    _, variant_iav_label, variant_iav_conf = variant_predictions[0]
                
                batch_output.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                final_predictions.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
            
            append_prediction_to_csv(output_csv_file, batch_output)
            pbar.update(len(batch_sequences))

    # Calculate and print statistics for virus labels
    virus_label_statistics = calculate_label_statistics(final_predictions, 'virus')
    sorted_virus_label_statistics = sorted(virus_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    top_virus_labels = [label for label, stats in sorted_virus_label_statistics[:2]]
    
    for label, stats in sorted_virus_label_statistics[:2]:
        print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Plot distribution of viral labels
    label_counts = defaultdict(int)
    for _, virus_label, _, _, _, _, _ in final_predictions:
        label_counts[virus_label] += 1

    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1%%', startangle=140)
    plt.title('Distribution of Viral Labels')
    plt.axis('equal')
    plt.show()

    # Calculate and print statistics for CoV-2 variants
    cov2_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'sars_cov_2']
    cov2_label_statistics = calculate_label_statistics(cov2_predictions, 'variant')
    sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    print("\nTop Variant Labels for sars_cov_2:")
    for label, stats in sorted_cov2_label_statistics[:1]:
        print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Calculate and print statistics for IAV variants
    iav_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'influenza_a']
    iav_label_statistics = calculate_label_statistics(iav_predictions, 'variant')
    sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    print("\nTop Variant Labels for influenza_a:")
    for label, stats in sorted_iav_label_statistics[:1]:
        print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict labels for RNA sequences from a FASTQ or FASTA file using pre-trained DNABERT models and calculate label statistics.")
    parser.add_argument("--fastq_path", type=str, required=True, help="Path to input FASTQ or FASTA file (e.g. path/to/fasta/sample.fasta)")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing models (e.g. path/to/DNABert2/model/directory)")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to the directory containing CSV files used to train models")
    parser.add_argument("--threshold", type=float, required=True, help="Confidence threshold for predictions.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing sequences.")
    
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    mrna_predictor(args.fastq_path, args.model_dir, args.csv_dir, args.threshold, args.batch_size)
