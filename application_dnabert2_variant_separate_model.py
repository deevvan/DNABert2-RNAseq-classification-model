# python mrna_predictor.py --fastq_path /path/to/your/input.fastq --model_dir /path/to/your/model/directory 
#    --csv_dir /path/to/your/csv/directory --threshold 0.9 --batch_size 64

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

def extract_sequences_from_fastq(fastq_file):
    # Read sequences from a FASTQ file
    sequences = []
    with open(fastq_file, 'r') as file:
        while True:
            header = file.readline().strip()  # Read header line
            if not header:
                break  # End of file
            sequence = file.readline().strip()  # Read sequence line
            file.readline().strip()  # Skip plus line
            file.readline().strip()  # Skip quality score line
            sequences.append(sequence)
    return sequences

def preprocess_sequences(sequences, max_length):
    # Truncate sequences to the specified max length
    return [seq[:max_length] for seq in sequences]

def predict_sequences(model, tokenizer, sequences, label_mapping, max_length, threshold, batch_size, device):
    # Predict labels for sequences using the given model and tokenizer
    model.eval()
    predictions = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence_scores, predicted_labels = torch.max(probs, dim=-1)

            for j in range(len(batch_sequences)):
                if confidence_scores[j] < threshold:
                    label = 'Unknown'
                else:
                    label = label_mapping[predicted_labels[j].item()]
                predictions.append((batch_sequences[j], label, confidence_scores[j].item()))
                
    return predictions

def calculate_label_statistics(predictions, label_type):
    # Calculate statistics for the predicted labels
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

def initialize_output_csv(output_csv_file):
    # Initialize the output CSV file with header
    with open(output_csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Virus_Label', 'Virus_Confidence', 'Variant_COV_Label', 'Variant_COV_Confidence', 'Variant_IAV_Label', 'Variant_IAV_Confidence'])

def append_prediction_to_csv(output_csv_file, sequence, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf):
    # Append a prediction to the output CSV file
    with open(output_csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([sequence, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])

#########################################
### Main function ###
#########################################

def mrna_predictor(fastq_path, model_dir, csv_dir, threshold, batch_size):
    base_filename = os.path.basename(fastq_path).rsplit('.', 1)[0]
    
    temp_csv_file = os.path.join(model_dir, f"{base_filename}_extracted_sequences.csv")
    output_csv_file = os.path.join(model_dir, f"{base_filename}_prediction.csv")
    
    virus_model_path = os.path.join(model_dir, "1_DNABert2_250bp_200overlap_virus_918epi/model")
    virus_labels_csv_path = os.path.join(csv_dir, "WGS_by_virus_finetune_250bp_200overlap_918epi.csv")
    
    cov2_model_path = os.path.join(model_dir, f"3_DNABert2_RBD_500k_VOC_labeled_250bp_1bp_overlap_5k_epi/model") 
    cov2_labels_csv_path = os.path.join(csv_dir, f"RBD_valid_nucleotides_500k_VOC_labeled_5000_epi_250bp_1bp_overlap.csv")

    iav_model_path = os.path.join(model_dir, "2_DNABert2_IAVstrains_finetune_250bp_50overlap_5k_epi/model") 
    iav_labels_csv_path = os.path.join(csv_dir, "WGS_by_VOC_IAV_finetune_5k_epi_250bp_fragments.csv")

    # Extract sequences from the FASTQ file
    sequences = extract_sequences_from_fastq(fastq_path)
    df_sequences = pd.DataFrame(sequences, columns=["Sequence"])
    df_sequences.to_csv(temp_csv_file, index=False)

    # Preprocess sequences to the specified max length
    preprocessed_sequences = preprocess_sequences(sequences, 250)

    # Load the virus model and tokenizer
    virus_model = BertForSequenceClassification.from_pretrained(virus_model_path)
    virus_tokenizer = AutoTokenizer.from_pretrained(virus_model_path)

    # Load the virus labels CSV and create a label mapping
    virus_df = pd.read_csv(virus_labels_csv_path)
    virus_label_mapping = {i: label for i, label in enumerate(virus_df['label_name'].unique())}

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    virus_model.to(device)

    # Initialize the output CSV file
    initialize_output_csv(output_csv_file)

    # Predict virus labels for the sequences
    virus_predictions = predict_sequences(virus_model, virus_tokenizer, preprocessed_sequences, virus_label_mapping, 250, threshold, batch_size, device)

    # Load the CoV-2 model and tokenizer
    cov2_model = BertForSequenceClassification.from_pretrained(cov2_model_path)
    cov2_tokenizer = AutoTokenizer.from_pretrained(cov2_model_path)

    # Load the IAV model and tokenizer
    iav_model = BertForSequenceClassification.from_pretrained(iav_model_path)
    iav_tokenizer = AutoTokenizer.from_pretrained(iav_model_path)

    # Load the CoV-2 and IAV labels CSV and create label mappings
    cov2_df = pd.read_csv(cov2_labels_csv_path)
    cov2_label_mapping = {i: label for i, label in enumerate(cov2_df['label_name'].unique())}

    iav_df = pd.read_csv(iav_labels_csv_path)
    iav_label_mapping = {i: label for i, label in enumerate(iav_df['label_name'].unique())}

    # Move CoV-2 and IAV models to the device
    cov2_model.to(device)
    iav_model.to(device)

    # Predict variants for the sequences
    total_sequences = len(preprocessed_sequences)
    with tqdm(total=total_sequences, desc="Predicting variants") as pbar:
        final_predictions = []
        for sequence, virus_label, virus_conf in virus_predictions:
            variant_cov_label = 'Unknown'
            variant_cov_conf = 0.0
            variant_iav_label = 'Unknown'
            variant_iav_conf = 0.0
            
            if virus_label == 'sars_cov_2':
                variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [sequence], cov2_label_mapping, 250, threshold, batch_size, device)
                if variant_predictions:
                    _, variant_cov_label, variant_cov_conf = variant_predictions[0]
            elif virus_label == 'influenza_a':
                variant_predictions = predict_sequences(iav_model, iav_tokenizer, [sequence], iav_label_mapping, 250, threshold, batch_size, device)
                if variant_predictions:
                    _, variant_iav_label, variant_iav_conf = variant_predictions[0]
            
            append_prediction_to_csv(output_csv_file, sequence, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf)
            final_predictions.append((sequence, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf))
            pbar.update(1)

    # Calculate and display label statistics
    virus_label_statistics = calculate_label_statistics(final_predictions, 'virus')
    sorted_virus_label_statistics = sorted(virus_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    top_virus_labels = [label for label, stats in sorted_virus_label_statistics[:2]]
    
    for label, stats in sorted_virus_label_statistics[:2]:
        print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Plot the distribution of viral labels
    label_counts = defaultdict(int)
    for _, virus_label, _, _, _, _, _ in final_predictions:
        label_counts[virus_label] += 1

    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Viral Labels')
    plt.axis('equal')
    plt.show()

    # Calculate and display statistics for CoV-2 variants
    cov2_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'sars_cov_2']
    cov2_label_statistics = calculate_label_statistics(cov2_predictions, 'variant')
    sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    print("\nTop Variant Labels for sars_cov_2:")
    for label, stats in sorted_cov2_label_statistics[:1]:
        print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Calculate and display statistics for IAV variants
    iav_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'influenza_a']
    iav_label_statistics = calculate_label_statistics(iav_predictions, 'variant')
    sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

    print("\nTop Variant Labels for influenza_a:")
    for label, stats in sorted_iav_label_statistics[:1]:
        print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Remove the temporary CSV file
    os.remove(temp_csv_file)

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Predict labels for RNA sequences from a FASTQ file using a pre-trained DNABERT model and calculate label statistics.")
    parser.add_argument("--fastq_path", type=str, required=True, help="Path to input FASTQ file (e.g. path/to/fasta/sample.fasta)")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing models (e.g. path/to/DNABert2/model/directory)")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to the directory containing CSV files used to train models")
    parser.add_argument("--threshold", type=float, required=True, help="Confidence threshold for predictions.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing sequences.")
    
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    mrna_predictor(args.fastq_path, args.model_dir, args.csv_dir, args.threshold, args.batch_size)
