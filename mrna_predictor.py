# python script_name.py 
#  --fastq_path <path_to_fastq> 
#  --model_dir <model_directory> 
#  --csv_dir <csv_directory> 
#  --rbd_fasta_file <path_to_rbd_fasta> 
#  --threshold 0.95 
#  --batch_size 1024

import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import pandas as pd
import argparse
import csv
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from Bio import SeqIO

#########################################
### Define helper functions ###
#########################################

# Function to extract sequences from a FASTQ file
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

# Function to extract sequences from a FASTA file
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

# Preprocess a sequence to a specified maximum length
def preprocess_sequence(sequence, max_length):
    return sequence[:max_length]

# Predict the labels and confidence scores for a batch of sequences
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

# Calculate statistics for each label type (virus or variant) based on predictions
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

# Initialize the CSV output file with the appropriate headers
def initialize_output_csv(output_csv_file):
    with open(output_csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Virus_Label', 'Virus_Confidence', 'Variant_COV_Label', 'Variant_COV_Confidence', 'Variant_IAV_Label', 'Variant_IAV_Confidence'])

# Append predictions to the output CSV file
def append_prediction_to_csv(output_csv_file, predictions):
    with open(output_csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)

# Load the RBD reference sequence from a FASTA file
def load_rbd_reference(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# Check if a sequence contains the RBD reference using pairwise alignment
def sequence_contains_rbd(sequence, rbd_reference, match_score=2, mismatch_penalty=-5, gap_open=-2, gap_extend=-1, threshold_percentage=0.8):
    """
    Use PairwiseAligner to check if the sequence aligns with the RBD reference sequence.
    
    Parameters:
    - sequence (str): The nucleotide sequence to check.
    - rbd_reference (str): The RBD reference sequence.
    - match_score (int): The score for a match in alignment.
    - mismatch_penalty (int): The penalty for a mismatch in alignment.
    - gap_open (float): The penalty for opening a gap in alignment.
    - gap_extend (float): The penalty for extending a gap in alignment.
    - threshold_percentage (float): The percentage of the maximum possible score to use as the threshold.
    
    Returns:
    - boolean: True if the sequence aligns with the RBD reference sequence with a score above the calculated threshold.
    """
    # Initialize the aligner
    aligner = PairwiseAligner()
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_penalty
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    aligner.mode = 'local'  # Smith-Waterman local alignment
    
    # Perform the alignment
    alignments = aligner.align(sequence, rbd_reference)
    
    if alignments:
        # Calculate the maximum possible alignment score for the sequence length
        max_possible_score = min(len(sequence), len(rbd_reference)) * match_score
        
        # Calculate the dynamic threshold based on the sequence length and the specified percentage
        alignment_threshold = max_possible_score * threshold_percentage
        
        # Check if the best alignment score is above the calculated threshold
        best_alignment = alignments[0]
        alignment_score = best_alignment.score  # The score of the best alignment
        if alignment_score >= alignment_threshold:
            return True
    
    return False


#########################################
### Main function ###
#########################################

def mrna_predictor(fastq_path, model_dir, csv_dir, threshold, batch_size, rbd_fasta_file):
    
    # Extract the base filename (without extension) from the input FASTQ/FASTA file path
    base_filename = os.path.basename(fastq_path).rsplit('.', 1)[0]
    
    # Path to the output CSV file where predictions will be stored
    output_csv_file = os.path.join(model_dir, f"{base_filename}_prediction.csv")
    
    # Virus classification model & associated CSV to map labels
    virus_model_path = os.path.join(model_dir, "1_DNABer2_virus_250bp_200overlap_900epi_complementary/model")
    virus_labels_csv_path = os.path.join(csv_dir, "WGS_by_virus_finetune_250bp_200overlap_900epi_complementary.csv")
    
    # InfluenzaA classification model & associated CSV to map labels
    #iav_model_path = os.path.join(model_dir, "2_DNABert2_HA_NA_IAV_250bp_50overlap_complementary_3k_epi/model") 
    #iav_labels_csv_path = os.path.join(csv_dir, "HA_NA_IAV_strains_250bp_50overlap_complementary_3k_epi.csv")
    iav_model_path = os.path.join(model_dir, "2_DNABert2_WGS_IAV_strains_250bp_50overlap_complementary_3k_epi/model") 
    iav_labels_csv_path = os.path.join(csv_dir, "WGS_IAV_strains_250bp_50overlap_complementary_3k_epi.csv")

    # SARS-CoV-2 classification model & associated CSV to map labels
    cov2_model_path = os.path.join(model_dir, "3_DNABer2_RBD_3mil_wo_nonvoc_28k_epi_250bp_50overlap_complementary/model") 
    cov2_labels_csv_path = os.path.join(csv_dir, "RBD_nucleotides_3mil_wo_nonvoc_28k_epi_250bp_50overlap_complementary.csv")
    
    # Load RBD reference sequence from the provided FASTA file
    rbd_reference = load_rbd_reference(rbd_fasta_file)
    
    # Determine file type and extract sequences accordingly
    if fastq_path.endswith(".fastq") or fastq_path.endswith(".fq"):
        sequence_generator = extract_sequences_from_fastq(fastq_path)
    elif fastq_path.endswith(".fasta") or fastq_path.endswith(".fa"):
        sequence_generator = extract_sequences_from_fasta(fastq_path)
    else:
        raise ValueError("Input file must be in FASTQ or FASTA format.")

    # Load the virus classification model and tokenizer
    virus_model = BertForSequenceClassification.from_pretrained(virus_model_path)
    virus_tokenizer = AutoTokenizer.from_pretrained(virus_model_path)

    # Load the virus label mapping from the CSV file
    virus_df = pd.read_csv(virus_labels_csv_path)
    virus_label_mapping = {i: label for i, label in enumerate(virus_df['label_name'].unique())}

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    virus_model.to(device)

    # Initialize the output CSV file
    initialize_output_csv(output_csv_file)

    # Load the SARS-CoV-2 and InfluenzaA classification models and tokenizers
    cov2_model = BertForSequenceClassification.from_pretrained(cov2_model_path)
    cov2_tokenizer = AutoTokenizer.from_pretrained(cov2_model_path)

    iav_model = BertForSequenceClassification.from_pretrained(iav_model_path)
    iav_tokenizer = AutoTokenizer.from_pretrained(iav_model_path)

    # Load the label mappings for SARS-CoV-2 and InfluenzaA from their respective CSV files
    cov2_df = pd.read_csv(cov2_labels_csv_path)
    cov2_label_mapping = {i: label for i, label in enumerate(cov2_df['label_name'].unique())}

    iav_df = pd.read_csv(iav_labels_csv_path)
    iav_label_mapping = {i: label for i, label in enumerate(iav_df['label_name'].unique())}

    # Move the models to the appropriate device
    cov2_model.to(device)
    iav_model.to(device)

    # Count the total number of sequences in the input file
    total_sequences = sum(1 for _ in sequence_generator)
    sequence_generator = extract_sequences_from_fastq(fastq_path) if fastq_path.endswith(".fastq") or fastq_path.endswith(".fq") else extract_sequences_from_fasta(fastq_path)

    # Process the sequences in batches and predict the labels
    with tqdm(total=total_sequences, desc="Predicting variants") as pbar:
        final_predictions = []
        batch_sequences = []
        for sequence in sequence_generator:
            preprocessed_sequence = preprocess_sequence(sequence, 250)
            batch_sequences.append(preprocessed_sequence)

            # If the batch size is reached, perform predictions
            if len(batch_sequences) == batch_size:
                virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, threshold, device)
                batch_output = []

                for seq, virus_label, virus_conf in virus_predictions:
                    variant_cov_label = 'Unknown'
                    variant_cov_conf = 0.0
                    variant_iav_label = 'Unknown'  # Low Pathogenic Influenza VS High Pathogenic Influenza (HPAI)
                    variant_iav_conf = 0.0

                    # Predict variant labels based on the virus label
                    if virus_label == 'sars_cov_2' and sequence_contains_rbd(seq, rbd_reference):
                        variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, threshold, device)
                        _, variant_cov_label, variant_cov_conf = variant_predictions[0]
                    elif virus_label == 'influenza_a':
                        variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, threshold, device)
                        _, variant_iav_label, variant_iav_conf = variant_predictions[0]
                    
                    # Store the predictions for this batch
                    batch_output.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                    final_predictions.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                
                # Append the batch results to the output CSV file
                append_prediction_to_csv(output_csv_file, batch_output)
                batch_sequences = []
                pbar.update(batch_size)

        # If there are remaining sequences, process them
        if batch_sequences:
            virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, threshold, device)
            batch_output = []

            for seq, virus_label, virus_conf in virus_predictions:
                variant_cov_label = 'Unknown'
                variant_cov_conf = 0.0
                variant_iav_label = 'Unknown'  # Low Pathogenic Influenza VS High Pathogenic Influenza (HPAI)
                variant_iav_conf = 0.0

                # Predict variant labels for the remaining sequences
                if virus_label == 'sars_cov_2' and sequence_contains_rbd(seq, rbd_reference):
                    variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, threshold, device)
                    _, variant_cov_label, variant_cov_conf = variant_predictions[0]
                elif virus_label == 'influenza_a':
                    variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, threshold, device)
                    _, variant_iav_label, variant_iav_conf = variant_predictions[0]
                
                # Store the predictions for this batch
                batch_output.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
                final_predictions.append([seq, virus_label, virus_conf, variant_cov_label, variant_cov_conf, variant_iav_label, variant_iav_conf])
            
            # Append the final batch results to the output CSV file
            append_prediction_to_csv(output_csv_file, batch_output)
            pbar.update(len(batch_sequences))

    # Calculate statistics for the virus labels
    virus_label_statistics = calculate_label_statistics(final_predictions, 'virus')
    sorted_virus_label_statistics = sorted(virus_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)

    total_classifications = sum(item[1]['Count'] for item in sorted_virus_label_statistics)
    
    # Calculate the percentages of the top two virus labels
    if len(sorted_virus_label_statistics) > 1:
        top_two_percentages = [
            sorted_virus_label_statistics[0][1]['Count'] / total_classifications * 100,
            sorted_virus_label_statistics[1][1]['Count'] / total_classifications * 100
        ]
    else:
        top_two_percentages = [sorted_virus_label_statistics[0][1]['Count'] / total_classifications * 100]

    # Display the top two virus labels if both have more than 15% of classifications
    if len(top_two_percentages) > 1 and all(pct > 15 for pct in top_two_percentages):
        for label, stats in sorted_virus_label_statistics[:2]:
            print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")
    else:
        # Display only the top label
        label, stats = sorted_virus_label_statistics[0]
        print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Count the occurrences of each virus label in the final predictions
    label_counts = defaultdict(int)
    for _, virus_label, _, _, _, _, _ in final_predictions:
        label_counts[virus_label] += 1

    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    
    # Plot the distribution of viral labels
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Viral Labels')
    plt.axis('equal')
    plt.show()

    # Calculate percentages for SARS-CoV-2 and InfluenzaA
    sars_cov_2_percentage = label_counts.get('sars_cov_2', 0) / total_classifications * 100
    iav_percentage = label_counts.get('influenza_a', 0) / total_classifications * 100

    # Display SARS-CoV-2 VOC stats if classified as > 15% or if 100% classified as SARS-CoV-2
    if sars_cov_2_percentage > 15 or sars_cov_2_percentage == 100:
        cov2_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'sars_cov_2']
        cov2_label_statistics = calculate_label_statistics(cov2_predictions, 'variant')
        sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

        print("\nTop Variant Labels for sars_cov_2:")
        for label, stats in sorted_cov2_label_statistics[:1]:
            print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Display IAV strain stats if classified as > 15% or if 100% classified as InfluenzaA
    if iav_percentage > 15 or iav_percentage == 100:
        iav_predictions = [(seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf) for seq, v_label, v_conf, cov_label, cov_conf, iav_label, iav_conf in final_predictions if v_label == 'influenza_a']
        iav_label_statistics = calculate_label_statistics(iav_predictions, 'variant')
        sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

        print("\nTop Variant Labels for influenza_a:")
        for label, stats in sorted_iav_label_statistics[:1]:
            print(f"Variant Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']:.4f}, Z-Score: {stats['Z_Score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mRNA Predictor for Virus and Variant Classification")
    
    parser.add_argument("--fastq_path", required=True, help="Path to the input FASTQ/FASTA file.")
    parser.add_argument("--model_dir", required=True, help="Directory containing the models.")
    parser.add_argument("--csv_dir", required=True, help="Directory containing the label CSV files.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Threshold for prediction confidence.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for predictions.")
    parser.add_argument("--rbd_fasta_file", required=True, help="Path to the RBD reference sequence FASTA file.")

    args = parser.parse_args()
    
    mrna_predictor(
        fastq_path=args.fastq_path, 
        model_dir=args.model_dir, 
        csv_dir=args.csv_dir, 
        threshold=args.threshold, 
        batch_size=args.batch_size, 
        rbd_fasta_file=args.rbd_fasta_file
    )
