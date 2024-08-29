# python script_name.py 
#  --fastq_path <path_to_fastq> 
#  --model_dir <model_directory> 
#  --csv_dir <path to mapping csv_directory> 
#  --prediction_output_dir <path to output_csv directory?
#  --rbd_fasta_file <path_to_rbd_fasta> 
#  --virus_threshold 0.95 
#  --variant_threshold 0.95
#  --batch_size 1024


import os
import torch
import torch.multiprocessing as mp
from transformers import BertForSequenceClassification, AutoTokenizer
import pandas as pd
import csv
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Bio.Align import PairwiseAligner
from Bio import SeqIO
import gzip
import re
import logging
import time
import argparse
from pycirclize import Circos

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#########################################
### Define helper functions ###
#########################################

def is_valid_sequence(sequence):
    """
    Checks if a sequence contains only valid nucleotides and is within a specified length range.

    Parameters:
    - sequence (str): The DNA sequence to check.

    Returns:
    - bool: True if the sequence is valid, False otherwise.
    """
    sequence = sequence.upper()  # Convert sequence to uppercase
    # Check if the sequence length is between 50 and 250 and contains only A, T, G, C
    return 50 <= len(sequence) <= 250 and re.fullmatch(r'[ATGC]+', sequence) is not None

def extract_sequences_from_fastq(fastq_file):
    """
    Extracts sequences from a FASTQ file, handling both uncompressed and gzipped files.
    
    Parameters:
    - fastq_file (str): Path to the FASTQ file.

    Yields:
    - sequence (str): DNA sequence from the FASTQ file.
    """
    # Handle gzipped files
    open_func = gzip.open if fastq_file.endswith(".gz") else open
    with open_func(fastq_file, 'rt') as file:
        while True:
            header = file.readline().strip()  # Read the header line
            if not header:
                break
            sequence = file.readline().strip()  # Read the sequence line
            file.readline().strip()  # Skip the plus line
            file.readline().strip()  # Skip the quality line
            if is_valid_sequence(sequence):  # Yield the sequence if it's valid
                yield sequence

def extract_sequences_from_fasta(fasta_file):
    """
    Extracts sequences from a FASTA file, handling both uncompressed and gzipped files.
    
    Parameters:
    - fasta_file (str): Path to the FASTA file.

    Yields:
    - sequence (str): DNA sequence from the FASTA file.
    """
    # Handle gzipped files
    open_func = gzip.open if fasta_file.endswith(".gz") else open
    with open_func(fasta_file, 'rt') as file:
        sequence = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence and is_valid_sequence(sequence):
                    yield sequence
                sequence = ""
            else:
                sequence += line
        if sequence and is_valid_sequence(sequence):
            yield sequence

def preprocess_sequence(sequence, max_length):
    """
    Truncates a sequence to a specified maximum length.
    
    Parameters:
    - sequence (str): The DNA sequence to preprocess.
    - max_length (int): The maximum allowed length of the sequence.

    Returns:
    - str: Truncated DNA sequence.
    """
    return sequence[:max_length]

def predict_sequences(model, tokenizer, sequences, label_mapping, max_length, threshold, device):
    """
    Predicts labels for a batch of sequences using a pretrained model.
    
    Parameters:
    - model (torch.nn.Module): Pretrained sequence classification model.
    - tokenizer (transformers.AutoTokenizer): Tokenizer corresponding to the model.
    - sequences (list of str): List of DNA sequences to classify.
    - label_mapping (dict): Mapping from model output indices to label names.
    - max_length (int): Maximum sequence length for tokenization.
    - threshold (float): Confidence threshold for classification.
    - device (torch.device): Device to perform computations on.

    Returns:
    - list of tuple: List of predictions with sequence, label, and confidence score.
    """
    model.eval()  # Set the model to evaluation mode
    inputs = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the appropriate device

    with torch.no_grad():  # Disable gradient calculation for efficiency
        outputs = model(**inputs)  
        # Calculate probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1) 
        # Extract maximum probability (confidence score) and associated predicted label for each sequence
        confidence_scores, predicted_labels = torch.max(probs, dim=-1)  

    predictions = []
    # Going through each sequence to obtain the probability and predicted label with max probability
    for i in range(len(sequences)):
        # If sequence's prediction value is less than threshold value change it to Other 
        if confidence_scores[i] < threshold:
            label = 'Other'
        # If sequence's prediction value is higher than threshold value then assign predicted label_name
        else:
            label = label_mapping[predicted_labels[i].item()]
        predictions.append((sequences[i], label, confidence_scores[i].item()))
    return predictions

def calculate_label_statistics(predictions, label_type):
    """
    Calculates statistics for labels in the predictions, such as count, average confidence, and Z-score.
    
    Parameters:
    - predictions (list of dict): List of predictions with sequence, label, and confidence score.
    - label_type (str): Type of label ('virus' or 'variant') to calculate statistics for.

    Returns:
    - dict: Statistics for each label, including count, confidence sum, average confidence, and Z-score.
    """
    label_counts = defaultdict(int)
    label_confidence_sums = defaultdict(float)

    for prediction in predictions:
        if label_type == 'virus':
            virus_label = prediction['Virus_Label']
            virus_conf = prediction['Virus_Confidence']
            if virus_label != 'Other':  # Exclude 'Other_virus' from the count
                label_counts[virus_label] += 1
                label_confidence_sums[virus_label] += virus_conf
        elif label_type == 'variant':
            variant_label = prediction['Variant_Label']
            variant_conf = prediction['Variant_Confidence']
            if variant_label not in ('unknown_variant', 'nonRBD', 'unknown_LPAI'):  # Exclude non-relevant labels
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
    """
    Initializes a CSV file for writing predictions with appropriate headers.
    
    Parameters:
    - output_csv_file (str): Path to the output CSV file.
    """
    with open(output_csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Sequence', 'Virus_Label', 'Virus_Confidence', 'Variant_Label', 'Variant_Confidence'])

def append_prediction_to_csv(output_csv_file, predictions):
    """
    Appends prediction results to the output CSV file.
    
    Parameters:
    - output_csv_file (str): Path to the output CSV file.
    - predictions (list of tuple): List of predictions to append.
    """
    with open(output_csv_file, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(predictions)

def load_rbd_reference(fasta_file):
    """
    Loads the RBD reference sequence from a FASTA file.
    
    Parameters:
    - fasta_file (str): Path to the RBD reference FASTA file.

    Returns:
    - str: RBD reference sequence.
    """
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

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
    - tuple: (boolean, float) indicating if the alignment score is above the threshold and the alignment score itself.
    """
    aligner = PairwiseAligner()
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_penalty
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    aligner.mode = 'local'

    alignments = aligner.align(sequence, rbd_reference)
    
    if alignments:
        max_possible_score = min(len(sequence), len(rbd_reference)) * match_score
        alignment_threshold = max_possible_score * threshold_percentage
        best_alignment = alignments[0]
        alignment_score = best_alignment.score
        return alignment_score >= alignment_threshold, alignment_score
    
    return False, 0.0


def analyze_predictions(output_csv_file, prediction_output_dir, base_filename):
    """
    Analyzes prediction results from a CSV file, calculates statistics, and generates visualizations.

    This function reads the prediction results from a specified CSV file, computes statistics for virus labels 
    and their respective variants, and generates a Circos plot to visualize the distribution of viral strains 
    and their variants. It also prints detailed statistics for the top virus labels and variants based on 
    classification percentages and Z-scores.

    Parameters:
    - output_csv_file (str): Path to the output CSV file containing prediction results.
    - prediction_output_dir (str): Directory to save the output plot file.
    - base_filename (str): Base name derived from the input FASTQ/FASTA file for naming the output plot file.

    Steps:
    1. Load prediction results from the CSV file into a DataFrame.
    2. Calculate statistics for virus labels including count, average confidence, and Z-score.
    3. Display statistics for the top virus labels if their classifications exceed 5%.
    4. Generate a pie chart showing the distribution of virus labels.
    5. Calculate and display statistics for SARS-CoV-2 variants if their classifications exceed 5%.
    6. Calculate and display statistics for Influenza A variants if their classifications exceed 5%.
    7. Create a Circos plot to visualize the distribution of virus labels and their top variants.
    8. Save the Circos plot as a JPEG file.

    Outputs:
    - Printed statistics for virus labels and variants.
    - A Circos plot for visualizing the distribution of virus labels and their variants.
    """
    df = pd.read_csv(output_csv_file)

    virus_label_statistics = calculate_label_statistics(df.to_dict('records'), 'virus')
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

    # Display the top two virus labels if both have more than 5% of classifications
    if len(top_two_percentages) > 1 and all(pct > 5 for pct in top_two_percentages):
        for label, stats in sorted_virus_label_statistics[:2]:
            print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, Z-Score: {stats['Z_Score']:.4f}")
    else:
        # Display only the top label
        label, stats = sorted_virus_label_statistics[0]
        print(f"Virus Label: {label}, Count: {stats['Count']}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Create a pie chart of the viral label distribution
    label_counts = df['Virus_Label'].value_counts().to_dict()
    labels = list(label_counts.keys())
    sizes = list(label_counts.values())
    
    # Calculate percentages
    sars_cov_2_percentage = label_counts.get('sars_cov_2', 0) / total_classifications * 100
    iav_percentage = label_counts.get('influenza_a', 0) / total_classifications * 100
    
    # Display SARS-CoV-2 VOC stats if classified as > 5% or if 100% classified as sars_cov_2
    if sars_cov_2_percentage > 5 or sars_cov_2_percentage == 100:
        cov2_predictions = df[df['Virus_Label'] == 'sars_cov_2']
        cov2_label_statistics = calculate_label_statistics(cov2_predictions.to_dict('records'), 'variant')
        sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

        print("\nTop Variant Labels for sars_cov_2:")
        for label, stats in sorted_cov2_label_statistics[:1]:
            print(f"Variant Label: {label}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, Z-Score: {stats['Z_Score']:.4f}")
    
    # Display IAV strain stats if classified as > 5% or if 100% classified as influenza_a
    if iav_percentage > 5 or iav_percentage == 100:
        iav_predictions = df[df['Virus_Label'] == 'influenza_a']
        iav_label_statistics = calculate_label_statistics(iav_predictions.to_dict('records'), 'variant')
        sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Z_Score'], reverse=True)

        print("\nTop Variant Labels for influenza_a:")
        for label, stats in sorted_iav_label_statistics[:1]:
            print(f"Variant Label: {label}, Average Confidence: {stats['Avg_Confidence']*100:.4f}, Z-Score: {stats['Z_Score']:.4f}")

    # Create a Circos plot of the viral label distribution
    label_counts = df['Virus_Label'].value_counts().to_dict()
    
    # Define sectors with names and sizes for virus labels
    sectors = {label: {'size': count, 'color': plt.cm.tab20(i / len(label_counts))} for i, (label, count) in enumerate(label_counts.items())}
    total_size = sum(v['size'] for v in sectors.values())

    # Initialize Circos with sectors and add space between them
    circos = Circos(
        {k: v['size'] for k, v in sectors.items()},
        space=8  # Adding space between sectors
    )
    circos.text(f"Virus_Labels", r=105, size = 15, weight ="bold", adjust_rotation=True, ha="center", va="center", orientation="horizontal")
    circos.text(f"Variant_Labels", r=50, size = 15, weight ="bold",adjust_rotation=True, ha="center", va="center", orientation="horizontal")
    
    # Add tracks to each sector with unique transparent colors and add labels inside
    for sector in circos.sectors:
        sector_name = sector.name
        sector_size = sectors[sector_name]['size']
        sector_color = sectors[sector_name]['color']
        percentage = (sector_size / total_size) * 100
        
        track = sector.add_track((60, 100), r_pad_ratio=20)
        track.axis(fc=sector_color, ec="black", alpha=0.5)
        
        # Add label with sector name and percentage inside the sector
        label = f"{sector_name}\n({percentage:.1f}%)"
        sector.text(label, r=65, size=15, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")
        
        # Check for specific virus labels and add top variant sector
        if sector_name == 'sars_cov_2':
            cov2_predictions = df[df['Virus_Label'] == 'sars_cov_2']
            cov2_label_statistics = calculate_label_statistics(cov2_predictions.to_dict('records'), 'variant')
            sorted_cov2_label_statistics = sorted(cov2_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)
            
            # Display only the top variant
            if sorted_cov2_label_statistics:
                top_variant, top_stats = sorted_cov2_label_statistics[0]
                var_percentage = (top_stats['Count'] / total_classifications) * 100
                var_color = plt.cm.Paired(0)  # Assign a distinct color for the top variant
                var_label = f"{top_variant}"
                
                variant_track = sector.add_track((0, 50), r_pad_ratio=20)
                variant_track.axis(fc=var_color, ec="black", alpha=0.5)
                variant_track.text(var_label, r=30, size=12, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")

        elif sector_name == 'influenza_a':
            iav_predictions = df[df['Virus_Label'] == 'influenza_a']
            iav_label_statistics = calculate_label_statistics(iav_predictions.to_dict('records'), 'variant')
            sorted_iav_label_statistics = sorted(iav_label_statistics.items(), key=lambda item: item[1]['Count'], reverse=True)
            
            # Display only the top variant
            if sorted_iav_label_statistics:
                top_variant, top_stats = sorted_iav_label_statistics[0]
                var_percentage = (top_stats['Count'] / total_classifications) * 100
                var_color = plt.cm.Paired(0.5)  # Assign a distinct color for the top variant
                var_label = f"{top_variant}"
                
                variant_track = sector.add_track((0, 50), r_pad_ratio=20)
                variant_track.axis(fc=var_color, ec="black", alpha=0.5)
                variant_track.text(var_label, r=30, size=12, color="black", ha="center", va="center", adjust_rotation=True, orientation="vertical")
    
    # Save the Circos plot as a JPEG file with the same base name as the input file
    plot_filename = os.path.join(prediction_output_dir, f"{base_filename}_circos_plot.jpeg")
    circos.savefig(plot_filename, dpi=300) 

    logging.info(f"Circos plot saved to {plot_filename}")
    
    
#########################################
### Main function ###
#########################################


def worker_process(gpu_id, sequence_chunk, model_dir, csv_dir, output_csv_file, virus_threshold, variant_threshold, batch_size, rbd_reference):
    """
    Worker function for processing a chunk of sequences on a specific GPU.

    Parameters:
    - gpu_id (int): The GPU ID assigned to this process.
    - sequence_chunk (list): List of sequences to process.
    - model_dir (str): Directory containing pre-trained model files.
    - csv_dir (str): Directory containing label CSV files for models.
    - output_csv_file (str): Path to the output CSV file.
    - virus_threshold (float): Probability confidence threshold for virus model predictions.
    - variant_threshold (float): Probability confidence threshold for variant model predictions.
    - batch_size (int): Number of sequences to process in each batch.
    - rbd_reference (str): RBD reference sequence.
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    # Load models and tokenizers
    virus_model_path = os.path.join(model_dir, "1_DNABer2_virus_250bp_50overlap_1300epi_complementary/model")
    virus_model = BertForSequenceClassification.from_pretrained(virus_model_path).to(device)
    virus_tokenizer = AutoTokenizer.from_pretrained(virus_model_path)

    cov2_model_path = os.path.join(model_dir, "3_DNABert2_RBD_3mil_wt_nonvoc_100k_epi_250bp_50overlap_complementary_old/model")
    cov2_model = BertForSequenceClassification.from_pretrained(cov2_model_path).to(device)
    cov2_tokenizer = AutoTokenizer.from_pretrained(cov2_model_path)

    iav_model_path = os.path.join(model_dir, "2_DNABert2_WGS_IAV_strains_250bp_50overlap_complementary_3k_epi_old/model")
    iav_model = BertForSequenceClassification.from_pretrained(iav_model_path).to(device)
    iav_tokenizer = AutoTokenizer.from_pretrained(iav_model_path)

    # Load label mappings
    virus_labels_csv_path = os.path.join(csv_dir, "WGS_by_virus_5labels_250bp_50overlap_complementary_1300epi.csv")
    virus_df = pd.read_csv(virus_labels_csv_path)
    virus_label_mapping = {i: label for i, label in enumerate(virus_df['label_name'].unique())}

    cov2_labels_csv_path = os.path.join(csv_dir, "RBD_nucleotides_3mil_wt_nonvoc_100k_epi_250bp_50overlap_complementary.csv")
    cov2_df = pd.read_csv(cov2_labels_csv_path)
    cov2_label_mapping = {i: label for i, label in enumerate(cov2_df['label_name'].unique())}

    iav_labels_csv_path = os.path.join(csv_dir, "WGS_IAV_strains_250bp_50overlap_complementary_3k_epi.csv")
    iav_df = pd.read_csv(iav_labels_csv_path)
    iav_label_mapping = {i: label for i, label in enumerate(iav_df['label_name'].unique())}

    # Initialize output file for this worker process
    initialize_output_csv(output_csv_file)

    batch_sequences = []
    batch_output = []

    for sequence in sequence_chunk:
        preprocessed_sequence = preprocess_sequence(sequence, 250)
        batch_sequences.append(preprocessed_sequence)

        if len(batch_sequences) == batch_size:
            # Process the batch
            virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, virus_threshold, device)
            for seq, virus_label, virus_conf in virus_predictions:
                variant_label = ''
                variant_conf = 0.0
                if virus_label == 'sars_cov_2':
                    rbd_aligned, rbd_alignment_score = sequence_contains_rbd(seq, rbd_reference)
                    if rbd_aligned:
                        variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, variant_threshold, device)
                        _, variant_label, variant_conf = variant_predictions[0]
                        if variant_conf < variant_threshold:
                            variant_label = 'unknown_variant'
                    else:
                        variant_label = 'nonRBD'
                elif virus_label == 'influenza_a':
                    variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, variant_threshold, device)
                    _, variant_label, variant_conf = variant_predictions[0]
                    if variant_conf < variant_threshold:
                        variant_label = 'unknown_LPAI'
                batch_output.append([seq, virus_label, virus_conf, variant_label, variant_conf])

            # Write batch output to CSV
            append_prediction_to_csv(output_csv_file, batch_output)
            batch_sequences = []
            batch_output = []

    # Process remaining sequences in the last batch
    if batch_sequences:
        virus_predictions = predict_sequences(virus_model, virus_tokenizer, batch_sequences, virus_label_mapping, 250, virus_threshold, device)
        for seq, virus_label, virus_conf in virus_predictions:
            variant_label = ''
            variant_conf = 0.0
            if virus_label == 'sars_cov_2':
                rbd_aligned, rbd_alignment_score = sequence_contains_rbd(seq, rbd_reference)
                if rbd_aligned:
                    variant_predictions = predict_sequences(cov2_model, cov2_tokenizer, [seq], cov2_label_mapping, 250, variant_threshold, device)
                    _, variant_label, variant_conf = variant_predictions[0]
                    if variant_conf < variant_threshold:
                        variant_label = 'unknown_variant'
                else:
                    variant_label = 'nonRBD'
            elif virus_label == 'influenza_a':
                variant_predictions = predict_sequences(iav_model, iav_tokenizer, [seq], iav_label_mapping, 250, variant_threshold, device)
                _, variant_label, variant_conf = variant_predictions[0]
                if variant_conf < variant_threshold:
                    variant_label = 'unknown_LPAI'
            batch_output.append([seq, virus_label, virus_conf, variant_label, variant_conf])

        # Write final batch output to CSV
        append_prediction_to_csv(output_csv_file, batch_output)

def mrna_predictor(fastq_path, model_dir, csv_dir, prediction_output_dir, virus_threshold, variant_threshold, batch_size, rbd_fasta_file):
    """
    Main function to classify mRNA sequences and predict viral strains using pre-trained models.

    Parameters:
    - fastq_path (str): Path to the input FASTQ/FASTA file.
    - model_dir (str): Directory containing pre-trained model files.
    - csv_dir (str): Directory containing label CSV files for models.
    - prediction_output_dir (str): Directory to save the prediction output CSV file.
    - virus_threshold (float): Probability confidence threshold for virus model predictions.
    - variant_threshold (float): Probability confidence threshold for variant model predictions.
    - batch_size (int): Number of sequences to process in each batch.
    - rbd_fasta_file (str): Path to the RBD reference FASTA file.
    """
    logging.info("Starting mRNA prediction process...")

    # Start timing the process
    start_time = time.time()

    # Load RBD reference sequence
    rbd_reference = load_rbd_reference(rbd_fasta_file)
    
    if fastq_path.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
        sequence_generator = extract_sequences_from_fastq(fastq_path)
    elif fastq_path.endswith((".fasta", ".fa", ".fasta.gz", ".fa.gz")):
        sequence_generator = extract_sequences_from_fasta(fastq_path)
    else:
        raise ValueError("Input file must be in FASTQ, FASTA, FASTQ.GZ, or FASTA.GZ format.")

    total_sequences = list(sequence_generator)
    num_sequences = len(total_sequences)

    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No GPU available. Please run on a machine with at least one GPU.")

    # Split sequences into chunks based on number of GPUs
    chunk_size = num_sequences // num_gpus
    sequence_chunks = [total_sequences[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    base_filename = os.path.basename(fastq_path).rsplit('.', 1)[0]

    # Create output files for each worker
    output_files = [os.path.join(prediction_output_dir, f"{base_filename}_prediction_gpu_{i}.csv") for i in range(num_gpus)]

    # Spawn processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, sequence_chunks[gpu_id], model_dir, csv_dir, output_files[gpu_id], virus_threshold, variant_threshold, batch_size, rbd_reference))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Combine all output CSV files into a single file
    combined_output_file = os.path.join(prediction_output_dir, f"{base_filename}_prediction.csv")
    initialize_output_csv(combined_output_file)

    for output_file in output_files:
        df = pd.read_csv(output_file)
        df.to_csv(combined_output_file, mode='a', header=False, index=False)
        # Remove the individual output file
        os.remove(output_file)

    # Analyze combined predictions and pass base_filename
    analyze_predictions(combined_output_file, prediction_output_dir, base_filename)

    # Log the total time taken for the process
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken for the process: {total_time:.2f} seconds")

    
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Predict viral strains from mRNA sequences using pre-trained models.")
    parser.add_argument("--fastq_path", type=str, required=True, help="Path to the input FASTQ/FASTA file.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing pre-trained model files.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing label CSV files for models.")
    parser.add_argument("--prediction_output_dir", type=str, required=True, help="Directory to save the prediction output CSV file.")
    parser.add_argument("--virus_threshold", type=float, required=True, help="Probability confidence threshold for virus model predictions.")
    parser.add_argument("--variant_threshold", type=float, required=True, help="Probability confidence threshold for variant model predictions.")
    parser.add_argument("--batch_size", type=int, required=True, help="Number of sequences to process in each batch.")
    parser.add_argument("--rbd_fasta_file", type=str, required=True, help="Path to the RBD reference FASTA file.")

    # Parse arguments
    args = parser.parse_args()
    
    # Run predictor
    mrna_predictor(
        fastq_path=args.fastq_path,
        model_dir=args.model_dir,
        csv_dir=args.csv_dir,
        prediction_output_dir=args.prediction_output_dir,
        virus_threshold=args.virus_threshold,
        variant_threshold=args.variant_threshold,
        batch_size=args.batch_size,
        rbd_fasta_file=args.rbd_fasta_file
    )
