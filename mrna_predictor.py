# python mrna_predictor.py --fastq_path /path/to/input.fq 
#                           --model_dir /path/to/model_dir 
#                           --csv_dir /path/to/csv_dir 
#                           --prediction_output_dir /path/to/output 
#                           --virus_threshold 0.95 
#                           --variant_threshold 0.95 
#                           --batch_size 2048 
#                           --rbd_fasta_file /path/to/rbd_reference.fasta 
#                           --bowtie2_index /path/to/bowtie2_index 
#                           --ref_genome /path/to/ref_genome

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertForSequenceClassification, AutoTokenizer
from Bio.Align import PairwiseAligner
from Bio import SeqIO
import gzip
import re
import logging
import time
import argparse
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable tokenizers parallelism to avoid fork issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#########################################
### Define helper functions ###
#########################################

def is_valid_sequence(sequence):
    """Checks if a sequence contains only valid nucleotides and is within a specified length range."""
    sequence = sequence.upper()
    return 50 <= len(sequence) <= 250 and re.fullmatch(r'[ATGC]+', sequence) is not None

def extract_sequences_from_fastq(fastq_file):
    """Extracts sequences from a FASTQ file, handling both uncompressed and gzipped files."""
    open_func = gzip.open if fastq_file.endswith(".gz") else open
    sequences = []
    
    with open_func(fastq_file, 'rt') as file:
        while True:
            header = file.readline().strip()
            if not header:
                break
            sequence = file.readline().strip()
            file.readline().strip()  # Skip plus line
            quality = file.readline().strip()  # Quality line
            if is_valid_sequence(sequence):
                sequences.append((header, sequence, quality))
    
    return sequences

def process_sequences(sequences, tokenizer, max_length):
    """Tokenizes and preprocesses a batch of sequences."""
    return tokenizer([seq[1] for seq in sequences], return_tensors='pt', padding=True, truncation=True, max_length=max_length)

def predict_sequences(model, inputs, device):
    """Predicts labels for a batch of tokenized sequences using a pretrained model on the GPU."""
    model.eval()
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence_scores, predicted_labels = torch.max(probs, dim=-1)

    return confidence_scores, predicted_labels

def load_rbd_reference(fasta_file):
    """Loads the RBD reference sequence from a FASTA file."""
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

def sequence_contains_rbd(sequence, rbd_reference, match_score=2, mismatch_penalty=-5, gap_open=-2, gap_extend=-1, threshold_percentage=0.7879):
    """Checks if the sequence aligns with the RBD reference sequence."""
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

def align_to_rbd_and_generate_consensus(sequences, bowtie2_index, ref_genome, output_dir, output_prefix, fastq_format=True):
    """Aligns sequences to RBD and generates consensus using Bowtie2 and ivar."""
    fq_temp_file = os.path.join(output_dir, f"{output_prefix}_rbd_tmp.fq" if fastq_format else f"{output_prefix}_rbd_tmp.fa")
    bam_file = os.path.join(output_dir, f"{output_prefix}_rbd.bam")
    consensus_fasta = os.path.join(output_dir, f"{output_prefix}_RBDconsensus.fa")
    
    # Write sequences to a temporary fastq or fasta file
    with open(fq_temp_file, 'w') as fq_file:
        for i, (header, seq, quality) in enumerate(sequences):
            if fastq_format:
                fq_file.write(f"{header}\n{seq}\n+\n{quality}\n")
            else:
                fq_file.write(f">{header}\n{seq}\n")

    # Align to RBD reference using Bowtie2
    bowtie2_cmd = f"bowtie2 -x {bowtie2_index} -U {fq_temp_file} | samtools view -bS - | samtools sort -o {bam_file}"
    subprocess.run(bowtie2_cmd, shell=True, check=True)
    subprocess.run(f"samtools index {bam_file}", shell=True, check=True)
    
    # Generate consensus using ivar
    mpileup_cmd = f"samtools mpileup -A -d 10000 -Q 20 -f {ref_genome} {bam_file} | ivar consensus -p {consensus_fasta} -t 0.6 -m 1 -n N"
    subprocess.run(mpileup_cmd, shell=True, check=True)
    
    # Cleanup intermediate files
    os.remove(fq_temp_file)
    os.remove(bam_file)
    os.remove(f"{bam_file}.bai")
    
    return consensus_fasta

#########################################
### Main function ###
#########################################

def mrna_predictor(fastq_path, model_dir, csv_dir, prediction_output_dir, virus_threshold, variant_threshold, batch_size, rbd_fasta_file, bowtie2_index, ref_genome):
    """
    Main function to classify mRNA sequences and predict viral strains using pre-trained models.
    """
    logging.info("Starting mRNA prediction process...")

    # Validate input file format
    valid_extensions = [".fastq", ".fq", ".fastq.gz", ".fq.gz"]
    if not any(fastq_path.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Invalid input file format. Supported formats are: {', '.join(valid_extensions)}")

    start_time = time.time()

    try:
        base_filename = os.path.basename(fastq_path).rsplit('.', 1)[0]
        output_csv_file = os.path.join(prediction_output_dir, f"{base_filename}_prediction.csv")
        logging.info(f"Predictions will be saved to {output_csv_file}")
        
        # Load models and tokenizers (keep them in memory)
        #virus_model_path = os.path.join(model_dir, "1_DNABer2_virus_250bp_50overlap_1300epi_4lables_complementary/model")
        virus_model_path = os.path.join(model_dir, "1_DNABer2_virus_multilength_1Xcoverage_ARTsimulated_4lables_complementary/model")
        virus_model = BertForSequenceClassification.from_pretrained(virus_model_path)
        virus_tokenizer = AutoTokenizer.from_pretrained(virus_model_path)

        cov2_model_path = os.path.join(model_dir, "3_DNABert2_whole_RBD_15mil_wo_nonvoc_100k_epi/model")
        cov2_model = BertForSequenceClassification.from_pretrained(cov2_model_path)
        cov2_tokenizer = AutoTokenizer.from_pretrained(cov2_model_path)

        #iav_model_path = os.path.join(model_dir, "2_DNABert2_WGS_IAV_strains_250bp_50overlap_complementary_2498_epi/model")
        iav_model_path = os.path.join(model_dir, "2_DNABert2_WGS_IAV_multilength_1Xcoverage_ARTsimulated_complementary/model")
        iav_model = BertForSequenceClassification.from_pretrained(iav_model_path)
        iav_tokenizer = AutoTokenizer.from_pretrained(iav_model_path)

        # Load label mappings (consistent with finetuning script)
        #virus_df = pd.read_csv(os.path.join(csv_dir, "WGS_by_virus_4labels_250bp_50overlap_complementary_1300epi.csv"), dtype={"label_name": str})
        virus_df = pd.read_csv(os.path.join(csv_dir, "ART_simulated_virus_finetune_1Xcoverage_complementary.csv"), dtype={"label_name": str})
        virus_df['label_number'], virus_label_names = pd.factorize(virus_df['label_name'])
        virus_label_mapping = dict(zip(virus_df['label_number'], virus_df['label_name']))

        cov2_df = pd.read_csv(os.path.join(csv_dir, "RBD_whole_nucleotides_15mil_wo_nonvoc_100k_epi.csv"), dtype={"label_name": str})
        cov2_df['label_number'], cov2_label_names = pd.factorize(cov2_df['label_name'])
        cov2_label_mapping = dict(zip(cov2_df['label_number'], cov2_df['label_name']))

        #iav_df = pd.read_csv(os.path.join(csv_dir, "WGS_IAV_strains_250bp_50overlap_complementary_2498_epi.csv"), dtype={"label_name": str})
        iav_df = pd.read_csv(os.path.join(csv_dir, "ART_simulated_iav_finetune_1Xcoverage_complementary.csv"), dtype={"label_name": str})
        iav_df['label_number'], iav_label_names = pd.factorize(iav_df['label_name'])
        iav_label_mapping = dict(zip(iav_df['label_number'], iav_df['label_name']))


        # Load RBD reference
        rbd_reference = load_rbd_reference(rbd_fasta_file)

        # Extract sequences from FASTQ format
        sequences = extract_sequences_from_fastq(fastq_path)
        total_sequences = len(sequences)

        # Set device and enable multi-GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUs for processing.")
            virus_model = torch.nn.DataParallel(virus_model)
            iav_model = torch.nn.DataParallel(iav_model)
            cov2_model = torch.nn.DataParallel(cov2_model)

        virus_model.to(device)
        cov2_model.to(device)
        iav_model.to(device)

        # Initialize memory storage for sars_cov_2 sequences
        temp_sars_cov_2 = []

        final_output = []  # Store final output for all non-sars_cov_2 sequences

        with tqdm(total=total_sequences, desc="Predicting variants") as pbar:
            for i in range(0, total_sequences, batch_size):
                batch_sequences = sequences[i:i + batch_size]

                # Tokenize and preprocess
                inputs = process_sequences(batch_sequences, virus_tokenizer, 250)

                # Predict virus labels on GPU
                confidence_scores, predicted_labels = predict_sequences(virus_model, inputs, device)

                for idx, seq in enumerate(batch_sequences):
                    header, sequence, quality = seq
                    virus_conf = confidence_scores[idx].item()
                    virus_label = virus_label_mapping[predicted_labels[idx].item()]

                    # Apply threshold for virus prediction
                    if virus_conf < virus_threshold:
                        virus_label = "unspecified"

                    variant_label, variant_conf = 'unspecified', 0.0

                    if virus_label == 'sars_cov_2':
                        # Collect SARS-CoV-2 sequences for further processing
                        temp_sars_cov_2.append([header, sequence, quality, virus_conf])
                    else:
                        # Process non-SARS-CoV-2 sequences directly
                        if virus_label == 'influenza_a':
                            variant_inputs = process_sequences([seq], iav_tokenizer, 250)
                            variant_confidence, variant_pred = predict_sequences(iav_model, variant_inputs, device)
                            variant_conf = variant_confidence.item()
                            variant_label = iav_label_mapping[variant_pred.item()]
                            
                            # Apply threshold for variant prediction
                            if variant_conf < variant_threshold:
                                variant_label = 'unspecified'

                        final_output.append([sequence, virus_label, virus_conf, variant_label, variant_conf])

                pbar.update(batch_size)

        # Align SARS-CoV-2 sequences to RBD and generate consensus
        if temp_sars_cov_2:
            logging.info(f"Generating consensus for {len(temp_sars_cov_2)} SARS-CoV-2 sequences")
            #rbd_aligned_sequences = [(header, sequence, quality) for header, sequence, quality in temp_sars_cov_2]
            rbd_aligned_sequences = [(header, sequence, quality) for header, sequence, quality, _ in temp_sars_cov_2]
            consensus_fasta = align_to_rbd_and_generate_consensus(rbd_aligned_sequences, bowtie2_index, ref_genome, prediction_output_dir, base_filename, fastq_format=True)

            # Load the consensus sequence
            with open(consensus_fasta, 'r') as file:
                consensus_seq = file.read().splitlines()[1]  # Get the consensus sequence

            # Predict variant using the consensus sequence
            consensus_inputs = process_sequences([(None, consensus_seq)], cov2_tokenizer, 250)
            variant_confidence, variant_pred = predict_sequences(cov2_model, consensus_inputs, device)
            consensus_variant_label = cov2_label_mapping[variant_pred.item()]
            consensus_variant_conf = variant_confidence.item()

            # Apply threshold for consensus variant
            if consensus_variant_conf < variant_threshold:
                consensus_variant_label = 'unspecified'

            # Append SARS-CoV-2 sequences with the final variant label
            for header, sequence, quality, virus_conf in temp_sars_cov_2:
                final_output.append([sequence, 'sars_cov_2', virus_conf, consensus_variant_label, consensus_variant_conf])

        # Write final output to CSV
        df = pd.DataFrame(final_output, columns=['Sequence', 'Virus_Label', 'Virus_Confidence', 'Variant_Label', 'Variant_Confidence'])
        df.to_csv(output_csv_file, index=False)

        logging.info("Prediction process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken for the process: {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mRNA Predictor Script with Multi-GPU support")
    parser.add_argument("--fastq_path", type=str, required=True, help="Path to the input FASTQ file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the models")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing the CSV files for label mapping")
    parser.add_argument("--prediction_output_dir", type=str, required=True, help="Directory to save prediction outputs")
    parser.add_argument("--virus_threshold", type=float, default=0.95, help="Virus prediction threshold")
    parser.add_argument("--variant_threshold", type=float, default=0.95, help="Variant prediction threshold")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for processing sequences")
    parser.add_argument("--rbd_fasta_file", type=str, required=True, help="Path to the RBD reference file")
    parser.add_argument("--bowtie2_index", type=str, required=True, help="Path to the Bowtie2 index for alignment")
    parser.add_argument("--ref_genome", type=str, required=True, help="Path to the reference genome for consensus generation")
    
    args = parser.parse_args()

    mrna_predictor(
        fastq_path=args.fastq_path,
        model_dir=args.model_dir,
        csv_dir=args.csv_dir,
        prediction_output_dir=args.prediction_output_dir,
        virus_threshold=args.virus_threshold,
        variant_threshold=args.variant_threshold,
        batch_size=args.batch_size,
        rbd_fasta_file=args.rbd_fasta_file,
        bowtie2_index=args.bowtie2_index,
        ref_genome=args.ref_genome
    )

        
