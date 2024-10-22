# "DNABert2-RNAseq-classification-model"

Python scripts to train DNABert2 models to classify RNASeq reads into respective respiratory virus pathogens IAV, IBV, rhinovirus, RSV and SARS-CoV-2

## Link to Model directory:
        https://drive.google.com/drive/folders/1gZChDgC4dUaqoS9AUK_pxQDtzvHv3Ffr?usp=drive_link

#### DAG that shows extraction pipeline unaligned reads onto fastq file/s to pass onto the DNABERT2 model and generate virus and variant labels for host unaligned reads:

<img width="687" alt="Screenshot 2024-10-22 at 3 15 01 PM" src="https://github.com/user-attachments/assets/14398f49-ff22-40a8-883b-b9f7f0b1690a">


## Environment Requirements: 

#### Conda environment required for nextflow pipeline execution:
      conda env create --file nextflow_env.yml --name nextflow_env

#### pip packages required to train and execute DNABERT2 application scripts:
- pandas
- numpy
- torch
- transformers
- scikit-learn
- matplotlib
- pycirclize
- biopython
- tqdm

      pip install torch transformers pandas numpy matplotlib tqdm biopython pycirclize
  

## Step 1) Finetuning scripts:
Python scripts to create Fine-tuning datasets to train each of the three models in DNABert2 virus classification, DNABert2 IAV subtype classification and DNABert2 hCOV19 VOC classifcation.

### a) Dataset for DNABert2 virus classification:

  Script: finetune_WGS_by_virus.ipynb
  
  Whole genome sequences (WGS) for Influenza A (IAV), Human Rhinovirus, Human Respiratory Syncytial virus (RSV) and SARS-CoV-2 from GISAID (https://gisaid.org/) & NCBI   (https://www.ncbi.nlm.nih.gov/labs/virus) 
  
  Transformed into simulated sequence reads of lengths 50, 75, 100, 150 and 250 bps long fragments with 1X coverage using NCBI simulation tool ART (https://www.niehs.nih.gov/research/resources/software/biostatistics/art)  to mimic RNA-Seq reads and assigned labels of respective viruses the reads come from. 
  
  Number of sequences per label balanced to contain a maximum of the number of sequences corresponding to the the virus label with the least sequences. 
  
  Additional reverse complementary sequences to each fragment generated using Biopython to take into account the negative strandedness in RNAseq reads.
            
### b) DNABert2 IAV strain classification:

  Script: finetune_WGS_by_IAV_strains.ipynb
  
  An option of WGS sequences for major highly pathogenic IAV subtypes collected from GISAID EpiFlu and subjected to the same transformation as with step 1a.
            
### c) DNABert2 hCOV19 VOC classifcation:

  Script: extract_RBD_from_MSA.ipynb
  
  Codon aware multiple sequence alignment file for SARS-CoV-2 WGS submissions in GISAID used to extract the RBD segment of spike gene from all submissions so only RBD segments can be used to train the model.

  Script: finetune_RBD_by_hCOV19_variants.ipynb
  
  Receptor binding domain (RBD) sequences extracted from above extract_RBD_from_MSA.ipynb script and 100K whole RBD segments for each major VOC are written out to trainind dataset.



## Step 2) Train Virus, IAV & COV Models 

Script 1: finetune_script_virus_model.py

Script 2: finetune_script_iav_model.py

Script 3: finetune_script_cov_model.py

Python scripts for training and evaluating a virus sequence classification model, Influenza A subtype classification model and SARS-CoV-2 Variant classification model respectively using DNABERT.

### Overview

This project involves training a sequence classification model to classify respective finetuning dataset sequences using DNABERT, a transformer model fine-tuned for DNA sequence data. The process includes data preparation, model training, evaluation, and visualization of the training metrics.

#### Data Preparation

- The data is read from a CSV file containing balanced number of virus genome sequences across all virus labels, subtype labels & VOC labels from Step 1a/1b/1c.
- NOTE: Number of viral genome sequences are balanced but fragmentation of respective genomes result in unbalanced number of fragments as shown below as a result of difference in genome sizes.
  
- The data is split into training, validation, and test sets with an 80-10-10 split. Data sizes in each split shown below:
  
        | Model | Train    | Evaluation | Test    |
        |-------|----------|------------|---------|
        | Virus | 777126   | 97141      | 97141   |
        | IAV   | 5200000  | 650036     | 650036  |
        | COV   | 6100000  | 771100     | 771100  |


#### Model Training Parameters

- Batch size: 32
- Learning rate: 5e-5
- Number of epochs: 8
- The model is trained using the `Trainer` class from Hugging Face's `transformers` library.

#### Evaluation

- The model is evaluated on the test set to compute accuracy, precision, recall and F1-score.
- The best model is saved, and evaluation results are printed.

#### Model Performances:

1. Virus Model Metrics:
   ![virus_model_confusion_matrix](https://github.com/user-attachments/assets/b06ed81d-dc53-444f-8156-ba7db282da4c)
   ![Virus_Model_metrics_plot](https://github.com/user-attachments/assets/94fc36a8-9cd5-4a09-8249-48323b774c7d)

2. IAV Model Metrics:
   ![iav_model_confusion_matrix](https://github.com/user-attachments/assets/fe93fd34-6c73-44bc-af78-3865454c797b)
   ![IAV_Model_WGS_metrics_plot](https://github.com/user-attachments/assets/6ee3283a-d5df-4b5a-a34e-da345dfeba80)

3. COV Model Metrics:
   ![cov_model_wo_nonVOC_confusion_matrix](https://github.com/user-attachments/assets/a1dc5647-1239-454b-a179-de9ea5ba37a9)
   ![COV_Model_RBD_metrics_plot](https://github.com/user-attachments/assets/b9d4fc02-dbd3-44ac-bc25-2d4ca9290ec0)


## Step 3) Model Application Script 

Python script (mrna_predictor.py) for classifying mRNA sequences and predicting viral strains using pre-trained models. It supports sequence classification for RNA-Seq files in FASTQ/FASTA formats (uncompressed and gzipped), and predicts virus labels and respective variants. The script handles multi-GPU support for large datasets and provides functionality for consensus generation using Bowtie2 and ivar for SARS-CoV-2 sequences.


Script: mrna_predictor.py

        python mrna_predictor.py --fastq_path /path/to/input.fq \
                         --model_dir /path/to/model_dir \
                         --csv_dir /path/to/csv_dir \
                         --prediction_output_dir /path/to/output \
                         --virus_threshold 0.95 \
                         --variant_threshold 0.95 \
                         --batch_size 2048 \
                         --rbd_fasta_file /path/to/rbd_reference.fasta \
                         --bowtie2_index /path/to/bowtie2_index \
                         --ref_genome /path/to/ref_genome


### Overview

The script extracts sequences from the input file, predicts virus labels using a pre-trained virus model, and classifies relevant variants (such as SARS-CoV-2 variants or Influenza A subtypes) using corresponding models. 
#### NOTE: SARS-CoV-2 sequences are further processed to generate consensus using Bowtie2 and ivar and consensus sequences are used for SARS-CoV-2 variant classification.


### Script Parameters

The script requires several command-line arguments to specify input files, directories, and parameters for model prediction:

**--fastq_path**: Path to the input FASTQ/FASTA file (supports .fastq, .fq, .fastq.gz, and .fq.gz).
**--model_dir**: Directory containing pre-trained model files for virus classification and variant prediction.
**--csv_dir**: Directory containing label mapping CSV files for the models.
**--prediction_output_dir**: Directory where the prediction output CSV file will be saved.
**--virus_threshold**: Confidence threshold for virus classification (default: 0.95).
**--variant_threshold**: Confidence threshold for variant classification (default: 0.95).
**--batch_size**: Number of sequences to process in each batch (default: 2048).
**--rbd_fasta_file**: Path to the RBD reference FASTA file for SARS-CoV-2 alignment.
**--bowtie2_index**: Path to the Bowtie2 index for SARS-CoV-2 alignment.
**--ref_genome**: Path to the reference genome for generating consensus.


### Workflow Overview

1.	Input Validation: Validates the input FASTQ/FASTA file format.
2.	Sequence Extraction: Extracts sequences from the input file, ensuring they meet length and nucleotide composition criteria.
3.	Virus Prediction: Uses a pre-trained virus classification model to assign virus labels to each sequence.
4.	Variant Prediction: For SARS-CoV-2 or Influenza A sequences, the script further classifies them into respective variants or subtypes.
5.	SARS-CoV-2 Consensus Generation: Aligns SARS-CoV-2 sequences to the RBD region using Bowtie2 and generates consensus with ivar.
6.	Output Compilation: Saves the prediction results (including virus and variant labels) to a CSV file in the specified output directory.


### Helper Functions:

#### a) is_valid_sequence

This function checks if a DNA sequence contains only valid nucleotides (A, T, G, C) and is within a specified length range (50-250 bases).

#### b) extract_sequences_from_fastq

Extracts sequences from a FASTQ file, handling both uncompressed and gzipped files. It yields valid sequences based on the criteria defined in `is_valid_sequence`.

#### c) preprocess_sequence

Truncates a sequence to a specified maximum length, ensuring it does not exceed the model's input size requirement.

#### d) predict_sequences

Utilizes a pre-trained model to predict labels for a batch of DNA sequences. It outputs the predicted label and confidence score for each sequence based on a specified threshold.

#### e) load_rbd_reference

Loads the RBD reference sequence for local alignment of sequences predicted to have sars_cov_2 virus label.

#### g) sequence_contains_rbd

Checks if a sequence aligns with the RBD reference using local alignment.

#### h) align_to_rbd_and_generate_consensus

Aligns sequences to RBD using Bowtie2 and generates a consensus sequence with ivar.


## Step 4) CIRCOS plot Generator Script 

Python script (mrna_predictor.py) for classifying mRNA sequences and predicting viral strains using pre-trained models. It supports sequence classification for RNA-Seq files in FASTQ/FASTA formats (uncompressed and gzipped), and predicts virus labels and respective variants. The script handles multi-GPU support for large datasets and provides functionality for consensus generation using Bowtie2 and ivar for SARS-CoV-2 sequences.


Script: circos_generator.py

        python mrna_predictor.py --csv path/to/*_prediction.csv --out_dir path/to/output_dir


### Overview

The script calculates virus copy number and count based statistics for each virus and variant label using input prediction csv file generated by mRNA_predictor.py and generates Circos plot. 


### Script Parameters

The script requires several command-line arguments to specify input files, directories, and parameters for model prediction:

**--csv**: Path to the input *_prediction.csv file generated by mrna_predictor.py
**--out_dir**: Output Directory to generate Circos plot for input prediction csv file



### Output Files for Model Application Script & Circos Plot Generator Script:

Output prediction csv:

<img width="1094" alt="image" src="https://github.com/user-attachments/assets/7f8951d2-c8d9-4203-a5ef-85820fca76e1">



Output prediction circos plot:

<img width="803" alt="image" src="https://github.com/user-attachments/assets/b6f9d038-9768-4ba8-a4be-623abec28ca3">




