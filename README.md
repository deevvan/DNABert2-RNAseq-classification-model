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

Python script for sequence classification of reads in RNA-Seq fasta/fa, fastq/fq, fasta.gz/fa.gz, fastq.gz/fq.gz file. 

Script: mrna_predictor.py

        python3 mrna_predictor.py --fastq_path /path/to/input.fasta \
                          --model_dir /path/to/main_model_directory \
                          --csv_dir /path/to/finetuning/csv/file \
                          --prediction_output_dir /path/to/output/directory \
                          --virus_threshold 0.95 \
                          --variant_threshold 0.95 \ 
                          --batch_size 5000 \
                          --rbd_fasta_file /path/to/RBD_reference_genome.fasta


### Overview

The application script involves extracting sequences from input file, to firstly predict virus labels using Virus model. Subsequently, reads with virus_label values of influenza_a are subjected to IAV model to be classified as respective IAV subtype. Similarly, reads with virus_label of sars_cov_2 are subjected to COV model to be classified as respective SARS-CoV-2 variant. 


### Script Parameters

The script requires several command-line arguments to specify input files and parameters for model prediction:

- **--fastq_path**: Path to the input FASTQ/FASTA file containing mRNA sequences.
- **--model_dir**: Directory containing pre-trained model files.
- **--csv_dir**: Directory containing label CSV files for models.
- **--prediction_output_dir**: Directory where the prediction output CSV file will be saved.
- **--virus_threshold**: Confidence threshold for virus model predictions.
- **--variant_threshold**: Confidence threshold for variant model predictions.
- **--batch_size**: Number of sequences to process in each batch.
- **--rbd_fasta_file**: Path to the RBD reference FASTA file for alignment.

### Helper Functions:

#### a) is_valid_sequence

This function checks if a DNA sequence contains only valid nucleotides (A, T, G, C) and is within a specified length range (50-250 bases).

#### b) extract_sequences_from_fastq

Extracts sequences from a FASTQ file, handling both uncompressed and gzipped files. It yields valid sequences based on the criteria defined in `is_valid_sequence`.

#### c) extract_sequences_from_fasta

Similar to `extract_sequences_from_fastq`, this function extracts sequences from a FASTA file, handling both uncompressed and gzipped files.

#### d) preprocess_sequence

Truncates a sequence to a specified maximum length, ensuring it does not exceed the model's input size requirement.

#### e) predict_sequences

Utilizes a pre-trained model to predict labels for a batch of DNA sequences. It outputs the predicted label and confidence score for each sequence based on a specified threshold.

#### f) calculate_label_statistics

Calculates statistics for predicted labels, such as count, average confidence, and Z-score, for either virus or variant classifications.

#### g) initialize_output_csv

Initializes a CSV file for storing prediction results, writing appropriate headers.

#### h) append_prediction_to_csv

Appends prediction results to the CSV file, adding new rows for each prediction.

#### i) load_rbd_reference

Loads the RBD reference sequence from a specified FASTA file, used for alignment checks.

#### j) sequence_contains_rbd

Uses a pairwise aligner to check if a given sequence aligns with the RBD reference sequence, indicating potential viral variants.

#### k) analyze_predictions

Analyzes prediction results from a CSV file, calculates statistics, and generates visualizations such as pie charts and Circos plots.

### Main Functionality:

The main function of the script, `mrna_predictor`, orchestrates the entire workflow of mRNA sequence classification and viral strain prediction:

1. **Input Validation**: Validates the input FASTQ/FASTA file format.
2. **Sequence Extraction**: Extracts sequences from the input file using the appropriate helper function.
3. **GPU Allocation**: Distributes sequences across available GPUs for parallel processing.
4. **Model Prediction**: Loads pre-trained models and predicts viral labels for each sequence chunk using helper functions.
5. **Output Compilation**: Combines prediction results from multiple processes into a single CSV file.
6. **Result Analysis**: Analyzes the combined predictions and generates visualizations.


### Output Files of Model Application Script:

Output prediction csv:

<img width="1094" alt="image" src="https://github.com/user-attachments/assets/7f8951d2-c8d9-4203-a5ef-85820fca76e1">



Output prediction circos plot:

<img width="803" alt="image" src="https://github.com/user-attachments/assets/b6f9d038-9768-4ba8-a4be-623abec28ca3">



##### NOTE: Finetuning csv files for respective models compressed & saved into finetuning_data_dir using p7zip:
        
        # Debian/Ubuntu 
        sudo apt-get install p7zip-full  
        
        # MacOS 
        brew install p7zip               

        # To zip
        7z a -t7z -mx=9 output.7z input.zip

        # Unzip
        7z x output.7z

