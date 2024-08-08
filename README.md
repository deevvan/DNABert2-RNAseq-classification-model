# DNABert2-RNAseq-classification-model
Python scripts that are used to train and apply DNABert2 models to classify RNASeq reads to detect coinfections of IAV, IBV, RSV and SARS-CoV-2

## Step 1) Creating fine-tuning datasets to train each of the three models in DNABert2 virus classification, DNABert2 IAV strain classification and DNABert2 hCOV19 VOC classifcation:

### a) Dataset for DNABert2 virus classification:

  Script: finetune_WGS_by_virus.ipynb
  
  Whole genome sequences (WGS) for Influenza A (IAV), Influenza B (IBV), Human Respiratory Syncytial virus (RSV) and SARS-CoV-2 from GISAID (https://gisaid.org/) transformed into 250 bps long                   fragments with 50 bps overlapping sliding windows to mimic RNA-Seq reads and assigned labels of respective viruses the reads come from. Number of sequences balanced to contain a maximum of the                number of sequences corresponding to the the virus label with the least number of sequences across all the virus labels.
            
### b) DNABert2 IAV strain classification:

  Script: finetune_WGS_by_IAV_strains.ipynb
  
  HA and NA segment sequences for major IAV strains collected from GISAID and subjected to the same transformation as with step 1a.
            
### c) DNABert2 hCOV19 VOC classifcation:

  Script: finetune_RBD_by_hCOV19_variants.ipynb
  
  Receptor binding domain (RBD) sequences extracted from GISAID's MSA file for all available WGS used to train the model. RBD sequences subjected to similar transformation of 250 bps long fragments             but with 1 bp overlapping sliding window.


## Step 2a) Train Virus Model 

Script: DNAbert2_finetune_script_virus.py

This repository contains a Python script for training and evaluating a virus sequence classification model using DNABERT.

### Overview

This project involves training a sequence classification model to classify virus genome sequences using DNABERT, a transformer model fine-tuned for DNA sequence data. The process includes data preparation, model training, evaluation, and visualization of the training metrics.

### Features

- **Custom Dataset Class:** A custom dataset class to handle input sequences, attention masks, and labels for training and evaluation.
- **Metrics Computation:** Functions to compute accuracy and F1-score for model evaluation.
- **Custom Callback for Early Stopping:** Implementation of a callback to stop training early if a certain accuracy threshold is reached.
- **Data Preparation:** Reading and preprocessing of virus genome sequences from a CSV file.
- **Model Training and Evaluation:** Training the DNABERT model on the prepared data, and evaluating its performance on a test set.
- **Visualization:** Plotting training and evaluation metrics over epochs.

#### Data Preparation

- The data is read from a CSV file containing virus genome sequences and their labels from Step 1a.
- The labels are encoded as integers.
- The data is split into training, validation, and test sets with an 80-10-10 split.

#### Model Training

- DNABERT model and tokenizer are loaded and configured.
- Data is tokenized and converted into PyTorch tensors.
- Training arguments are set, including learning rate, batch size, number of epochs, and more.
- The model is trained using the `Trainer` class from Hugging Face's `transformers` library.

#### Evaluation

- The model is evaluated on the test set to compute accuracy and F1-score.
- The best model is saved, and evaluation results are printed.

#### Visualization

- Training and evaluation losses, as well as evaluation accuracy and F1-score, are plotted over epochs.
- The plots are saved as an image file.


## Step 2b) Train IAV strain Classification Model 

Script: DNAbert2_finetune_script_IAV.py

This repository contains a Python script for training and evaluating a model to classify Influenza A Virus (IAV) strains using DNABERT.

### Overview

This project involves training a sequence classification model to classify Influenza A Virus (IAV) genome sequences using DNABERT. The process includes data preparation, model training, evaluation, and saving the trained model.

### Features

- **Custom Dataset Class:** A custom dataset class to handle input sequences, attention masks, and labels for training and evaluation.
- **Metrics Computation:** Functions to compute accuracy and F1-score for model evaluation.
- **Custom Callback for Early Stopping:** Implementation of a callback to stop training early if a certain accuracy threshold is reached.
- **Data Preparation:** Reading and preprocessing of IAV genome sequences from a CSV file.
- **Model Training and Evaluation:** Training the DNABERT model on the prepared data, and evaluating its performance on a test set.

#### Data Preparation

- The data is read from a CSV file containing IAV genome sequences and their labels.
- The labels are encoded as integers.
- The data is split into training, validation, and test sets with a 70-15-15 split.

#### Model Training

- DNABERT model and tokenizer are loaded and configured.
- Data is tokenized and converted into PyTorch tensors.
- Training arguments are set, including learning rate, batch size, number of epochs, and more.
- The model is trained using the `Trainer` class from Hugging Face's `transformers` library.

#### Evaluation

- The model is evaluated on the test set to compute accuracy and F1-score.
- The best model is saved, and evaluation results are printed.


## Step 2c) Train SARS-CoV-2 VOC Classification Model 

Script: DNAbert2_finetune_script_sarscov2.py

This repository contains a Python script for training and evaluating a SARS-CoV-2 sequence classification model using DNABERT, with functionality to resume from the latest checkpoint.

### Overview

This project involves training a sequence classification model to classify SARS-CoV-2 genome sequences using DNABERT. The process includes data preparation, model training, evaluation, logging of metrics, and visualization of training metrics.

### Features

- **Custom Dataset Class:** A custom dataset class to handle input sequences, attention masks, and labels for training and evaluation.
- **Metrics Computation:** Functions to compute accuracy and F1-score for model evaluation.
- **Custom Callback for Logging Metrics:** Implementation of a callback to log metrics at the end of each epoch.
- **Data Preparation:** Reading and preprocessing of SARS-CoV-2 genome sequences from a CSV file.
- **Model Training and Evaluation:** Training the DNABERT model on the prepared data, and evaluating its performance on a test set.
- **Checkpoint Resumption:** Checks for the latest checkpoint and resumes training from there if available.
- **Visualization:** Plotting training and evaluation metrics over epochs.

#### Data Preparation

- The data is read from a CSV file containing SARS-CoV-2 genome sequences and their labels.
- The labels are encoded as integers.
- The data is split into training, validation, and test sets with an 80-10-10 split.

#### Model Training

- DNABERT model and tokenizer are loaded and configured.
- Data is tokenized and converted into PyTorch tensors.
- Training arguments are set, including learning rate, batch size, number of epochs, and more.
- The model is trained using the `Trainer` class from Hugging Face's `transformers` library.
- If a checkpoint is found, training resumes from the latest checkpoint.

#### Evaluation

- The model is evaluated on the test set to compute accuracy and F1-score.
- The best model is saved, and evaluation results are printed.

#### Visualization

- Training and evaluation losses, as well as evaluation accuracy and F1-score, are plotted over epochs.
- The plots are saved as an image file.


#### Requirements

- pandas
- numpy
- torch
- transformers
- scikit-learn
- matplotlib


## Step 3) Application script for Sequence Classification

This repository contains a Python script for sequence classification of RNA-seq data using DNABERT. The script processes RNA-seq FASTQ and FASTA files, predicts virus and variant labels, and logs results.

Script: mrna_predictor.py
### Overview

The project involves extracting sequences from FASTQ files, preprocessing them, and predicting their labels using pre-trained DNABERT models. It also calculates and logs statistics for the predicted labels, and visualizes the distribution of virus labels.

### Features

- **Sequence Extraction:** Extracts sequences from FASTQ files.
- **Preprocessing:** Preprocesses sequences to a fixed maximum length.
- **Prediction:** Predicts virus and variant labels using DNABERT models.
- **Label Statistics Calculation:** Calculates and logs statistics for virus and variant labels.
- **Visualization:** Plots the distribution of virus labels.

### Helper Functions

- **extract_sequences_from_fastq:** Extracts sequences from a given FASTQ file.
- **preprocess_sequences:** Truncates sequences to a specified maximum length.
- **predict_sequences:** Predicts labels for sequences using a specified model and tokenizer.
- **calculate_label_statistics:** Calculates statistics for predicted labels.
- **initialize_output_csv:** Initializes the output CSV file for storing predictions.
- **append_prediction_to_csv:** Appends a prediction to the output CSV file.

### Main Function

- **mrna_predictor:** Main function to process the FASTQ file, predict labels, calculate statistics, and visualize results.


### Requirements

- os
- torch
- transformers
- pandas
- csv
- collections
- tqdm
- numpy
- matplotlib

