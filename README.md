# Modal Verbs Modality Detector V1

## Overview
The "Modal Verbs Modality Detector" project aims to build a machine learning model that detects and classifies the different modalities of the French verbs (this repo contains a test conducted on the French verb 'pouvoir'). The project leverages data augmentation, cross-validation, and other techniques to ensure the model's robustness and accuracy.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Steps to Train the Model](#steps-to-train-the-model)
3. [Usage](#usage)
4. [Contributors](#contributors)

## Prerequisites
- Python 3.6+
- Libraries: `pandas`, `numpy`, `glob`, `spacy`, `gensim`, `sklearn`, `transformers`, `datasets`
- Pre-trained HuggingFace model and tokenizer
- Pre-trained Word2Vec model for French word embeddings

## Steps to Train the Model
To have the model trained, the following scripts have been called in order:

1. **dataset_creation**: This phase involves loading and preparing the annotated data, specifically for the verb "pouvoir".
2. **augmentation**: Data augmentation is performed to ensure balanced class distribution by replacing verbs with their synonyms.
3. **train_eval**: The final phase involves splitting the data into training, evaluation, and test sets, performing cross-validation, training the model, and evaluating its performance.

## Usage
To test the detector:

1. Load the dataset.
2. Perform data augmentation if necessary.
3. Split the data into train, test, and evaluation sets.
4. Execute cross-validation.
5. Evaluate the model.

## Contributors
- Diego Rossini
- Anna Colli

---

**Note:** This project specifically focuses on the French verb "pouvoir", and the resulting model will only work for detecting and classifying the modalities of "pouvoir".

Feel free to contribute to this project by opening issues or submitting pull requests.