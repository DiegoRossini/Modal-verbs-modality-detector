from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
import torch


def prepare_datasets_for_huggingface(df, test_size=0.2, eval_size=0.5, random_state=42, drop_last_column=True):
    """
    Split the DataFrame into train, evaluation, and test sets, and convert them into HuggingFace datasets.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    test_size (float): The proportion of the dataset to include in the test split.
    eval_size (float): The proportion of the temp dataset to include in the eval split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    drop_last_column (bool): Whether to drop the last column of the DataFrame.

    Returns:
    tuple: Three HuggingFace datasets for training, evaluation, and testing.
    """
    # Initial split (80% train, 20% temp)
    df_train, df_temp = train_test_split(df, test_size=test_size, random_state=random_state)

    # Split the temp set (50% eval, 50% test of the 20% temp set -> 10% eval, 10% test of total)
    df_eval, df_test = train_test_split(df_temp, test_size=eval_size, random_state=random_state)

    # Reset indices
    df_train.reset_index(drop=True, inplace=True)
    df_eval.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    if drop_last_column:
        # Drop the last column
        df_train = df_train.iloc[:, :-1]
        df_test = df_test.iloc[:, :-1]
        df_eval = df_eval.iloc[:, :-1]

    # Transform the pandas DataFrames into HuggingFace datasets
    dataset_train = Dataset.from_pandas(df_train)
    dataset_test = Dataset.from_pandas(df_test)
    dataset_eval = Dataset.from_pandas(df_eval)

    return dataset_train, dataset_test, dataset_eval


# Subwords alignment with labels
def tokenize_and_align_labels(dataset):
    tokenized_inputs = tokenizer(
        dataset["tokens"],
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(dataset["aam_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def cross_validate_model(test_name, dataset_train, tokenizer, model, data_collator, compute_metrics, k=5, num_epochs=7, learning_rate=2e-5, batch_size=5, output_dir='output'):
    """
    Perform cross-validation training and evaluation on a given dataset.

    Parameters:
    test_name (str): Name of the test for the model.
    dataset_train (Dataset): The HuggingFace Dataset for training data.
    tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing.
    model (PreTrainedModel): The HuggingFace model to train.
    data_collator (DataCollator): Data collator for dynamic padding.
    compute_metrics (function): Function to compute evaluation metrics.
    k (int): Number of folds for cross-validation.
    num_epochs (int): Number of epochs to train the model.
    learning_rate (float): Learning rate for the optimizer.
    batch_size (int): Batch size for training and evaluation.
    output_dir (str): Output directory to save models.

    Returns:
    dict: A dictionary containing average metrics across all folds.
    """
    # Initialize KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Convert your dataset to a list of indices
    indices = np.arange(len(dataset_train))

    # Lists to store results
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"Fold {fold + 1}/{k}")

        # Get train and validation datasets
        train_dataset = dataset_train.select(train_indices)
        eval_dataset = dataset_train.select(val_indices)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/pouvoir_modalite_roberta_large_fold_{fold + 1}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        fold_results.append(metrics)

        # Save the model and tokenizer
        save_directory = f'{output_dir}/roberta_large_{test_name}_fold_{fold + 1}'
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)

    # Calculate average results across folds
    avg_metrics = {
        key: np.mean([fold[key] for fold in fold_results])
        for key in fold_results[0]
    }

    print("Average metrics across all folds:")
    print(avg_metrics)

    return avg_metrics


def evaluate_model(tokenized_dataset, labels, predictions, label2id):
    """
    Evaluate the model by generating the classification report and confusion matrix.

    Parameters:
    tokenized_dataset (Dataset): The tokenized dataset used for testing.
    labels (list): The true labels for the dataset.
    predictions (list): The model predictions.
    label2id (dict): Dictionary mapping of label names to label IDs.

    Returns:
    tuple: Classification report and confusion matrix.
    """
    true_labels = []
    pred_labels = []

    for i, label in enumerate(labels):
        true_label = []
        pred_label = []
        for j, word_id in enumerate(tokenized_dataset[i]["input_ids"]):
            if tokenized_dataset[i]["attention_mask"][j] == 1 and tokenized_dataset[i]["labels"][j] != -100:
                true_label.append(label[j])
                pred_labels.append(predictions[i][j])
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    # Flatten the lists for evaluation
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    pred_labels_flat = [item for sublist in pred_labels for item in sublist]

    # Generate classification report
    report = classification_report(true_labels_flat, pred_labels_flat, target_names=list(label2id.keys()))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels_flat, pred_labels_flat)

    return report, conf_matrix