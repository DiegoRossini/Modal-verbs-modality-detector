from train_eval_functions import *
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
import evaluate
import numpy as np
from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
hf_token = "your_token_here"

# Log in
login(token=hf_token)

# Prepare datasets for HuggingFace
dataset_train, dataset_test, dataset_eval = prepare_datasets_for_huggingface(df_merged_train)

# Load the tokenizer for Flaubert
tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased", add_prefix_space=True)

# Tokenize and align labels for the datasets
tokenized_dataset_train = dataset_train.map(tokenize_and_align_labels, batched=True)
tokenized_dataset_test = dataset_test.map(tokenize_and_align_labels, batched=True)
tokenized_dataset_eval = dataset_eval.map(tokenize_and_align_labels, batched=True)

# Initialize DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")

# Load the seqeval metric
seqeval = evaluate.load("seqeval")

# Define label list
label_list = [
    "O",
    "éventualité",
    "permission",
    "possibilité matérielle ou capacité",
    "sporadicité",
]

# Example labels for verification
labels = [label_list[i] for i in dataset_train["aam_tags"][5]]

# Define ID to label mapping
id2label = {
    0: "O",
    1: "éventualité",
    2: "permission",
    3: "possibilité matérielle ou capacité",
    4: "sporadicité",
}

# Define label to ID mapping
label2id = {
    "O": 0,
    'éventualité': 1,
    'permission': 2,
    'possibilité matérielle ou capacité': 3,
    'sporadicité': 4,
}

# Load the model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    "flaubert/flaubert_base_cased", num_labels=5, id2label=id2label, label2id=label2id,
)

# Ask for the test name
test_name = input("Insert the name of the test among: 'corpus_base', 'corpus_base_context', 'corpus_base_augmented', 'corpus_base_augmented_context'")

# Perform cross-validation
avg_metrics = cross_validate_model(test_name, tokenized_dataset_train, tokenizer, model, data_collator, compute_metrics)

# Initialize Trainer with the trained model
trainer = Trainer(
    model=model  # your pre-trained model
)

# Make predictions on the test dataset
predictions, test_labels, _ = trainer.predict(tokenized_dataset_test)

# Get the predicted class labels
predictions = np.argmax(predictions, axis=2)

# Evaluate the model
report, conf_matrix = evaluate_model(tokenized_dataset_test, test_labels, predictions, label2id)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)