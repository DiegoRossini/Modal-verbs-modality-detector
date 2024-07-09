from augmentation_functions import *
from dataset_creation_functions import generate_pouvoir_forms
import ast
import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the CSV file
file_path = './df_merged_train_context.tsv'

# Load the CSV file into a DataFrame
df_merged_train = pd.read_csv(file_path, sep="\t")

# Convert columns from strings to lists
columns_to_convert = ["tokens", "aam_tags", "counts"]
for column in columns_to_convert:
    df_merged_train[column] = df_merged_train[column].apply(ast.literal_eval)

# Apply the function to the 'aam_tags' column to count tokens with labels
df_merged_train['counts'] = df_merged_train['aam_tags'].apply(count_tokens_with_labels)

# Create a DataFrame from the 'counts' to easily sum the values for each label
counts_df = pd.DataFrame(df_merged_train['counts'].tolist(), columns=['eventualite', 'permission', 'possibilité_matérielle_ou_capacité', 'sporadicite'])

# Sum the columns to get the totals for each label
label_totals = counts_df.sum()

# Plot the distribution of tokens with labels 1-4
label_totals.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of tokens with labels 1-4')

# Add text labels to each bar in the plot
for i, value in enumerate(label_totals):
    plt.text(i, value, str(value), ha='center', va='bottom')

# Display the plot
plt.show()

# Define parameters for data augmentation
spacy_model = 'fr_dep_news_trf'
word2vec_model_path = './cc.fr.300.vec.gz'
pouvoir_forms = generate_pouvoir_forms()

# Augment the data to create balanced classes
df_augmented = augment_data_with_balanced_classes(df_merged_train, counts_df, spacy_model, word2vec_model_path, pouvoir_forms)

# Specify the path where you want to save the augmented DataFrame
save_path = './df_merged_train_context.csv'

# Save the augmented DataFrame to a CSV file
df_augmented.to_csv(save_path, index=False)

# Apply the function to the 'aam_tags' column again for the augmented data
df_augmented['counts'] = df_augmented['aam_tags'].apply(count_tokens_with_labels)

# Create a new DataFrame from the 'counts' to sum the values for each label
aug_counts_df = pd.DataFrame(df_augmented['counts'].tolist(), columns=['eventualite', 'permission', 'possibilité_matérielle_ou_capacité', 'sporadicite'])

# Sum the columns to get the totals for each label in the augmented data
aug_label_totals = aug_counts_df.sum()

# Plot the distribution of tokens with labels 1-4 for the augmented data
aug_label_totals.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of tokens with labels 1-4 after Augmentation')

# Add text labels to each bar in the plot
for i, value in enumerate(aug_label_totals):
    plt.text(i, value, str(value), ha='center', va='bottom')

# Display the plot
plt.show()