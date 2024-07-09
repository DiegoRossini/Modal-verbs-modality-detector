from dataset_creation_functions import *

# Annotators and corpus paths
annotateurs = ['Anna', 'Tanina', 'Andrea']
corpus_path = '/path/to/corpus'
context_corpus_path = '/path/to/context_corpus'

# Load annotation data
df_train = load_annotation_data(annotateurs, corpus_path)
df_context = load_context_annotation_data(annotateurs, context_corpus_path)

# Generate verb forms for "pouvoir"
pouvoir_forms = generate_pouvoir_forms()

# Label dictionary and list definitions
label_dict = {
    'éventualité': 1,
    'permission': 2,
    'possibilité matérielle ou capacité': 3,
    'sporadicité': 4,
}

label_list = [
    "O",
    "éventualité",
    "permission",
    "possibilité matérielle ou capacité",
    "sporadicité",
]

# Process context to remove all forms of "pouvoir" to avoid confusing the model
df_train['context_before'], df_train['context_after'] = zip(*df_train['phrase'].apply(
    lambda phrase: process_context(
        phrase, 
        df_context.loc[df_train[df_train['phrase'] == phrase].index[0], 'phrase']
    )
))

# Merge contexts and update 'phrase' column
df_train['phrase'] = df_train.apply(merge_contexts, axis=1)

# Drop unnecessary columns
df_train.drop(['context_before', 'context_after'], axis=1, inplace=True)

# Generate tokens and labels columns
df_tokens_train = get_tokens_column(df_train, pouvoir_forms, label_dict)

# Merge token labels
df_merged_train = merge_token_labels(df_train)

# Reset index if needed
df_merged_train = df_merged_train.reset_index(drop=True)

# Clean data by removing empty rows
df_merged_train = df_merged_train[df_merged_train['aam_tags'].apply(sum) != 0]
df_merged_train = df_merged_train.reset_index(drop=True)

# Save DataFrame to TSV file
df_merged_train.to_csv('./df_merged_train.tsv', sep='\t', index=False)