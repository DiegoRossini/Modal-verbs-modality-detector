import glob
import pandas as pd
from pyconjugator import Conjugator
import spacy
import stanza
import re
from itertools import zip_longest


def load_annotation_data(annotateurs, corpus_path):
    """
    Load and concatenate annotation data from specified annotators and corpora.

    Parameters:
    annotateurs (list): List of annotator names.
    corpus_path (str): The base path where the corpus folders are located.

    Returns:
    pd.DataFrame: Concatenated DataFrame from all TSV files provided by the annotators.
    """
    dfs_train_paths = []
    
    for annotateur in annotateurs:
        # Paths to the results files of the CFPP and ESLO corpuses
        for corpus in ['CFPP', 'ESLO']:
            for tsv_path in glob.glob(f"{corpus_path}/{annotateur}/{corpus}/Resultats_{annotateur}/*.tsv"):
                dfs_train_paths.append(tsv_path)

    # Reading all the TSV files and concatenating them into a single DataFrame
    dfs = [pd.read_csv(df_path, sep='\t') for df_path in dfs_train_paths]
    df_train = pd.concat(dfs, ignore_index=True)

    # Dropping the last column
    df_train.drop(df_train.columns[-1], axis=1, inplace=True)

    return df_train


def load_context_annotation_data(annotateurs, context_corpus_path):
    """
    Load and concatenate context annotation data from specified annotators and corpora.

    Parameters:
    annotateurs (list): List of annotator names.
    context_corpus_path (str): The base path where the context corpus folders are located.

    Returns:
    pd.DataFrame: Concatenated DataFrame from all TSV files provided by the annotators with context.
    """
    dfs_train_paths_context = []
    
    for annotateur in annotateurs:
        # Paths to the results files of the CFPP and ESLO corpuses
        for corpus in ['CFPP', 'ESLO']:
            for tsv_path in glob.glob(f"{context_corpus_path}/{annotateur}/{corpus}/Resultats_{annotateur}/*.tsv"):
                dfs_train_paths_context.append(tsv_path)

    # Reading all the TSV files and concatenating them into a single DataFrame
    dfs = [pd.read_csv(df_path, sep='\t') for df_path in dfs_train_paths_context]
    df_context = pd.concat(dfs, ignore_index=True)

    # Dropping the last column
    df_context.drop(df_context.columns[-1], axis=1, inplace=True)

    return df_context


def generate_pouvoir_forms(lang='fr', verb='pouvoir', spacy_model='fr_core_news_sm'):
    """
    Generate a set of all possible forms of the verb 'pouvoir' (without auxiliaries).

    Parameters:
    lang (str): Language code for the conjugator.
    verb (str): The verb to conjugate.
    spacy_model (str): The name of the spaCy model to use for tokenization.

    Returns:
    set: A set containing different conjugated forms of the verb 'pouvoir'.
    """
    # Initialize the Conjugator and spaCy NLP model
    cg = Conjugator(lang=lang)
    spacy_nlp = spacy.load(spacy_model)

    # Conjugate the verb
    c = cg.conjugate(verb)
    pouvoir_forms = set()

    # Iterate through the conjugation tree to extract verb forms
    for item in c.items():
        if item[0] == 'moods':
            times = item[1].keys()
            for time in times:
                sub_times = item[1][time].keys()
                for sub_time in sub_times:
                    forms = item[1][time][sub_time]
                    for form in forms:
                        doc = spacy_nlp(form)
                        tokens = [token.text for token in doc if token.text.startswith('p')]
                        pouvoir_forms.add(tokens[0])

    return pouvoir_forms


def remove_pouvoir_forms(text, forms):
    words = text.split()
    filtered_words = [word for word in words if word not in forms]
    return ' '.join(filtered_words)


def process_context(target_phrase, context_phrase):
    """
    Split context into before and after target phrase and clean from 'pouvoir' forms.

    Parameters:
    target_phrase (str): The phrase to locate within the context.
    context_phrase (str): The context sentence containing the target phrase.

    Returns:
    tuple: Before and after context phrases cleaned of 'pouvoir' forms.
    """
    target_index = context_phrase.find(target_phrase)
    before_target = context_phrase[:target_index]
    after_target = context_phrase[target_index + len(target_phrase):]

    # Clean up before and after context by removing pouvoir forms
    before_clean = remove_pouvoir_forms(before_target, pouvoir_forms)
    after_clean = remove_pouvoir_forms(after_target, pouvoir_forms)

    return before_clean, after_clean


def merge_contexts(row):
    """
    Concatenate context_before, phrase, and context_after.

    Parameters:
    row (pd.Series): A row of the DataFrame containing context_before, phrase, and context_after.

    Returns:
    str: The merged sentence.
    """
    return f"{row['context_before']} {row['phrase']} {row['context_after']}"


def get_tokens_column(df, pouvoir_forms, label_dict):
    """
    Create a column with tokenized sentences and labels for each token.

    Parameters:
    df (pd.DataFrame): The DataFrame containing phrases to tokenize.
    pouvoir_forms (set): Set of all "pouvoir" verb forms.
    label_dict (dict): Dictionary mapping label names to numerical values.

    Returns:
    pd.DataFrame: DataFrame with additional tokens and aam_tags columns.
    """
    # Initialize Stanza pipeline for French tokenization
    nlp = stanza.Pipeline('fr', processors='tokenize')

    all_tokens = []
    aam_tags = []

    # Assuming df contains a column 'phrase' and multiple labeling columns
    columns_to_iterate = df.columns[1:7]

    for idx, row in df.iterrows():
        text = row['phrase'].strip()
        # Remove extra spaces
        text = re.sub(' +', ' ', text)
        # Delete the pattern $<text>/$ including the enclosing $ signs and the preceding /
        text = re.sub(r'\$|\/\$', '', text)

        # Tokenize the cleaned text
        doc = nlp(text)
        tokens = [word.text for sent in doc.sentences for word in sent.words]
        all_tokens.append(tokens)

        # Determine the appropriate tags for the tokens
        tag = None
        for label in columns_to_iterate:
            if row[label] == 1.0:
                tag = label_dict.get(label)

        # Initialize an empty list for aam tags
        aam_tag = []
        for token in tokens:
            if token not in pouvoir_forms:
                aam_tag.append(0)
            else:
                if tag is not None:
                    aam_tag.append(tag)
                else:
                    aam_tag.append(0)

        aam_tags.append(aam_tag)

    # Add resulting lists as columns to the DataFrame
    df['tokens'] = all_tokens
    df['aam_tags'] = aam_tags

    return df


def merge_token_labels(df_tokens):
    """
    Merge rows with the same sentence and merge aam_tags lists.

    Parameters:
    df_tokens (pd.DataFrame): DataFrame containing tokenized sentences and aam_tags.

    Returns:
    pd.DataFrame: DataFrame with merged labels.
    """
    # Convert the lists to tuples for groupby to prevent list hashing issues
    df_tokens['tokens'] = df_tokens['tokens'].apply(tuple)

    # Custom aggregation function to merge the aam_tags lists by taking the maximum value at each position
    def merge_labels(labels_list):
        # Use zip_longest to fill in missing values with 0 and then take the max at each position
        merged_labels = [max(values) for values in zip_longest(*labels_list, fillvalue=0)]
        return merged_labels

    # Group by 'phrase' and 'tokens' (as tuples), and apply merge_labels to 'aam_tags' column
    df_merged = df_tokens.groupby(['phrase', 'tokens'], as_index=False).agg({'aam_tags': merge_labels})

    # Convert tuples back to list for the 'tokens' column
    df_merged['tokens'] = df_merged['tokens'].apply(list)

    return df_merged
