import matplotlib.pyplot as plt
import random
import pandas as pd
import spacy
from gensim.models import KeyedVectors

# Define the function to count the tokens
def count_tokens_with_labels(tags):
    """
    Count tokens with specific labels.

    Parameters:
    tags (list): List of tags.

    Returns:
    list: A list of counts for each label.
    """
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for tag in tags:
        if tag in counts:
            counts[tag] += 1
    return [counts[1], counts[2], counts[3], counts[4]]

def augment_data_with_balanced_classes(df, counts_df, spacy_model, word2vec_model_path, pouvoir_forms):
    """
    Augment data to create balanced classes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be augmented.
    counts_df (pd.DataFrame): The DataFrame containing the counts of each class.
    spacy_model (str): The spaCy model name to use for NLP tasks.
    word2vec_model_path (str): Path to the pre-trained Word2Vec model.
    pouvoir_forms (set): Set of conjugated forms of the verb "pouvoir".

    Returns:
    pd.DataFrame: The augmented DataFrame with balanced classes.
    """
    # Load spaCy model
    spacy_nlp = spacy.load(spacy_model)
    
    # Load Word2Vec model
    model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False)
    
    # Initialize class counts
    class_counts = {
        'eventualite': counts_df.sum()[0],
        'permission': counts_df.sum()[1],
        'possibilité_matérielle_ou_capacité': counts_df.sum()[2],
        'sporadicite': counts_df.sum()[3],
    }
    
    # Calculate the maximum count among all classes
    max_count = max(class_counts.values())
    
    # Augment data until all classes have the same number of elements
    while len(set(class_counts.values())) > 1:
        for idx, tag_list in enumerate(df['counts']):
            # Ensure all elements of tag_list are integers
            tag_list = [int(tag) for tag in tag_list]

            # Skip this row if 'possibilité_matérielle_ou_capacité' is already labeled, or if multiple labels are present
            if tag_list[2] != 0 or sum(tag_list) > 1:
                continue

            try:
                # Find the current class label
                tag = tag_list.index(1) + 1
                # Get the class name based on the label
                classname = list(class_counts.keys())[tag - 1]

                # Update the maximum count at each iteration
                max_count = max(class_counts.values())

                # Check if augmenting this class would exceed the threshold
                if class_counts[classname] >= max_count:
                    continue

                # Get the tokens for the current row
                tokens = df['tokens'][idx]
                tokens = [spacy_nlp(token) for token in tokens]

                # Find all verbs in the sentence that are not auxiliaries and not in the pouvoir_forms list
                candidate_verbs = []
                for token in tokens:
                    for t in token:
                        if t.pos_ == 'VERB' and not t.dep_.startswith('aux') and t.text.lower() not in pouvoir_forms:
                            candidate_verbs.append(t.text)

                if not candidate_verbs:
                    continue

                # Choose a random verb from the list of candidate verbs
                phrase_verb = random.choice(candidate_verbs)

                # Find the most similar word to the identified verb using Gensim
                try:
                    similar_words = model.most_similar(phrase_verb, topn=5)
                    synonyms = [word for word, _ in similar_words]
                    if not synonyms:
                        continue

                    # Randomly choose a synonym to replace the verb
                    new_verb = synonyms[0]

                    # Replace the verb in the original sentence
                    new_phrase = ' '.join([new_verb if token.text == phrase_verb else token.text for token in tokens])

                    # Replace the verb in the original tokens
                    new_tokens = [new_verb if token.text == phrase_verb else token.text for token in tokens]

                    # Update the augmented row
                    df.loc[len(df)] = {
                        'phrase': new_phrase,
                        'tokens': new_tokens,
                        'aam_tags': df['aam_tags'][idx],
                        'counts': df['counts'][idx]  # Ensure to copy the 'counts' column as well
                    }

                    # Update the class count
                    class_counts[classname] += 1

                except KeyError:
                    # Skip if the verb is not found in the word embeddings model
                    continue

            except Exception as e:
                # Log the exception or handle it accordingly
                print(f"Exception occurred: {e}")
                continue

        print(df.tail())

    return df