import os
import pickle
import pandas as pd
import yaml
import numpy as np
import scipy.sparse
from itertools import product
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Tokenizes and stems the given text.

    Parameters:
        text (str): Input text to preprocess.

    Returns:
        str: Stemmed and tokenized text.
    """
    tokens = word_tokenize(text)  # Tokenization
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return ' '.join(stemmed_tokens)

def train(data_path, model_path, tfidf_matrix_path, stop_words, max_features_values, min_df_values, token_pattern, input_statements):
    """
    Trains multiple TF-IDF models with different parameter combinations,
    evaluates their performance using cosine similarity, and saves the best model.

    Parameters:
        data_path (str): Path to the processed dataset.
        model_path (str): Path to save the trained TF-IDF vectorizer.
        tfidf_matrix_path (str): Path to save the TF-IDF matrix.
        stop_words (list or str): Stop words to be used in vectorization.
        max_features_values (list): List of max_features values for tuning.
        min_df_values (list): List of min_df values for tuning.
        token_pattern (str): Tokenization pattern for TF-IDF vectorizer.
        input_statements (list): Sample input statements to evaluate model performance.

    Returns:
        None
    """
    try:
        # Load dataset
        df = pd.read_csv(data_path)
        if "Plot" not in df.columns:
            raise ValueError("Missing 'Plot' column in dataset!")

        # Initialize tracking variables for the best model
        best_vectorizer = None
        best_tfidf_matrix = None
        best_score = float("-inf")
        best_params = {}

        # Iterate over parameter combinations to find the best model
        for max_features, min_df in product(max_features_values, min_df_values):

            # Create TF-IDF Vectorizer with current parameters
            vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                max_features=max_features,
                min_df=min_df,
                token_pattern=token_pattern
            )

            # Compute TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(df["Plot"])

            # Evaluate model using cosine similarity across input statements
            total_similarity = 0
            for statement in input_statements:
                statement = preprocess_text(statement)
                input_vector = vectorizer.transform([statement])
                similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
                total_similarity += similarities.mean()

            # Compute average similarity score
            avg_similarity = total_similarity / len(input_statements)

            # Track the best model based on similarity score
            if avg_similarity > best_score:
                best_score = avg_similarity
                best_vectorizer = vectorizer
                best_tfidf_matrix = tfidf_matrix
                best_params = {"max_features": max_features, "min_df": min_df}
                print(f"New Best Model: max_features={max_features}, min_df={min_df}, score={best_score}")

        # Ensure that a valid model was selected
        if best_vectorizer is None:
            raise RuntimeError("No suitable model was found!")

        # Ensure output directories exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(tfidf_matrix_path).parent.mkdir(parents=True, exist_ok=True)

        # Save the best TF-IDF vectorizer model
        with open(model_path, "wb") as f:
            pickle.dump(best_vectorizer, f)
        print(f"Model saved at: {model_path}")

        # Save the best TF-IDF matrix in compressed format
        scipy.sparse.save_npz(tfidf_matrix_path, best_tfidf_matrix)
        print(f"TF-IDF matrix saved at: {tfidf_matrix_path}")

        # Save the best parameters to a YAML file
        params_path = Path(__file__).parent / r"../best_params.yaml"
        best_params["best_score"] = best_score
        with open(params_path, "w") as f:
            yaml.dump(best_params, f)
        print(f"Best parameters saved at: {params_path}")

    except Exception as e:
        print(f"Error in training: {str(e)}")