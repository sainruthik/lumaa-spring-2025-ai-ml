import os
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK datasets for text processing
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize Porter Stemmer
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

def preprocess(input_path, output_path):
    """
    Processes the movie dataset by:
    - Removing missing values in important columns.
    - Filtering only English-language movies.
    - Reducing dataset size to 500 rows.
    - Selecting relevant columns.
    - Creating a combined "Plot" column for text-based analysis.
    - Applying stemming to the plot text.
    - Saving the cleaned dataset to a new file.

    Parameters:
        input_path (str): Path to the raw dataset CSV.
        output_path (str): Path to save the processed dataset.

    Returns:
        DataFrame: Processed dataset.
    """
    try:
        # Load the dataset
        data = pd.read_csv(input_path)

        # Drop rows with missing values in key columns
        required_columns = ["original_language", "overview", "keywords", "genres", "title"]
        data.dropna(subset=required_columns, inplace=True)

        # Filter only English-language movies
        data = data[data["original_language"] == "en"]

        # Limit dataset size to 500 entries
        data = data.iloc[:500]

        # Select relevant columns
        selected_columns = ["id", "title", "genres", "overview", "vote_average", "keywords"]
        data = data[selected_columns]

        # Fill NaN values with empty strings before concatenation
        data.fillna("", inplace=True)

        # Create a new "Plot" column by combining relevant textual data
        data["Plot"] = data["overview"] + " " + data["keywords"] + " " + data["genres"] + " " + data["title"]

        # Retain only necessary columns for further processing
        data = data[["id", "title", "genres", "vote_average", "Plot"]]

        # Apply text preprocessing (stemming)
        data["Plot"] = data["Plot"].astype(str).apply(preprocess_text)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save cleaned dataset
        data.to_csv(output_path, index=False)
        print(f"Data preprocessing complete. Cleaned dataset saved to: {output_path}")

        return data

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
