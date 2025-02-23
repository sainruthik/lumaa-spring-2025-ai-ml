from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import yaml
import pandas as pd
import scipy.sparse
import os
from difflib import get_close_matches
from src import Create_model
from tabulate import tabulate
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmer for text preprocessing
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Tokenizes and stems the given text to normalize it for better matching.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        str: The preprocessed text.
    """
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def find_similar_titles(input_text, data, top_n=5):
    """
    Finds movies with similar titles using string matching.
    
    Parameters:
        input_text (str): User input query.
        data (DataFrame): The dataset containing movie titles.
        top_n (int): Number of similar titles to return.
    
    Returns:
        DataFrame: Movies with similar titles.
    """
    possible_titles = data["title"].tolist()
    matched_titles = get_close_matches(input_text, possible_titles, n=top_n, cutoff=0.6)
    
    # Find titles that contain the input text as a substring
    substring_matches = data[data["title"].str.contains(input_text, case=False, na=False)]
    
    # Combine both matching methods and remove duplicates
    all_matches = pd.concat([
        data[data["title"].isin(matched_titles)],
        substring_matches
    ], ignore_index=True).drop_duplicates()
    
    return all_matches[["title", "genres", "vote_average", "Plot"]]

def recommend_movies(input_text, vectorizer, tfidf_matrix, data, top_n=5):
    """
    Recommends movies based on the user's input statement by computing cosine similarity
    between the input and movie plots. Also searches for similar movie titles.
    
    Parameters:
        input_text (str): User query or input statement.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): The TF-IDF matrix of movie plots.
        data (DataFrame): The original dataset with movie information.
        top_n (int): Number of recommendations to return.
    
    Returns:
        DataFrame: Top recommended movies with title, genres, vote_average, and similarity score.
    """
    # Find similar titles first
    title_matches = find_similar_titles(input_text, data, top_n)
    
    # Preprocess the input text
    input_text = preprocess_text(input_text)
    
    # Convert input text into TF-IDF vector
    input_vector = vectorizer.transform([input_text])
    
    # Compute cosine similarity between input and all movie plots
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # If no similarity is found, refine the query using title-based matches
    if max(similarity_scores) == 0 and not title_matches.empty:
        refined_query = " ".join(title_matches["title"] + " " + title_matches["Plot"] + " " + title_matches["genres"])
        input_vector = vectorizer.transform([refined_query])
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
        title_matches.drop(columns=["Plot"], inplace=True)
    
    # Get indices of top N similar movies
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # Retrieve movie details for recommendations
    similarity_recommendations = data.iloc[top_indices][["title", "genres", "vote_average"]].copy()
    
    # Combine title-based and similarity-based recommendations, avoiding duplicates
    final_recommendations = pd.concat([title_matches, similarity_recommendations], ignore_index=True)
    
    # Drop "Plot" column if present
    if "Plot" in final_recommendations.columns:
        final_recommendations.drop(columns=["Plot"], inplace=True)
    
    return final_recommendations.drop_duplicates().head(top_n)

if __name__ == "__main__":
    # Load parameters from params.yaml
    recommendation_params = yaml.safe_load(open("params.yaml"))["Recommandation"]
    
    # Load the trained TF-IDF vectorizer
    model_path = recommendation_params["model_path"]
    if not os.path.exists(model_path):
        print("Recommendation model not found. Creating model...")
        Create_model.Creating_model()
    
    vectorizer = joblib.load(model_path)
    
    # Load the TF-IDF matrix 
    tfidf_matrix_path = recommendation_params["tfidf_matrix"]
    tfidf_matrix = scipy.sparse.load_npz(tfidf_matrix_path)
    
    # Load the dataset
    data_path = recommendation_params["data"]
    data = pd.read_csv(data_path)
    
    # Interactive recommendation loop
    input_statement = input("\nWhich kind of movies do you want to watch today? \nEnter 'exit' to stop recommendations.\n")
    
    while input_statement.lower() != "exit":
        recommendations = recommend_movies(input_statement, vectorizer, tfidf_matrix, data, top_n=5)
        
        # Print the results in a properly formatted table
        print("\nRecommended Movies:\n")
        print(tabulate(recommendations, headers="keys", tablefmt="fancy_grid", showindex=False))
        
        input_statement = input("\nWhich kind of movies do you want to watch today? \nEnter 'exit' to stop recommendations.\n")