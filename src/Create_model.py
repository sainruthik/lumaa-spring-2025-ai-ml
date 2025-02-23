import yaml
from . import preprocessing
from . import Train

def Creating_model():
    """
    This function orchestrates the creation of a recommendation model.
    It follows these steps:
    1. Loads preprocessing and training parameters from params.yaml.
    2. Calls the preprocessing module to prepare the data.
    3. Calls the training module to train and store the model.
    """

    # Load preprocessing parameters from params.yaml
    with open("./params.yaml", "r") as param_file:
        params = yaml.safe_load(param_file)

    preprocessing_params = params["preprocess"]
    train_params = params["train"]

    # Step 1: Preprocess the data
    preprocessing.preprocess(
        preprocessing_params["input"],  # Input file path
        preprocessing_params["output"]  # Output file path after preprocessing
    )

    # Step 2: Train and store the model
    Train.train(
        train_params["data"],               # Path to training data
        train_params["model"],              # Path to save the trained model
        train_params["tfidf_matrix"],       # Path to store the TF-IDF matrix
        train_params["stop_words"],         # Stop words list
        train_params["max_features_values"], # Max features for vectorization
        train_params["min_df_values"],      # Min document frequency
        train_params["token_pattern"],      # Token pattern for tokenization
        train_params["input_statements"]    # Example input statements for testing
    )

