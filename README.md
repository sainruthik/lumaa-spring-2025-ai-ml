# Movie Recommendation System

## How to Run the Model

Follow these steps to set up and run the recommendation system:

### Prerequisites

- Ensure you have **Git** and **Conda** installed on your system.
- Python dependencies are listed in `requirements.txt`.

### Steps to Run:

1. **Clone the repository**

   ```sh
   git clone https://github.com/sainruthik/lumaa-spring-2025-ai-ml.git
   ```

2. **Navigate to the project folder**

   ```sh
   cd lumaa-spring-2025-ai-ml
   ```

3. **Open a Command Prompt in the project folder**

4. **Create a Conda virtual environment**

   ```sh
   conda create -p env python=3.8 -y
   ```

5. **Activate the virtual environment**

   ```sh
   conda activate ./env
   ```

6. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

7. **Run the recommendation script**

   ```sh
   python Recommendation.py
   ```

8. **First-time execution behavior**

   - If the model does not exist, it will be created.
   - If the model already exists, the next step happens directly.

9. **User prompt interaction**

   - The system will print:
     ```
     "Which kind of movies do you want to watch today?
     Enter 'exit' to stop recommendations."
     ```

10. **Input your movie preference**

    - Enter the genre, movie name or movie statement to get recommendations.

11. **Repeat until exit**

    - The process continues until you type:
      ```
      exit
      ```

### Project Structure and Code Explanation

- `Data/raw/movies.csv`: Contains raw movie data used for training and recommendations. The dataset was sourced from [Kaggle - Millions of Movies](https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies).
- `src/preprocessing.py`: Processes the dataset by cleaning missing values, filtering English movies, and creating a text-based plot representation with stemming for recommendation.
- `src/Create_model.py`: Loads preprocessing and training parameters from `params.yaml`, runs data preprocessing, and calls the training module to build and store the model.
- `src/Train.py`: Trains multiple TF-IDF models with various hyperparameters from `params.yaml`, selects the best model based on cosine similarity, and saves the trained model and TF-IDF matrix.
- `Recommendation.py`: Loads the trained model and dataset, accepts user input, preprocesses text, finds similar movie titles, and computes cosine similarity scores to suggest relevant movies.
- `best_params.yaml`: Stores the best hyperparameters found during training.
- `params.yaml`: Configuration file specifying paths for raw and processed data, model storage, TF-IDF settings, stop words, tokenization patterns, and example movie-related input queries.
- `requirements.txt`: Lists required Python dependencies.

### Notes

- Ensure all dependencies are installed correctly.
- If you face any issues, check the error messages and verify that Conda and Python are set up properly.

Enjoy your movie recommendations!

### Salary Expectations
- 25-30$/hr
