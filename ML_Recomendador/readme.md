## ML_Recomendador - What the Code Does

- **Environment Setup:**  
  - Imports necessary Python and Spark libraries.  
  - Configures the Spark context and adjusts system paths to ensure Spark libraries are available.

- **Data Loading and Preparation:**  
  - Loads movie ratings data from a text file, where each record contains a user ID, movie ID, and rating (using `::` as the delimiter).  
  - Parses the data into tuples and filters out ratings with a value of 0.  
  - Counts and prints the total number of ratings, distinct users, and distinct movies.

- **Data Splitting:**  
  - Splits the dataset randomly into three subsets:
    - **Training set (70%)**
    - **Validation set (20%)**
    - **Test set (10%)**

- **Model Training and Hyperparameter Tuning:**  
  - Trains multiple ALS (Alternating Least Squares) models on the training subset using varying parameters (different ranks and iteration counts).  
  - Uses the validation set to:
    - Predict movie ratings.
    - Compute performance metrics such as Mean Absolute Error (MAE) and the R² coefficient.
  - Selects the model with the lowest MAE from the validation step.

- **Evaluation on Test Data:**  
  - Retrains the best model on the training data.  
  - Predicts ratings on the test set and evaluates model performance using MAE and R² metrics.

- **Movie Recommendation:**  
  - Loads a file containing the current user's movie ratings, where a rating of 0 means the movie is unrated.  
  - Predicts ratings for the movies the user hasn't rated yet.  
  - Sorts the predictions and selects the top 5 recommendations.  
  - Maps movie IDs to movie titles using a separate lookup file and displays the recommended movie titles.

- **Cleanup:**  
  - Stops the Spark context to free up cluster and local resources.

> **Note:** An updated version named `updated.py` has been added. This version uses the newer DataFrame API and replaces Python 2 with Python 3, offering improved performance, better readability, and enhanced functionality.

---

Feel free to copy this section directly into your GitHub README.md file! Happy coding! 🚀😊
