# Movie-recommendation-system
This project implements a movie recommendation system using content-based filtering. It suggests movies to users based on the similarity of their preferred movie.

Project Overview

Data Loading: The project starts by loading the movie data from the movies.csv file into a pandas DataFrame using the pd.read_csv() function.

Data Preprocessing: The relevant features for recommendation are selected, which include 'genres', 'keywords', 'tagline', 'cast', and 'director'. Null values in these features are replaced with empty strings using a for loop.

Feature Combination: The selected features are combined into a single feature named combined_features by concatenating them together. This step aims to create a comprehensive representation of each movie for similarity calculation.

Feature Vectorization: The combined text data in combined_features is converted into feature vectors using the TfidfVectorizer from scikit-learn. This vectorization step transforms the text data into numerical representations that can be used for similarity calculation.

Similarity Calculation: The cosine similarity metric is used to calculate the similarity scores between each pair of movies based on their feature vectors. The cosine similarity algorithm measures the similarity between two vectors by calculating the cosine of the angle between them. The resulting similarity matrix is stored in the similarity variable.

Movie Recommendation: The user is prompted to enter their favorite movie name. The code then finds the closest match to the input movie name from the list of movie titles in the dataset using the difflib.get_close_matches() function.

Similarity Score Sorting: The similarity scores for the closest match movie are retrieved from the similarity matrix and stored in the similarity_score list. The list is then sorted in descending order based on the similarity scores using the sorted() function.

Printing Recommendations: The top similar movies are printed as recommendations using a for loop. The loop iterates over the sorted similarity scores and retrieves the movie title based on the movie index. It prints the movie titles along with their corresponding numerical index.

Prerequisites

To run this project, you need to have the following installed:

Python (version 3.0 or higher)
NumPy
pandas
scikit-learn (sklearn)
difflib
