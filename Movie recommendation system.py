#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib #provides classes and functions for comparing and manipulating sequences, It is often used for tasks such as string matching and finding similarities between texts.
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer is a utility class in scikit-learn that converts a collection of raw documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features
from sklearn.metrics.pairwise import cosine_similarity #Cosine similarity is often used to measure the similarity between two vectors or documents.


# In[4]:


# loading the data from the csv file to a pandas dataframe
movies_data = pd.read_csv('movies.csv')


# In[5]:


# printing the first 5 rows of the dataframe
movies_data.head()


# In[6]:


# number of rows and columns in the data frame

movies_data.shape


# In[7]:


# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[8]:


# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[9]:


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[10]:


print(combined_features)


# In[11]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[12]:


feature_vectors = vectorizer.fit_transform(combined_features) #transform the data into numerical values 


# In[13]:


print(feature_vectors)


# In[14]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[15]:


print(similarity)


# In[16]:


print(similarity.shape)


# In[17]:


# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[18]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[19]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[20]:


close_match = find_close_match[0]
print(close_match)


# In[21]:


# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[22]:


# getting a list of similar movies
#enumerate function is a built-in Python function that returns an iterator consisting of pairs of elements and their corresponding indices
similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[23]:


len(similarity_score)


# In[24]:


# sorting the movies based on their similarity score
#x[1] accesses the second element (the similarity score) of each tuple. By sorting based on this similarity score, the most similar movies will be at the beginning of the sorted list
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[25]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[28]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]
# retrieves the index of the movie from the movies_data where the title matches the close_match value. The .values[0] at the end extracts the first element of the resulting index array and assigns it to the variable index_of_the_movie.
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:





# In[ ]:




