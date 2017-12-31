
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd


# In[40]:


tagheader = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\tags.csv',header=None,skiprows = 1, names=tagheader)
#Reading tags.csv file
ratingheader = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\ratings.csv', header=None,skiprows = 1, names=ratingheader)
#Reading ratings.csv file
movieheader = ['movie_id', 'title', 'genres']
movies = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\movies.csv', header=None,skiprows = 1, names=movieheader)
#Reading movies.csv file
movie_titles = movies.title.tolist()


# In[41]:


#Combined both movies and tags data
dataframe = movies.join(ratings, on=['movie_id'], rsuffix='_a').join(tags, on=['movie_id'], rsuffix='_b')


# In[42]:


#deleting unwanted columns
del dataframe['movie_id_a']
del dataframe['user_id_b']
del dataframe['movie_id_b']
del dataframe['timestamp_b']


# In[43]:


#Changed structure of dataframe by converting index to user_id; columns to movie_id and their values as rating
pivottable = dataframe.pivot_table(columns='movie_id',index='user_id',values='rating')


# In[44]:


pivottable = pivottable.fillna(0) # Replace NaN with 0
pivottable.head()


# In[45]:


#Generating transpose of the pivottable so that index is movie_id and column is user_id
pivottablet = pivottable.T
#Storing it in matrix form
finalmatrix = pivottablet.values


# In[46]:


finalmatrix.shape


# In[47]:


#normalizing the matrix by subtracting it with the mean
normalised_mat = finalmatrix - np.asarray([(np.mean(finalmatrix, 1))]).T


# In[48]:


#normalized_data is the final data used to find similarity
normalized_data = normalised_mat.T / np.sqrt(8256 - 1)
#Applied SVD to normalied_data
U, S, V = np.linalg.svd(normalized_data)


# In[14]:


#Calculated covariance matrix and alo eigen values and eigen vectors to obtain similar movies using PCA
cov_mat = np.cov(normalised_mat)
evals, evecs = np.linalg.eig(cov_mat)


# In[53]:


# Cosine similarity function to get the top_n similar movies and also the similarity matrix to obtain similarities of 
#specified movie with other movies
def cosinesimilarity(data, movie_id, top_n=10):
    index = movie_id - 1 
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n],similarity

#Print function to print the movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])


# In[54]:


k = 50
movie_id = 3 # Grab an id from movies.csv
top_n = 10
user_id = 1
sliced = V1.T[:, :k]


# In[57]:


#using SVD
k = 50 # number of latent factors
movie_id = 4 # Grab an id from movies.csv
top_n = 10 #number of movies that are very similar
sliced = V1.T[:, :k]
indexes,similarity = cosinesimilarity(sliced, movie_id, top_n)
print_similar_movies(movies, movie_id, indexes)
print(similarity)


# In[31]:


#using PCA
k = 50
movie_id = 3 # Grab an id from movies.csv
top_n = 10
sliced = evecs[:, :k]
top_indexes,similarity = cosinesimilarity(sliced, movie_id, top_n)
similarmovies(movies, movie_id, top_indexes)

