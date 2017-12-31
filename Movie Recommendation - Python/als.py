
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Reading tags.csv file 
tag_headers = ['user_id', 'movie_id', 'tag', 'timestamp']
tags = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\tags.csv',header=None,skiprows = 1, names=tag_headers)
#Reading ratings.csv file
rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\ratings.csv', header=None,skiprows = 1, names=rating_headers)
#Reading movies.csv file
movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\movies.csv', header=None,skiprows = 1, names=movie_headers)
movie_titles = movies.title.tolist()


# In[3]:


#Combined both movies and tags data
df = movies.join(ratings, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
#df is the dataframe obtained 


# In[4]:


del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']


# In[5]:


#Changed structure of dataframe by converting index to user_id; columns to movie_id and their values as rating
rp = df.pivot_table(columns='movie_id',index='user_id',values='rating')


# In[6]:


rp = rp.fillna(0) # Replace NaN


# In[7]:


Q = rp.values #Converting dataframe to matrix form


# In[31]:


Q


# In[8]:


W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)
#W is the weight matrix to consider the users and movies who only rated


# In[9]:


lambda_ = 0.1
n_factors = 100 # number of latent factors to be considered
m, n = Q.shape
n_iterations = 1 # number of iterations


# In[46]:


np.random.rand(2,2)


# In[10]:


_X = 5 * np.random.rand(m, n_factors) #random matrix of m users with n_factors latent factors
Y = 5 * np.random.rand(n_factors, n) #random matrix of n_factors latent factors with n users


# In[11]:


#Error function between given data and modeled data
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


# In[12]:


np.dot(Y, Y.T) + lambda_ * np.eye(n_factors)


# In[13]:


np.dot(Y, Q.T)


# In[14]:


#calculating errors and also obtaining the best convergence point by iterating through n_iterations
#This is without considering the W (weight) factor
errors = []
for ii in range(n_iterations):
    X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(n_factors), 
                        np.dot(Y, Q.T)).T
    Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(n_factors),
                        np.dot(X.T, Q))
    if ii % 100 == 0:
        print('{}th iteration is completed'.format(ii))
    errors.append(get_error(Q, X, Y, W))
Q_hat = np.dot(X, Y)
print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))


# In[15]:


#print function to obtain the recommendend movies with predicted rating
def print_recommendations(W=W, Q=Q, Q_hat=Q_hat, movie_titles=movie_titles):
    #Q_hat -= np.min(Q_hat)
    #Q_hat[Q_hat < 1] *= 5
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')


# In[16]:


#Calculating weighter errors and iterating through n_iterations to get the best convergence point
weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)


# In[17]:


print_recommendations(Q_hat=weighted_Q_hat)


# In[18]:


from sklearn.metrics import accuracy_score


# In[25]:


Q1 = Q.tolist()


# In[26]:


weighted_Q_hat1 = weighted_Q_hat.tolist()


# In[47]:


Q_hat


# In[41]:


weighted_Q_hat


# In[48]:


Q


# In[38]:


sum = 0
for i in range(len(weighted_errors)):
    sum = sum + weighted_errors[i]
sum = sum /n_iterations
sum = sum ** (1/2)
print(sum)

