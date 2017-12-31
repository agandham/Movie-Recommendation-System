
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('rec').getOrCreate()


# In[29]:


movie_headers = ['movie_id', 'title', 'genres']
movies = spark.read.csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\movies.csv',header = True)
movies = movies.toPandas()
movie_titles = movies.title.tolist()


# In[4]:


data = spark.read.csv(r'C:\Users\Abhilash Gandham\Desktop\ml-latest-small\ratings.csv',inferSchema = True,header = True)
training,test = data.randomSplit([0.8,0.2])


# In[ ]:


import pandas


# In[5]:


df = training.toPandas()


# In[7]:


rp = df.pivot_table(columns='movieId',index='userId',values='rating')


# In[9]:


rp = rp.fillna(0) # Replace NaN
rp.head()


# In[10]:


Q = rp.values


# In[12]:


import numpy as np


# In[13]:


W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)


# In[22]:


W


# In[14]:


lambda_ = 0.1
n_factors = 100
m, n = Q.shape
n_iterations = 1


# In[23]:


X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)


# In[19]:


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)


# In[20]:


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


# In[24]:


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


# In[30]:


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


# In[31]:


print_recommendations(Q_hat=weighted_Q_hat)

