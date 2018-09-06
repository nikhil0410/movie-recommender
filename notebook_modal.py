
# coding: utf-8

# In[139]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVD, evaluate


# In[140]:


md = pd.read_csv('the-movies-dataset/movies_metadata.csv')
md.head().transpose()


# In[141]:


md['genres'] = md['genres'].fillna('[]').apply(ast.literal_eval)
md['genres'] = md['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
md['genres'].head()


# In[142]:


md['vote_count'].quantile(0.95)


# In[143]:


vote_count = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_average = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_average.mean()
C


# In[144]:


m = vote_count.quantile(0.95)


# In[145]:


md['year'] = pd.to_datetime(md['release_date'],errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


# In[146]:


qualified = md[(md['vote_count'] > m) & (md['vote_count'].notnull()) &( md['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity','genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape


# In[147]:


def weighted_rating(x):
    v = x['vote_count']
    r = x['vote_average']
    wr = ((v/(v+m))*r) + ((m/(m+v))*C)
    return wr


# In[148]:


qualified['wr'] = qualified.apply(weighted_rating,axis=1)


# In[149]:


qualified = qualified.sort_values('wr',ascending=False).head(250)
qualified.head(25)


# In[150]:


s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1,drop=True)
s.name = 'genre'
gen_md = md.drop('genres',axis=1).join(s)


# In[151]:


gen_md


# In[152]:


def build_chart(genre,percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    m = vote_counts.quantile(percentile)
    C = vote_averages.mean()
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified


# In[153]:


build_chart('Romance')


# ## Content Based Recommender

# In[154]:


md['tagline'] = md['tagline'].fillna('')
md['description'] = md['tagline'] + md['overview']
md['description'] = md['description'].fillna('')
#md.head()


# In[155]:


# tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=10, stop_words='english')
# tfidf_matrix = tf.fit_transform(md['description'])
# tfidf_matrix.shape


# In[156]:


#cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
                                                                                        


# In[157]:


links_small = pd.read_csv('the-movies-dataset/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd.shape


# In[158]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=10,stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape


# In[159]:


cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)


# In[160]:


smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[161]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[162]:


get_recommendations('Aladdin')


# # Recommendation as per the Cast,Director,Genres and Keywords

# In[163]:


credits = pd.read_csv('the-movies-dataset/credits.csv')
keywords = pd.read_csv('the-movies-dataset/keywords.csv')


# In[164]:


credits['id'] = credits['id'].astype('int')
keywords['id'] = keywords['id'].astype('int')
md['id'] = md['id'].astype('int')


# In[165]:


md = md.merge(credits,on='id')
md = md.merge(keywords,on='id')
smd = md[md['id'].isin(links_small)]
smd.head()


# In[166]:


smd.head().transpose()


# In[167]:


smd['cast'] = smd['cast'].apply(ast.literal_eval)
smd['crew'] = smd['crew'].apply(ast.literal_eval)
smd['keywords'] = smd['keywords'].apply(ast.literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))


# In[168]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']


# In[169]:


smd['director'] = smd['crew'].apply(get_director)


# In[170]:


smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x)>=3 else x)


# In[171]:


# smd = smd.drop('cast',axis=1)
# smd['cast']


# In[172]:


smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(' ','')))
smd['director'] = smd['director'].apply(lambda x: [x,x,x])
#smd['director']


# In[173]:


smd['cast'] = smd['cast'].astype('str').apply(lambda x: [str.lower(i.replace(' ','') ) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])


# In[174]:


stemmer = SnowballStemmer('english')
smd['keywords2'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords2'] = smd['keywords2'].apply(lambda x: [str.lower(i.replace(' ','')) for i in x])


# In[175]:


smd['soup'] = smd['keywords2'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[176]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])


# In[177]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[178]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[179]:


get_recommendations('Mean Girls').head(10)


# Popularity and Ratings
# One thing that we notice about our recommendation system is that it recommends movies regardless of ratings and popularity. It is true that Batman and Robin has a lot of similar characters as compared to The Dark Knight but it was a terrible movie that shouldn't be recommended to anyone.
# 
# Therefore, we will add a mechanism to remove bad movies and return movies which are popular and have had a good critical response.
# 
# I will take the top 25 movies based on similarity scores and calculate the vote of the 60th percentile movie. Then, using this as the value of  m , we will calculate the weighted rating of each movie using IMDB's formula like we did in the Simple Recommender section.

# In[180]:


def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[181]:


improved_recommendations('Shutter Island')


# In[182]:


def more_improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:70]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified


# In[183]:


more_improved_recommendations('Shutter Island')


# ## Collaborative Filtering
# 
# Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.
# 
# Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who s/he is.
# 
# Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers. Collaborative Filtering is based on the idea that users similar to a me can be used to predict how much I will like a particular product or service those users have used/experienced but I have not.
# 
# I will not be implementing Collaborative Filtering from scratch. Instead, I will use the Surprise library that used extremely powerful algorithms like Singular Value Decomposition (SVD) to minimise RMSE (Root Mean Square Error) and give great recommendations.

# In[184]:


reader = Reader()


# In[185]:


ratings = pd.read_csv('the-movies-dataset/ratings_small.csv')
ratings.head()


# In[186]:


data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
data.split(n_folds=5)


# In[187]:


svd = SVD()
evaluate(svd,data,measures=['RMSE','MAE'])


# In[188]:


trainset = data.build_full_trainset()
svd.fit(trainset)


# In[189]:


svd.predict(1, 302)


# # Hybrid Recommender
# Merging Collebrative filtering with Content Based Recommender

# In[190]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[196]:


id_map = pd.read_csv('the-movies-dataset/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
id_map.head()


# In[197]:


indeces_map = id_map.set_index('id')


# In[198]:


def hybrid_algo(user_id,title):
    idx = indices[title]
    tmbdId = id_map.loc[title]['id']
    movieId = id_map.loc[title]['movieId']
    
    sim_score = list(enumerate(cosine_sim[int(idx)]))
    sim_score = sorted(sim_score,key=lambda x:x[1],reverse=True)
    sim_score = sim_score[1:26]
    
    movie_indeces = [i[0] for i in sim_score]
    movies = smd.iloc[movie_indeces][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(user_id,indeces_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est',ascending=False)
    
    return movies.head(10)
    


# In[202]:


print(hybrid_algo(21,'Aladdin'))


# In[203]:


print(hybrid_algo(213,'Aladdin'))

