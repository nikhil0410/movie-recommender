
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import json,ast
import plotly
import plotly.offline as py


# In[167]:


df = pd.read_csv('the-movies-dataset/movies_metadata.csv')
df.head().transpose()


# In[168]:


print(df.shape)
df.columns


# The above dataset has 24 Coulmns

# In[169]:


df.info()


# There are a total of 45,466 movies with 24 features. Most of the features have very few NaN values (apart from homepage and tagline). We will attempt at cleaning this dataset to a form suitable for analysis in the next section.

# # Data Wrangling

# In[170]:


df = df.drop(['imdb_id'], axis=1)
df = df.drop(['original_title'],axis=1)


# In[171]:


df.shape


# In[172]:


df['revenue'] = df['revenue'].replace(0,np.nan)


# In[173]:


df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['budget'] = df['budget'].replace(0, np.nan)
df[df['budget'].isnull()].shape


# In[174]:


df['return'] = df['revenue']/df['budget']


# In[175]:


df['year'] = pd.to_datetime(df['release_date'],errors='coerce').apply(lambda x: str(x).split('-')[0] if x!=np.nan else np.nan)


# In[176]:


#df['year']


# In[177]:


df['adult'].value_counts()


# there are close to 0 adult movies in the dataset so the 'adult' column is not relevant. 

# In[178]:


df = df.drop('adult',axis=1)


# In[179]:


df['video'].value_counts()


# In[180]:


base_poster_url = 'http://image.tmdb.org/t/p/w185/'
df['poster_path'] = "<img src='" + base_poster_url + df['poster_path'] + "' style='height:100px;'>"


# # EDA

# In[181]:


df['title'] = df['title'].astype('str')
df['overview'] = df['overview'].astype('str')


# In[182]:


title_corpus = ' '.join(df['title'])
overview_corpus = ' '.join(df['overview'])


# In[128]:


title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[14]:


overview_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(overview_corpus)
plt.figure(figsize=(16,8))
plt.imshow(overview_wordcloud)
plt.axis('off')
plt.show()


# In[37]:


df['production_countries'].isnull().sum()


# In[183]:


df['production_countries'] = df['production_countries'].fillna('[]').apply(ast.literal_eval)
df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])


# In[184]:


s = df.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'countries'


# In[185]:


con_df = df.join(s)
con_df = pd.DataFrame(con_df['countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies', 'country']
con_df = con_df.reset_index()
con_df = con_df.drop('index',axis=1)

con_df.sort_values('num_movies',ascending=False).head(10)


# In[103]:


con_df = con_df[con_df['country'] != 'United States of America']


# In[119]:


data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_movies'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0, 0)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the MovieLens Movies (Apart from US)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
plt.show()


# # Franchise Movies

# In[186]:


df_fran = df[df['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection'] = df_fran['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x,dict) else np.nan)


# In[187]:


df_fran['belongs_to_collection'] = df_fran[df_fran['belongs_to_collection'].notnull()]
df_fran['belongs_to_collection'].isnull().sum()
#df_fran['belongs_to_collection']


# In[188]:


fran_pivot = df_fran.pivot_table(index='belongs_to_collection', values='revenue',aggfunc={'revenue': ['mean','sum','count']})
fran_pivot.sort_values('count', ascending=False)


# The Bowery Boys has the largest collection with the count 29, it also consist of Jamesbond Collection with the count 26

# In[189]:


fran_pivot.sort_values('mean', ascending=False).head(10)


# With just 1 count, Avatar is the highest grossing movie

# In[190]:


fran_pivot.sort_values('sum', ascending=False).head(10)


# The Harry Potter Franchise is the most successful movie franchise raking in more than 7.707 billion dollars from 8 movies. The Star Wars Movies come in a close second with a 7.403 billion dollars from 8 movies too. James Bond is third but the franchise has significantly more movies compared to the others in the list and therefore, a much smaller average gross.

# In[194]:


df['production_companies'] = df['production_companies'].fillna('[]').apply(ast.literal_eval)

df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[195]:


df['production_companies']


# In[196]:


s = df.apply(lambda x: pd.Series(x['production_companies']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'companies'
s


# In[197]:


com_df = df.drop('production_companies', axis=1).join(s)


# In[198]:


com_df.head()


# In[199]:


com_sum = pd.DataFrame(com_df.groupby('companies')['revenue'].sum().sort_values(ascending=False))
com_sum.columns = ['Total']
com_mean = pd.DataFrame(com_df.groupby('companies')['revenue'].mean().sort_values(ascending=False))
com_mean.columns = ['Average']
com_count = pd.DataFrame(com_df.groupby('companies')['revenue'].count().sort_values(ascending=False))
com_count.columns = ['Number']

com_pivot = pd.concat((com_sum, com_mean, com_count), axis=1)


# In[200]:


com_pivot.sort_values('Total', ascending=False).head(10)


# In[203]:


com_pivot[com_pivot['Number'] >= 15].sort_values('Average', ascending=False).head(10)


# In[202]:


com_pivot.sort_values('Number', ascending=False).head(10)


# #### Eda for the original lannguages
# 

# In[204]:


df['original_language'].drop_duplicates().count()


# In[220]:


lang = pd.DataFrame(df['original_language'].value_counts())
lang['lang'] = lang.index
lang.columns = ['num', 'language']
lang.head(10)


# In[222]:


plt.figure(figsize=(12,5))
sns.barplot(x='language',y='num',data = lang.iloc[1:11])
plt.show()


# from the above we can see that most movies are on english language, also hindi is at 8th position

# In[207]:


df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(ast.literal_eval)
df['spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])


# In[208]:


print(df['spoken_languages'])

