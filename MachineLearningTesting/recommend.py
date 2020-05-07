import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

### Reading the files ###
df = pd.read_csv('user_likes.csv', sep=',', names=['user_id','movie_id','swipe'])
movie_titles = pd.read_csv('movie_titles.csv', sep=',', names=['movie_id','movie_title'])

### Merge databases ###
df = pd.merge(df, movie_titles, on='movie_id')


swipes = pd.DataFrame(df.groupby('movie_title')['swipe'].mean())
swipes['number_of_swipes'] = df.groupby('movie_title')['swipe'].count()
movie_matrix = df.pivot_table(index='user_id', columns='movie_title', values='swipe')
swipes.sort_values('number_of_swipes', ascending=False).head(10)
