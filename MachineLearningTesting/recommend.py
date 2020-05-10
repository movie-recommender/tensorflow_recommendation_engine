import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Reading the files 
df = pd.read_csv('user_likes.csv', sep=',', names=['user_id','movie_id','swipe']) # ratings file
movie_titles = pd.read_csv('movie_titles.csv', sep=',', names=['movie_id','movie_title']) # movie file

# Merge databases on rated movies
df = pd.merge(df, movie_titles, on='movie_id')

# Average rating and number of swipes per movie
swipes = pd.DataFrame(df.groupby('movie_title')['swipe'].mean())
swipes['number_of_swipes'] = df.groupby('movie_title')['swipe'].count()

# Constructing the matrix, 2 = Like, 1 = Don't like, 0 = Not rated
movie_matrix = df.pivot_table(index='user_id', columns='movie_title', values='swipe').fillna(0)

# Top 10 rated movies
swipes.sort_values('number_of_swipes', ascending=False).head(10)

user_swipes_mean = np.mean(movie_matrix, axis = 1)
swipes_demeaned = movie_matrix - user_swipes_mean.values.reshape(-1, 1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(swipes_demeaned, k = 50)
sigma = np.diag(sigma)
all_user_predicted_swipes = np.dot(np.dot(U, sigma), Vt) + user_swipes_mean.values.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_swipes, columns = movie_matrix.columns)


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movie_id', right_on = 'movie_id').
                     sort_values(['swipe'], ascending=False)
                 )

    print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print ('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movie_id',
               right_on = 'movie_id').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations

already_rated, predictions = recommend_movies(preds, 1, movie_titles, df, 20)

