# Import libraries
import numpy as np
import pandas as pd

# Reading ratings file
ratings = pd.read_csv('converted_data/100k/user_likes.csv', sep=',', names=['userId','movieId','rating'])
print('Printing rating file...')
print(ratings.head())

# Reading movies file
movies = pd.read_csv('converted_data/100k/movie_titles.csv', sep=',', names=['movieId','title'])
print('\nPrinting movie file...')
print(movies.head())

n_users = ratings.userId.unique().shape[0]
n_movies = ratings.movieId.unique().shape[0]

Ratings = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
Ratings.head()

R = Ratings.values
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)


sparsity = round(1.0 - len(ratings) / float(n_users * n_movies), 3)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(Ratings_demeaned, k = 50)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds = pd.DataFrame(all_user_predicted_ratings, columns = Ratings.columns)


def recommend_movies(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.userId == (userID)]
    user_full = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )
    print('\nUser {} has already swiped {} movies.'.format(userID, user_full.shape[0]))
    print('Recommending {} movies not watched yet.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )

    return user_full, recommendations


already_rated, predictions = recommend_movies(preds, 1, movies, ratings, 5) # Number of recommendations

print(predictions.head())
print('\n')
#####################################################################################

# Surprise Python Package
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the dataset for 5-fold evaluation
# data.split(n_folds=5) # FUNCTION NOT WORKING

svd = SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
svd.fit(trainset) # CHANGED .train() TO .fit()

ratings[ratings['userId'] == 1]
print('\n')
print(svd.predict(1, 276)) # UserID = 1, MovieID = 276