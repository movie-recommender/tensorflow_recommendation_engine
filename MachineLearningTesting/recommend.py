import numpy as np
import pandas as pd

def train(userLikes, movieTitles):
    # Reading files
    ratings = pd.read_csv(userLikes, sep=',', names=['userId','movieId','rating'])
    print('Printing rating file...')
    print(ratings.head())

    movies = pd.read_csv(movieTitles, sep=',', names=['movieId','title'])
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

    return preds, movies, ratings

def recommend(predictions, userID, movies, original_ratings, num_recommendations):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1
    sorted_user_predictions = predictions.iloc[user_row_number].sort_values(ascending=False) # User ID starts at 1
    
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

    return recommendations

predictionMatrix, movieTable, ratingTable = train('converted_data/100k/user_likes.csv','converted_data/100k/movie_titles.csv')
predictions = recommend(predictionMatrix, 196, movieTable, ratingTable, 5) # UserID = 1, No. of recommendations = 5

print(predictions.head())
predictions.to_csv('predictions.csv', index = False)

# # Predicting recommendation accuracy

# # Surprise Python Package
# from surprise import Reader, Dataset, SVD, accuracy
# from surprise.model_selection import cross_validate

# def predict(ratingTable, UserID, MovieID):
#     reader = Reader()
#     data = Dataset.load_from_df(ratingTable[['userId', 'movieId', 'rating']], reader)

#     # Split the dataset for 5-fold evaluation
#     # data.split(n_folds=5) # FUNCTION NOT WORKING

#     svd = SVD()

#     cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#     trainset = data.build_full_trainset()
#     svd.fit(trainset) # CHANGED .train() TO .fit()

#     ratingTable[ratingTable['userId'] == UserID]
#     print('\n')
#     print(svd.predict(UserID, MovieID))

# predict(ratingTable, 1, 276)
