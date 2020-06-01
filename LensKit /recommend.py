import lenskit.datasets as ds
import pandas as pd
import csv

data = ds.MovieLens('lab4-recommender-systems/')
print("Successfully installed dataset.")
rows_to_show = 10
data.ratings.head(rows_to_show)

minimum_to_include = 20 #<-- You can try changing this minimum to include movies rated by fewer or more people

average_ratings = (data.ratings).groupby(['item']).mean()
rating_counts = (data.ratings).groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')

print("RECOMMENDED FOR ANYBODY:")
print(joined_data.head(rows_to_show))

personA_dict = {}
personB_dict = {}

with open('lab4-recommender-systems/personA.csv', newline='') as csvfile:
	ratings_reader = csv.DictReader(csvfile)
	for row in ratings_reader:
		personA_dict.update({int(row['item']): float(row['ratings'])})

with open('lab4-recommender-systems/personB.csv', newline='') as csvfile:
	ratings_reader = csv.DictReader(csvfile)
	for row in ratings_reader:
		personB_dict.update({int(row['item']): float(row['ratings'])})
     
print("Rating dictionaries assembled!")
print("Sanity check:")
print("\tPerson A's rating for 1197 (The Princess Bride) is " + str(personA_dict[1197]))
print("\tPerson B's rating for 1197 (The Princess Bride) is " + str(personB_dict[1197]))

from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

num_recs = 5  #<---- This is the number of recommendations to generate. You can change this if you want to see more recommendations

user_user = UserUser(15, min_nbrs=3) #These two numbers set the minimum (3) and maximum (15) number of neighbors to consider. These are considered "reasonable defaults," but you can experiment with others too
algo = Recommender.adapt(user_user)
algo.fit(data.ratings)

print("Set up a User-User algorithm!")

personA_recs = algo.recommend(-1, num_recs, ratings=pd.Series(personA_dict))  #Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings, while 10 is how many recommendations it should generate
    
joined_data_personA = personA_recs.join(data.movies['title'], on='item')
joined_data_personA = joined_data_personA[joined_data_personA.columns[2:]]
print("\n\nRECOMMENDED FOR PERSON A:")
print(joined_data_personA)

personB_recs = algo.recommend(-1, num_recs, ratings=pd.Series(personB_dict))  #Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings, while 10 is how many recommendations it should generate
  
joined_data_personB = personB_recs.join(data.movies['title'], on='item')
joined_data_personB = joined_data_personB[joined_data_personB.columns[2:]]
print("\n\nRECOMMENDED FOR PERSON B:")
print(joined_data_personB)