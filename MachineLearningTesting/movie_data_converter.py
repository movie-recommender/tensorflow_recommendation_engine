import csv
# Converts movies.csv to movie_titles.csv, only keeping the titles
# Genre is available in movies.csv
with open('movies.csv', encoding = "ISO-8859-1") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="|")
    csv_writer = csv.writer(open("movie_titles.csv", 'w'))
    for row in csv_reader:
    	result = []
    	result.append(row[0]) 
    	result.append(row[1])
    	csv_writer.writerow(result)

