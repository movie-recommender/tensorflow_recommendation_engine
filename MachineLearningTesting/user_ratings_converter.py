# Converts u.data to user_likes.csv, removing 1-5 rating, changing to 0-1.
import csv
with open('ml-100k/u.data') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="\t")
    csv_writer = csv.writer(open("user_likes.csv", 'w'))
    for row in csv_reader:
        if int(row[2]) >= 3:
            row[2] = True
        elif int(row[2]) < 3:
            row[2] = False
        row.remove(row[3])
        csv_writer.writerow(row)
