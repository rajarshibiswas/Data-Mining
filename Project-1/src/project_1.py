import csv
import sys


# Read the data file.
# fileName - The name of the file.
def read_data_file (fileName, f):

    try:
        f = open (fileName, 'rt')
        # read file
        reader = csv.reader(f)
        #print row by row
        for row in reader:
            print row[2,0:4]
    except IOError:
        print "Could not open file: ", fileName
    fileName = "hi"
    # Close the file.
    if (f):
        f.close()
    return

f = []
# fileName = input("Enter he data set path: ")
fileName = "../DataSet/Iris.csv"
read_data_file(fileName, f)
