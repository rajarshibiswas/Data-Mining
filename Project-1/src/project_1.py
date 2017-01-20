import csv
import sys

# Cosine Similarity proximity function.
def Cosine Similarity(f)
    # Algo for Cosine Similarity

    return

# Read the data file.
# fileName - The name of the file.
def read_data_file(fileName, f):
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

def analyze_data()
    f = []
    # fileName = input("Enter he data set path: ")
    fileName = "../DataSet/Iris.csv"
    read_data_file(fileName, f)
    print "1. Cosine Similarity"
    print "2. "
    choice = input("Enter your choice:")

    if choice == 1:
        # call the particular function
    ellif choice == 2:
        #
    else:
        print "Error in choosing"

    return

analyze_data()
