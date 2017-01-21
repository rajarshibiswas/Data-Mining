import pandas as pd

# Cosine Similarity proximity function.
#def Cosine Similarity(f)
    # Algo for Cosine Similarity
    #return

# Read the data file.
# fileName - The name of the file.
def read_data_file(fileName, f):
    try:
        #f = open (fileName, 'rt')
        # read the file
        data = pd.read_csv(fileName)
        #print row by row
        #for row in reader:
        #    print row[2,0:4]
        print data.head()
    except IOError:
        print "Could not open file: ", fileName
    return

# the main function
def analyze_data():
    f = []
    # fileName = input("Enter he data set path: ")
    fileName = r'../DataSet/Iris.csv'
    read_data_file(fileName, f)
#
#    print "1. Cosine Similarity"
#    print "2. "
#    choice = input("Enter your choice:")

#    if choice == 1:
        # call the particular function
#    ellif choice == 2:
        #
#    else:
#        print "Error in choosing"

    return

# call the runner function.
analyze_data()
