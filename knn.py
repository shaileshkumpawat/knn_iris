import csv
import random

#spliting trainingset and testset , using spliting ratio as 67%
def loaddataset(trainingset , testset):
    with open('IRIS.csv' , 'r') as csv_file:    #parsing a csv file for dataset
        lines = csv.reader(csv_file)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < 0.67:  #trainingset:testset = 0.67
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])

trainingset = list()
testset = list()
loaddataset(trainingset , testset)
#print(len(trainingset))
#print(len(testset))

#Euclidean norm defined below
import math
def norm(p , q , dim):  #dim is for dimension
    d = 0
    for x in range(dim):
        d += pow((p[x]-q[x]) , 2)
    return math.sqrt(d)

#getting a sorted-list of k-nearest neighbors
import operator
def knn(trainingset , test , k):
    distances = list()
    length = len(test)-1
    for x in range(len(trainingset)):
        d = norm(trainingset[x] , test , length)
        distances.append((trainingset[x] , d))
    distances.sort(key=operator.itemgetter(1))
    #print(distances)
    knn = list()
    for x in range(k):
        knn.append(distances[x][0])
    return knn

#collecting the maximum votes for a class
import operator
def get_class(knn):
    classvotes = {} #created a dictionary to keep counts of neighbors
    for x in range(len(knn)):
        cl = knn[x][-1]
        if cl in classvotes:
            classvotes[cl] += 1
        else:
            classvotes[cl] = 1
        sortedvotes = sorted(classvotes.items(), key=operator.itemgetter(1) , reverse = True)
    return sortedvotes[0][0]

#putting the entire implementation in a main() function 
def main():
    trainingset = list()
    testset = list()
    loaddataset(trainingset , testset)
    print("train set" , len(trainingset))
    print("test set" , len(testset))
    #predictions = list() 
    k=5
    for x in range(len(testset)):
        neighbors = knn(trainingset,testset[x],k)
        result = get_class(neighbors)
        #predictions.append(result)
        print('predicted: ', result , 'actual: ' , testset[x][-1])

main()
