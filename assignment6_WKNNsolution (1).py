import numpy as np
import pandas as pd

def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "Lionel Messi"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "012345678"


 
#You can also define your own functions here if necessary 
# calculate the Euclidean distance between two vectors be aware that the last element is for the class
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1[:] - row2[:])**2))


# Locate the most similar neighbors
def get_neighbors(train, test_instance, num_neighbors):
    distances = []
    train_copy=train.copy()
    for j in range(len(train)):
        dist = euclidean_distance(test_instance, train.iloc[j,:])
        distances.append(dist)
    train_copy['distance']=distances
    train_copy['weights']=1/train_copy['distance']
    return train_copy.nsmallest(num_neighbors, ['distance'])
    

# Make a classification prediction with wieghted neighbors
def predict_classes(train, test, num_neighbors):
    prediction=[]
    for j in range(len(test)):
        neighbors = get_neighbors(train, test.iloc[j,:], num_neighbors)
        Class_Weight_Sums=[]
        for i in np.unique(neighbors.variety):
            Class_Weight_Sums.append((i,np.sum(neighbors[neighbors.variety==i].weights)))
            print(Class_Weight_Sums)
        predicted_class = max(Class_Weight_Sums, key=lambda x:x[1])[0]
        prediction.append(predicted_class)
    return np.array(prediction)

    
def compute_accuracy(prediction,test_y):
    return np.mean(prediction==test_y)


def Weighted_KNN(train,test,number_neighbors):
    test_y=test.variety
    predictions=predict_classes(train, test, number_neighbors)
    accuracy=compute_accuracy(predictions,test_y)
    return accuracy,predictions

train = pd.read_csv("HW6Train.csv")
test = pd.read_csv("HW6Test.csv")

accuracy,predictions = Weighted_KNN(train, test, number_neighbors=80)

print(predictions)
print(accuracy)