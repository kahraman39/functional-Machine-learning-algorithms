import numpy as np
import pandas as pd

#You can also define your own functions here if necessary 
def p_y(y):
    p_y=[]
    for i in np.sort(y.unique()):
        pyi=len(y[y==i])/len(y)
        p_y.append(pyi)
    return p_y

def compute_means_for_each_class(x,y):
    means=[]
    for i in np.sort(y.unique()):
        means_i=x[y==i].mean()
        means.append(means_i)
    return means
    
def compute_std_correlation_for_each_column(x):
    return (x.std(),x.corr().iloc[0,1])

# Calculate Gaussian Probability Density Function
def calculateGaussianProbability(x,k,means,sds,corr):
    expo = np.exp(-((sds[1]**2)*(x.iloc[:,0]-means[k][0])**2
                    +(sds[0]**2)*(x.iloc[:,1]-means[k][1])**2
                    -2*corr*sds[0]*sds[1]*(x.iloc[:,0]-means[k][0])*(x.iloc[:,1]-means[k][1])) 
                  / (2*(1-corr**2)*(sds[0]**2)*(sds[1])**2))
    return (1 / (2*np.pi*sds[0]*sds[1]*np.sqrt(1-corr**2))) * expo
    
def compute_accuracy(prediction,test_y):
    return (1-np.sum(np.abs(prediction-test_y))/len(prediction))*100


def not_so_naive_bayes(train,test):
    y=train['y']
    x=train.drop('y',axis=1)

    class_probs=p_y(y)
    means=compute_means_for_each_class(x,y)
    stdevs,correlation=compute_std_correlation_for_each_column(x)
    
    probabilities=[]
    
    x=test.drop('y',axis=1)
    for i in range(len(class_probs)):
        fi=calculateGaussianProbability(x=x,k=i,means=means,sds=stdevs,corr=correlation)
        numerator=fi*class_probs[i]
        probabilities.append(numerator)
    probabilities=probabilities/np.sum(probabilities,axis=0)
    prediction=(probabilities[1]>=0.5).astype('int')
    accuracy=compute_accuracy(prediction,test.y)
    return accuracy,prediction

train = pd.read_csv("HW5TrainData.csv")
test=pd.read_csv("HW5TestData.csv")

accuracy,prediction = not_so_naive_bayes(train,test)

print(accuracy,prediction)

output: 92.15 [1 1 1 ... 0 0 0]
