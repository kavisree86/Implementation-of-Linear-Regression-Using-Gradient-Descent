# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe
   
  
2. Write a function computeCost to generate the cost function 
3. perform iterations og gradient steps with learning rate 
4. Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: kavisree s
RegisterNumber:212222047001
*/
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd   
data=pd.read_csv("/content/ex1.txt",header = None)   

plt.scatter(data[0],data[1])   
plt.xticks(np.arange(5,30,step=5))   
plt.yticks(np.arange(-5,30,step=5))   
plt.xlabel("Population of City(10,000s)")  
plt.ylabel("Profit ($10,000)")  
plt.title("Profit Prediction")  

def computeCost(X,y,theta):   
  """
  Take in a numpy array X,y,theta and generate the cost function of using the in a linear regression model
  """
  m=len(y) # length of the training data   
  h=X.dot(theta) #hypothesis   
  square_err=(h-y)**2   

  return 1/(2*m) * np.sum(square_err) #returning J   

data_n=data.values  
m=data_n[:,0].size   
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)  
y=data_n[:,1].reshape(m,1)  
theta=np.zeros((2,1))   
computeCost(X,y,theta) #Call the function  

from matplotlib.container import ErrorbarContainer  
from IPython.core.interactiveshell import error   
def gradientDescent(X,y,theta,alpha,num_iters):

    """
    Take the numpy array X,y,theta and update theta by taking the num_tiers gradient with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)  
    J_history=[]  

    for i in range(num_iters):  
      predictions=X.dot(theta)  
      error=np.dot(X.transpose(),(predictions -y))  
      descent=alpha *1/m*error  
      theta-=descent   
      J_history.append(computeCost(X,y,theta))  

    return theta,J_history   

theta,J_history = gradientDescent(X,y,theta,0.01,1500)   
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")   

#Testing the implementation   
plt.plot(J_history)   
plt.xlabel("Iteration")  
plt.ylabel("$J(\Theta)$")  
plt.title("Cost function using Gradient Descent")   


plt.scatter(data[0],data[1])  
x_value=[x for x in range(25)]   
y_value=[y*theta[1]+theta[0] for y in x_value]  
plt.plot(x_value,y_value,color="r")  
plt.xticks(np.arange(5,30,step=5))  
plt.yticks(np.arange(-5,30,step=5))  
plt.xlabel("Population of City (10,000s)")  
plt.ylabel("Profit($10,000")   
plt.title("Profit Prediction")   

def predict(x,theta):

  """
  Tkes in numpy array of x and theta and return the predicted value of y base
  """

  predictions=np.dot(theta.transpose(),x)  

  return predictions[0]  

predict1=predict(np.array([1,3.5]),theta)*10000 
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))  

predict2=predict(np.array([1,7]),theta)*10000   
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0))) 
```

## Output:
![linear regression using gradient descent](sam.png)

![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/63d65191-28e0-4b84-a1ec-f601c2aef6e4)

![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/7bea3767-d3e1-49a7-a0b9-299a32cda803)

## compute cost value:
![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/0be13603-de6a-4ab8-8db5-36f0f82ec843)

## Cost function using Gradient Descent Graph :

![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/6b1b33e9-c105-4277-9b2c-9bf674cee81d)


## Profit for the Population 35,000 :

![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/0d5b1918-6824-4e4a-a00d-958ba3efa875)


## Profit for the Population 70,000 :

![image](https://github.com/kavisree86/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145759687/391f6342-d079-4f70-8905-459de4007e0e)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
