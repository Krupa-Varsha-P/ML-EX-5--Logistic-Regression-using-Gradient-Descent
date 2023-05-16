# Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset
2. Assign x and y values
3. Calculate logistic sigmoid function and plot the graph
4. Calculate the cost function
5. Calculate x_train and y_train grad value
6. Calculate and plot decision boundary
7. Calculate probability value and predict mean value

## Program:
```txt
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Krupa Varsha P
RegisterNumber:  212220220022
```
```python3
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize
```
```python3
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
```
```python3
x=data[:,[0,1]]
```
```python3
y=data[:,2]
```
```python3
plt.figure() 
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted") 
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted") 
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score") 
plt.legend() 
plt.show()
```
```python3
def sigmoid(z): 
  return 1/(1+np.exp(-z)) 
plt.plot()
```
```python3
x_plot=np.linspace(-10,10,100) 
plt.plot(x_plot,sigmoid(x_plot)) 
plt.show()
```
```python3
def costfunction(theta,x,y): 
  h=sigmoid(np.dot(x,theta)) 
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0] 
  grad=np.dot(x.T,h-y)/x.shape[0] 
  return j,grad
```
```python3
x_train=np.hstack((np.ones((x.shape[0],1)),x)) 
theta=np.array([0,0,0]) 
j,grad=costfunction(theta,x_train,y) 
print(j) 
print(grad)
```
```python3
x_train=np.hstack((np.ones((x.shape[0],1)),x)) 
theta=np.array([-24,0.2,0.2]) 
j,grad=costfunction(theta,x_train,y) 
print(j) 
print(grad)
```
```python3
def cost(theta,x,y): 
  h=sigmoid(np.dot(x,theta)) 
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0] 
  return j 
def gradient(theta,x,y): 
  h=sigmoid(np.dot(x,theta)) 
  grad=np.dot(x.T,h-y)/x.shape[0] 
  return grad
```
```python3
x_train=np.hstack((np.ones((x.shape[0],1)),x)) 
theta=np.array([0,0,0]) 
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient) 
print(res.fun) 
print(res.x)
```
```python3
def plotdecisionboundary(theta,x,y): 
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1 
  y_min,y_max=x[:,1].min()-1,x[:,0].max()+1 
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1)) 
  x_plot=np.c_[xx.ravel(),yy.ravel()] 
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot)) 
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)
```
```python3
plt.figure() 
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted") 
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted") 
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score") 
plt.legend() 
plt.show()
plotdecisionboundary(res.x,x,y)
```
```python3 
prob=sigmoid(np.dot(np.array([1,45,85]),res.x)) 
print(prob)
```
```python3
def predict(theta,x): 
  x_train=np.hstack((np.ones((x.shape[0],1)),x)) 
  prob=sigmoid(np.dot(x_train,theta)) 
  return (prob>=0.5).astype(int) 
np.mean(predict(res.x,x)==y)
```


## Output:
![d4e28101-8f6a-458d-a9a0-b4e03702383c](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/b8ca5bcc-131f-4b8c-84d7-7d634f6a6ee8)

![c1622fbe-13ee-4072-a5ab-9c5fc641bd26](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/052ab7b5-2663-44e8-9c0a-c99273288268)

![8c2359d3-7d40-4d43-8698-14e4cb35ceba](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/32c46773-0761-403e-bc8b-b2e3dd7443d7)

![fc4e3024-05d0-414a-8a60-8ca496f62d49](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/d4d693a9-3289-40ba-b8fd-6bee6321d065)

![9c9ea059-0f9b-4a80-b480-9dd6fb9a85cb](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/ae1539e4-953d-4324-9465-b82e6c59f945)

![8dfd001c-e152-49ea-8495-f5eef36db6df](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/0c68eb87-3e7c-4d6e-82f6-9bebb35ddace)

![8e2591fe-791b-4b96-846a-53b6e8a916a7](https://github.com/Krupa-Varsha-P/ML-EX-5--Logistic-Regression-using-Gradient-Descent/assets/100466625/36a16b45-7259-4a03-8ad9-39d60d2e4fc1)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
