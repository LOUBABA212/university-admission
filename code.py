import numpy as np
import matplotlib.pyplot as plt
import math


# load dataset


def load_data(filename):
    data=np.loadtxt(filename, delimiter = ',')
    x = data[:,:2]
    y = data[:,2]
    return x,y


x_train, y_train = load_data("data1.txt")

#Display variables
print("First five elements in x_train are:\n", x_train[:5])
print("Type of X_train:",type(x_train))
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))


#Data visulization
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
plot_data(x_train, y_train[:], pos_label="Admit", neg_label="Pas admit")
plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()

#Building logistic regression model
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def compute_cost(x, y, w, b, *argv):
    m, n = x.shape
    loss_sum = 0
    for i in range(m):
        z_w_ij = 0  # Reset z_w_ij for each sample i
        for j in range(n):
            z = w[j] * x[i][j]
            z_w_ij += z
        z_wb = z_w_ij + b
        loss = (-y[i] * math.log(sigmoid(z_wb)) - (1 - y[i]) * math.log(1 - sigmoid(z_wb)))
        loss_sum += loss
    total_cost = (1 / m) * loss_sum
    return total_cost


# Compute and display cost with w and b initialized to zeros
m, n = x_train.shape
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))
# Compute and display cost with non-zero w and b
# Compute and display cost with non-zero w and b 
test_w = np.array([0.2, 0.2]) 
test_b = -24. 
cost = compute_cost(x_train, y_train, test_w, test_b) 
print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


#Gradient Descent for logistic regression
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in) 
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        if i < 100000:
            cost = cost_function(X, y, w_in, b_in)
            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f} ")
    return w_in, b_in, J_history, w_history 

def compute_gradient(x, y, w, b):  
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = predict(x[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i][j]
        dj_db += err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def predict(x, w, b):
    p=np.dot(x,w) +b
    return p


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = predict(x[i], w, b)
        cost += (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return np.squeeze(cost)
   
    
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8
# Some gradient descent settings
iterations =200000
alpha = 0.0001
w,b, J_history,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)    

print("w: ",w, " , b",b)


#plot the border of decision
def plot_data_with_decision_boundary(X, y, w, b, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
    # Plotting the decision boundary
    x_values = [np.min(X[:, 0] - 0.1), np.max(X[:, 0] + 0.1)]
    y_values = [-(w[0] * x_values[0] + b) / w[1], -(w[0] * x_values[1] + b) / w[1]]
    plt.plot(x_values, y_values, label='Decision Boundary')
    
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    plt.legend(loc="upper right")
    plt.show()

# Usage
plot_data_with_decision_boundary(x_train, y_train[:], w, b, pos_label="Admit", neg_label="Pas admit")

def predict(x, w, b): # number of training examples 
    m, n = x.shape 
    p = np.zeros(m)
    for i in range(m):
        z=np.dot(w,x[i]) + b
        f_xi=sigmoid(z)
        if f_xi >= 0.5:
            p[i]=1
    return p 



#Compute accuracy on our training set 
p = predict(x_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))


    
