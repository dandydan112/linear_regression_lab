import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# Step 1, load data
x_train, y_train = load_data()

# 2 print dataen
print("Type of x_train:", type(x_train))
print("First five elements of x_train are:\n", x_train[:5])
print("Type of y_train:", type(y_train))
print("First five elements of y_train are:\n", y_train[:5])
print('Shape of x_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)
print('Number of training examples (m):', len(x_train))

#3 idk
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.show()

# 4
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        total_cost += cost
    total_cost = total_cost / (2 * m)
    return total_cost


#5
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

#6 gradient descent
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history

#kør GD
initial_w = 0
initial_b = 0
iterations = 1500
alpha = 0.01

w, b, _ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"w,b found by gradient descent: {w:.4f}, {b:.4f}")

# 8 - Plot the fit
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.show()

# 9 - Predict on new data
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1 * 10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2 * 10000))