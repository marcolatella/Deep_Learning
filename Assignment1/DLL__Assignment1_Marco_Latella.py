import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

w_s = torch.tensor([-8, -4, 2, 1])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 1 Complete the code above.
def create_dataset(w_star, x_range, sample_size, sigma, seed=None, fourth_grade=False):
    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0], x_range[1], sample_size)
    ncol = w_star.shape[0]
    X = np.zeros((sample_size, ncol))
    for i in range(sample_size):
        X[i, 0] = 1.
        for j in range(1, ncol):
            # the statement below was completed adding the i-th element of x to the power of j which
            # iterate from 1 to 3
            X[i, j] = (x[i] ** j)
    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    if fourth_grade:
        X = np.c_[X, X[:, 1] ** 4]
    return X, y


def add_to_gpu(x_tr, y_tr, x_v, y_v):
    x_tr = torch.from_numpy(x_tr).float().to(DEVICE)
    y_tr = torch.from_numpy(y_tr.reshape((y_tr.shape[0], 1))).float().to(DEVICE)
    x_v = torch.from_numpy(x_v).float().to(DEVICE)
    y_v = torch.from_numpy(y_v.reshape((y_v.shape[0], 1))).float().to(DEVICE)
    return x_tr, y_tr, x_v, y_v


# Train and Evaluate the Neural Network
def run_nn(model, x_tr, y_tr, x_v, y_v, steps, learning_rate):
    losses_tr = []
    losses_v = []

    loss_f = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        y_ = model(x_tr)
        loss = loss_f(y_, y_tr)
        print(f"Step {step}: train loss: {loss}")
        losses_tr.append(loss)

        loss.backward()
        optimizer.step()

        # Eval on validation set
        model.eval()
        with torch.no_grad():
            y_ = model(x_v)
            val_loss = loss_f(y_, y_v)
            print(f"Step {step}: val loss: {val_loss}")
            losses_v.append(val_loss)

    return losses_tr, losses_v


# Plots the losses in a function of the steps
def plot_losses(losses_tr, losses_v, steps):
    x = [step for step in range(steps)]
    yt = [loss.detach().numpy() for loss in losses_tr]
    yv = [loss.detach().numpy() for loss in losses_v]
    plt.figure(figsize=(12, 6))
    plt.plot(x, yt, label='train')
    plt.plot(x, yv, label='val')
    plt.legend(loc='upper right')
    plt.show()


def task_2_3():
    # 2.1 - Use the completed code and the following parameters to generate training and validation data points:
    # Use a sample of size 100 created with a seed of 0 for training.
    x_tr, y_tr = create_dataset(w_s, (-3, 2), 100, 0.5, 0)

    # 2.2 - Use sample of size 100 created with a seed of 1 for validation.
    x_v, y_v = create_dataset(w_s, (-3, 2), 100, 0.5, 1)

    # 3 - Create a 2D scatter plot (using x and y) of the generated
    # training and validation dataset.
    xtr = x_tr[:, 1]
    xv = x_v[:, 1]
    plt.scatter(xtr, list(y_tr), label='train')
    plt.scatter(xv, list(y_v), label='val')
    plt.title("2D Scatter Plot")
    plt.legend(loc='lower left')
    plt.show()


def task_5_7():
    # 5 - find and report an estimate of w∗ = [−8, −4, 2, 1]T

    x_tr, y_tr = create_dataset(w_s, (-3, 2), 100, 0.5, 0)
    x_v, y_v = create_dataset(w_s, (-3, 2), 100, 0.5, 1)

    x_tr, y_tr, x_v, y_v = add_to_gpu(x_tr, y_tr, x_v, y_v)

    model = torch.nn.Linear(4, 1, False)
    steps = 1000
    learning_rate = 0.01
    losses_tr, losses_v = run_nn(model, x_tr, y_tr, x_v, y_v, steps, learning_rate)
    print(f"Estimate of w*: {model.weight.detach().numpy()[0]}")

    # 7 - Plot the training and validation losses as a function of the gradient descent iterations.
    plot_losses(losses_tr, losses_v, steps)

    return model.weight.detach().numpy()


def task_8(ws_pred):
    # 8 - Plot the polynomial defined by w∗ and the polynomial defined by your estimate wˆ
    w = [-8, -4, 2, 1]
    pred_w = [item for item in ws_pred[0]]

    x = np.linspace(-3, 2, 1000)
    y = w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3
    y_pred = pred_w[0] + pred_w[1] * x + pred_w[2] * (x ** 2) + pred_w[3] * (x ** 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, y_pred, 'b', label='pred')
    plt.plot(x, y, 'r', label='func')
    plt.legend(loc='lower left')
    plt.show()


def task_9(rows, sd=0.5, plot_graph=True):
    # 9 - Report and explain what happens when the training dataset is reduced to 50, 10, and 5 observations

    x_tr, y_tr = create_dataset(w_s, (-3, 2), rows, sd, 0)
    x_v, y_v = create_dataset(w_s, (-3, 2), 100, sd, 1)

    x = x_tr[:, 1]
    y = list(y_tr)
    xv = x_v[:, 1]
    yv = list(y_v)

    x_tr, y_tr, x_v, y_v = add_to_gpu(x_tr, y_tr, x_v, y_v)

    model = torch.nn.Linear(4, 1, False)
    steps = 1000
    learning_rate = 0.01
    losses_tr, losses_v = run_nn(model, x_tr, y_tr, x_v, y_v, steps, learning_rate)

    w = model.weight.detach().numpy()
    pred_w = [item for item in w[0]]
    i = np.linspace(-3, 2)
    y_pred = pred_w[0] + pred_w[1] * i + pred_w[2] * (i ** 2) + pred_w[3] * (i ** 3)

    # Condition used by task 11 to not show not needed plots
    if plot_graph:
        plot_losses(losses_tr, losses_v, steps)
        plt.scatter(x, y, label='train data')
        plt.scatter(xv, yv, label='val data')
        plt.plot(i, y_pred, 'b-')
        plt.legend()
        plt.show()

    return y_pred


def task_10(sd):
    # 10 - Report and explain what happens when σ is increased to 2, 4, and 8.
    # Illustrate your observations with some plots of your choice.
    task_9(100, sd)


def task_11():
    # 11 - Bonus Task

    # Creates modified datasets to meet the requirements for the polynomial of degree four
    x_tr, y_tr = create_dataset(w_s, (-3, 2), 10, 0.5, 0, True)
    x_v, y_v = create_dataset(w_s, (-3, 2), 100, 0.5, 1, True)

    x = x_tr[:, 1]
    print(x)
    y = list(y_tr)

    xv = x_v[:, 1]
    yv = list(y_v)

    x_tr, y_tr, x_v, y_v = add_to_gpu(x_tr, y_tr, x_v, y_v)

    y_pred = task_9(10, 0.5, False)

    model1 = torch.nn.Linear(5, 1, False)
    steps = 1000
    learning_rate = 0.01
    losses_tr1, losses_v1 = run_nn(model1, x_tr, y_tr, x_v, y_v, steps, learning_rate)
    plot_losses(losses_tr1, losses_v1, steps)

    w = model1.weight.detach().numpy()
    pred_w = [item for item in w[0]]

    i = np.linspace(-3, 2)
    y_pred1 = pred_w[0] + pred_w[1] * i + pred_w[2] * (i ** 2) + pred_w[3] * (i ** 3) * pred_w[4] * (i ** 4)

    plt.scatter(xv, yv, color='#f1c232', label='val data')
    plt.plot(i, y_pred, 'k-', label='3° poly.')
    plt.plot(i, y_pred1, 'm-', label='4° poly.')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Execution of Task 2 and 3
    task_2_3()

    # Execution of Task 5,7 and 8
    w_pred = task_5_7()
    task_8(w_pred)

    # Execution of Task 9
    task_9(50)
    task_9(10)
    task_9(5)

    # Execution of Task 10
    task_10(2)
    task_10(4)
    task_10(8)

    # Execution of Task 11
    task_11()
