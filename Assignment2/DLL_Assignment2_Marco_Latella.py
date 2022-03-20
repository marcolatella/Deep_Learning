import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformation sequence applied to Training and Test Set
trans_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])


# Class of Convolutional Neural Network
class ConvNet(nn.Module):
    def __init__(self, dropout_value):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_value)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_value)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.Dropout(dropout_value),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(-1, 64 * 5 * 5)
        out = self.fc3(out)
        return out


# Print some images taken by the Training set
def task_dataset_1():
    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=transforms.ToTensor())

    print(f"Train set length: {len(train_set)} images")
    print(f"Test set length: {len(test_set)} images")

    image = train_set[565]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    image = train_set[200]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    image = train_set[2000]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    image = train_set[535]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()


# Normalize Training and Test set
def task_dataset_2():
    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=transforms.ToTensor())

    image = train_set[0]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()

    mean = train_set.data.mean(axis=(0, 1, 2)) / 255
    std = train_set.data.std(axis=(0, 1, 2)) / 255
    print(f"Train_set > mean: {mean}, std: {std}")
    # mean: [0.49139968 0.48215841 0.44653091], std: [0.24703223 0.24348513 0.26158784]

    trans_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=trans_train, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=trans_train)

    image = train_set[0]
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()


def task_dataset_3(batch_size):
    # use the trans transformations defined in the global scope
    train_set = torchvision.datasets.CIFAR10(root='./ data', train=True, transform=trans_train, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./ data', train=False, transform=trans_train)

    indexes = np.arange(len(train_set))

    val_indexes = indexes[49000:]
    train_indexes = indexes[0:49000]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indexes)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indexes)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=2
                                               )
    val_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             num_workers=2
                                             )

    print(f"Length train set: {len(train_sampler)}")
    print(f"Length val set: {len(val_sampler)}")

    return train_loader, val_loader


# Implement the training pipeline, with hyper-parameters required
def train_model(train_loader, val_loader, ep=20, momentum=0.9, lr=0.001, bs=32, dropout=0):
    print(
        f"\n\nModel Hyperparameters: \nEpochs number => {ep}, \nMomentum => {momentum}, \nLearning rate => {lr}, \nDroput probability => {dropout} \n\n")

    num_epochs = ep
    momentum = momentum
    learning_rate = lr

    model = ConvNet(dropout)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    total_steps_train = 0

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        run_step = 0
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss_tr = loss_fn(outputs, labels)
            optimizer.zero_grad()  # reset gradients.
            loss_tr.backward()  # compute gradients.
            optimizer.step()  # update parameters.

            running_loss += loss_tr.item()
            running_total += labels.size(0)

            with torch.no_grad():
                _, predicted = outputs.max(1)
            running_correct += (predicted == labels).sum().item()
            run_step += 1
            total_steps_train += 1
            if i % 150 == 0:
                tmp_loss = running_loss / run_step
                tmp_acc = 100 * running_correct / running_total
                print(f'epoch: {epoch}, steps: {i}, '
                      f'train_loss: {tmp_loss:.3f}, '
                      f'running_acc: {tmp_acc:.1f} %')

                train_loss.append([tmp_loss, total_steps_train])
                train_accuracy.append(tmp_acc)

                running_loss = 0.0
                running_total = 0
                running_correct = 0
                run_step = 0
            if len(images) < bs:
                train_loss.append([tmp_loss, total_steps_train])
                train_accuracy.append(tmp_acc)

        # Evaluation
        with torch.no_grad():
            correct = 0
            total = 0
            tmp_val_loss = []
            model.eval()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss_val = loss_fn(outputs, labels)
                _, predicted = outputs.max(dim=1)
                tmp_val_loss.append(loss_val.item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_acc = 100 * correct / total

            loss_val = np.mean(tmp_val_loss)
            val_loss.append([loss_val, total_steps_train])
            val_accuracy.append([val_acc, epoch])

        print(f'Val_loss: {loss_val: .2f} , '
              f'Validation_acc: {val_acc} %')
    return train_loss, train_accuracy, val_loss, val_accuracy, model


# Train the model and plot the Losses and the accuracies
def task_training_1_2_3_4():
    train_loader, val_loader = task_dataset_3(batch_size=32)
    data = train_model(train_loader, val_loader, ep=20, momentum=0.9, lr=0.001, bs=32, dropout=0)

    train_loss = data[0]
    train_accuracy = data[1]
    val_loss = data[2]
    val_accuracy = data[3]

    best_val_acc = max(val_accuracy)
    print(f"\n\nBest validation accuracy: {best_val_acc[0]} in epoch: {best_val_acc[1]}\n\n")

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    t_loss = train_loss[:, 0]
    v_loss = val_loss[:, 0]

    train_steps = train_loss[:, 1]
    val_steps = val_loss[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_steps, t_loss, 'b', label="Train loss")
    ax.plot(val_steps, v_loss, 'r', label="Val loss")
    plt.legend()
    plt.show()

    val_accuracy = np.array(val_accuracy)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(train_steps, train_accuracy, 'b', label="Train acc")
    ax.plot(val_steps, val_accuracy[:, 0], 'r', label="Val acc")
    plt.legend()
    plt.show()


# Try multiple combinations of Hyper-parameters to find the best one
def task_training_6():
    train_loader, val_loader = task_dataset_3(batch_size=32)
    learning_rate = [0.001, 0.005, 0.008]
    momentum = [0.75, 0.9]
    dropout = [0.4, 0.6]
    track = []
    val_losses = []
    val_accuracies = []
    best_accuracies = []

    for do in dropout:
        for m in momentum:
            for lr in learning_rate:
                data = train_model(train_loader, val_loader, 35, m, lr, 32, do)
                conf = f"Drop:{do}, mom:{m}, lr:{lr}"
                track.append(conf)
                accs = data[3]
                best_accuracies.append([max(accs), conf])
                val_losses.append(data[2])
                val_accuracies.append(data[3])

    val_losses = [np.array(val) for val in val_losses]
    val_accuracies = [np.array(val) for val in val_accuracies]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'pink', 'gray', 'cyan', 'slateblue', 'aquamarine', 'sienna',
              'tan']
    for i, val in enumerate(colors):
        ax.plot(val_losses[0][:, 1], val_losses[i][:, 0], val, label=track[i])
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, val in enumerate(colors):
        ax.plot(val_accuracies[0][:, 1], val_accuracies[i][:, 0], val, label=track[i])
    plt.legend()
    plt.show()


# Train and Test the model with the best Hyper-parameters found in task_training_6()
def task_training_7():
    train_loader, val_loader = task_dataset_3(batch_size=32)
    model_data = train_model(train_loader, val_loader, ep=35, momentum=0.75, lr=0.008, bs=32, dropout=0.4)
    model = model_data[4]

    test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=trans_train)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=32, shuffle=False)

    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()  # Set model in eval mode. Don’t forget!
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print(f'Test accuracy: {test_acc} %')
        print(f'Test error rate: {100 - 100 * correct / total: .2f} %')


if __name__ == '__main__':
    # Execution of tasks 1 (Dataset)
    task_dataset_1()
    # Execution of tasks 2 (Dataset)
    task_dataset_2()
    # Execution of tasks 3 (Dataset)
    task_dataset_3(batch_size=32)

    # Execution of tasks 1,2,3,4 (Training)
    task_training_1_2_3_4()

    # Task 5 is the Model class defined in the global scope

    # Execution of task 6 (Training)
    task_training_6()
    # Execution of task 7 (Training)
    task_training_7()
