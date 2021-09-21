import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

torch.manual_seed(73)

train_data = datasets.MNIST(
    "data", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.MNIST(
    "data", train=False, download=True, transform=transforms.ToTensor()
)

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)


class Net(torch.nn.Module):
    def __init__(self, hiddens=[128, 92, 64, 32], output=10):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, hiddens[0])
        self.fc2 = torch.nn.Linear(hiddens[0], hiddens[1])
        self.fc3 = torch.nn.Linear(hiddens[1], hiddens[2])
        self.fc4 = torch.nn.Linear(hiddens[2], hiddens[3])
        self.fc5 = torch.nn.Linear(hiddens[3], output)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x


def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # model in training mode
    model.train()
    for epoch in range(1, n_epochs + 1):

        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # model in evaluation mode
    model.eval()
    return model


model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = train(model, train_loader, criterion, optimizer, 10)


def test(model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # model in evaluation mode
    model.eval()

    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


test(model, test_loader, criterion)


# Homomorphic side

import numpy as np
import hnumpy as hnp
from time import time

DATA_RANGE = (0, 1.0)
INPUT_SIZE = 28 * 28

fc = {
    1: (np.array(model.fc1.weight.T.tolist()), np.array(model.fc1.bias.tolist())),
    2: (np.array(model.fc2.weight.T.tolist()), np.array(model.fc2.bias.tolist())),
    3: (np.array(model.fc3.weight.T.tolist()), np.array(model.fc3.bias.tolist())),
    4: (np.array(model.fc4.weight.T.tolist()), np.array(model.fc4.bias.tolist())),
    5: (np.array(model.fc5.weight.T.tolist()), np.array(model.fc5.bias.tolist())),
}


def inference(x):
    for i in range(1, 6):
        x = np.dot(x, fc[i][0]) + fc[i][1]
        x = 1 / (1 + np.exp(-x))
    return x


# compile the function
config = hnp.config.CompilationConfig(parameter_optimizer="handselected")
h = hnp.homomorphic_fn(
    inference,
    hnp.encrypted_ndarray(bounds=DATA_RANGE, shape=(INPUT_SIZE,)),
    config=config,
)

expected_results = []
results = []
targets = []
times = []

# don't batch
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

n_iter = 100
i = 0
for data, target in test_loader:
    i += 1
    x = np.array(data.tolist()).flatten()
    expected_results.append(inference(x))
    tick = time()
    results.append(h(x))
    tock = time()
    times.append(tock - tick)
    targets.append(target)
    if i == n_iter:
        break

diff_plain_enc = 0
diff_plain = 0
diff_enc = 0
for i in range(n_iter):
    p = results[i][0].argmax()
    ep = expected_results[i].argmax()
    t = targets[i]
    if p != ep:
        diff_plain_enc += 1
    if ep != t:
        diff_plain += 1
    if p != t:
        diff_enc += 1

print(f"diff between encrypted and plain output is {diff_plain_enc}")
print(f"plain accuracy {(100 - diff_plain) / n_iter}")
print(f"enc accuracy {(100 - diff_enc) / n_iter}")
print(f"average time {sum(times[1:]) / 99}")
