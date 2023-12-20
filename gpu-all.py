import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import json
import os

# !put seed here, because I want nn.Sequential and others like train_loader and sample_generator to
# produce the same init/sample at each run, so that I can find the problem and solve it
# torch.manual_seed(20010302)
device = torch.device("cuda")
# print(device)

input_size = 150
output_size = 15

# Create the MLP model using nn.Sequential
MLP = nn.Sequential(
    nn.Linear(input_size, 100),
    nn.ReLU(),
    nn.Linear(100, 15)
).to(device)


# A specific loss function for a concrete problem
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, y_label):
        y_pred = torch.max(y_pred, torch.ones_like(y_pred)) ** 2
        y_label = torch.max(y_label, torch.ones_like(y_label)) ** 2
        loss = torch.mean(y_label / y_pred + y_pred / y_label)
        return loss

# Define training model
def train(model, criterion, optimizer, train_loader, number_epochs):
    # print(list(MLP.parameters()))
    loss_list = []

    for epoch in range(number_epochs):
        loss_run = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and accuracy record
            loss_run += loss.item()

        loss_run /= len(train_loader)
        loss_list.append(loss_run)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{number_epochs}], Loss: {loss_run:.4f}')

    # draw the curves
    plt.semilogy(loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


# make train_loader & test_loader
with open('data.json', 'r') as f:
    data_raw = json.load(f)

x_raw = torch.tensor([[int(i) for i in k.split()] for k in data_raw.keys()], dtype=torch.float)
y_raw = torch.tensor(list(data_raw.values()), dtype=torch.float)
dataset = data.TensorDataset(x_raw, y_raw)
train_dataset, test_dataset = data.random_split(dataset,
                                                [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_loader = data.DataLoader(train_dataset, batch_size=int(len(train_dataset)), shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# train the model
learning_rate = 1e-5

num_epochs = 10000
# criterion = nn.MSELoss().to(device)
criterion = CustomLoss().to(device)

optimizer = optim.Adam(MLP.parameters(), lr=1, betas=(0.9, 0.999))
# optimizer = optim.SGD(MLP.parameters(), learning_rate)

# print(next(MLP.parameters()).device)

train(MLP, criterion, optimizer, train_loader, num_epochs)

os.system('truncate -s 0 predicts.output')
q_errors = [(0,0)]*output_size
for inputs, labels in test_loader:
    outputs = MLP(inputs)
    # loss = criterion(outputs, labels)
    outputs = [max(1, round(float(i))) for i in outputs[0]]
    q_error = [float(max(outputs[i]/labels[0][i], labels[0][i]/outputs[i])) for i in range(len(outputs))]
    q_error = [0 if math.isnan(e) else round(e, 2) for e in q_error]
    with open('predicts.output', 'a') as outfile:
        outfile.write(str(q_error)+'\n')
    q_errors = [(round(q_errors[i][0]+q_error[i],  2), q_errors[i][1]+1) if math.isinf(q_error[i])==False else q_errors[i] for i in range(output_size)]
print(q_errors)
print([round(t[0]/t[1], 2) for t in q_errors])