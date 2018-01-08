import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # shape=(100,1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

plt.ion()

for t in range(300):
    prediction = net(x)

    loss = loss_func(prediction, y)
    optimizer.zero_grad() #clear gradients for next train
    loss.backward()
    optimizer.step() # apply gradients

    if t % 5 == 0:
        plt.cla()
        sc = plt.scatter(x.data.numpy(), y.data.numpy(), label='real data')
        lines = plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5, label='prediction data')
        plt.text(0.5, 0, 'loss= %.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.legend(loc='upper left')
        plt.draw()        
        plt.pause(0.1)

plt.ioff()
plt.show()
