import torch
import torch.nn as nn
import  matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#writer = SummaryWriter('./logs/fit')

num_inputs = 100
num_output = 100
num_hidden = 64

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),

            #nn.Linear(num_hidden, num_hidden),
            #nn.ReLU(),
            #nn.Linear(num_hidden, num_hidden),
            #nn.ReLU(),
            #nn.Linear(num_hidden, num_hidden),
            #nn.ReLU(),
            nn.Linear(num_hidden, num_output),

        )
        self.initialize_weights()
    def forward(self, x):
        return self.net(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

def function(x):
    return torch.sin(16*x)

inputdata = torch.arange(-100, 100, 2.0,requires_grad=True)



#inputdata = (inputdata - inputdata.mean()) / (inputdata.std()*0.5)
target = function(inputdata)
print(target)
#target = (target_half - target_half.mean()) / (target_half.std()*0.5)
Loss = nn.MSELoss(reduction='mean')


def train(input_data, target):
    model = MLP()
    num_epochs = 300
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(),lr= lr ,weight_decay=1e-6)

   # writer.add_graph(model, input_data)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_data)
        print(output)
        loss = Loss(output, target)
        loss.sum().backward(retain_graph=True)

        optimizer.step()

        #writer.add_scalar('Loss/train', loss.mean().item(), epoch)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.mean().item():.4f}')
            # 绘制训练过程中模型的输出与真实数据的比较
            plt.figure(figsize=(10, 5))
            plt.title(f'Epoch {epoch + 1}')
            plt.plot(input_data.detach().numpy(), target.detach().numpy(), label='Target')
            plt.plot(input_data.detach().numpy(), output.detach().numpy(), label='Predictions')
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.legend()
            plt.grid(True)
            plt.show()
    #writer.close()
train(inputdata, target)