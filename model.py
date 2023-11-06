import torch
import numpy as np

#original net A3C-LSTM used in https://arxiv.org/pdf/1602.01783.pdf
class ActorCritic(torch.nn.Module):
    def __init__(self,input_shape, layer1, kernel_size1, stride1, layer2, kernel_size2, stride2, fc1_dim, lstm_dim, out_actor_dim, out_critic_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_shape, out_channels=layer1, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = torch.nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=kernel_size2, stride=stride2)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=32*9*9, out_features=fc1_dim)
        self.out_actor = torch.nn.Linear(in_features=lstm_dim, out_features=out_actor_dim)
        self.out_critic = torch.nn.Linear(in_features=lstm_dim, out_features=out_critic_dim)
        #lstm cell
        self.lstm_cell = torch.nn.LSTMCell(fc1_dim, lstm_dim)
        
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        for name, param in self.lstm_cell.named_parameters():
            if 'bias' in name:
                param.data.zero_()
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_critic.weight)
        self.out_critic.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.out_actor.weight)
        self.out_actor.bias.data.zero_()
                

    def forward(self,x):
        x, (hx, cx) = x
        out_backbone = self.conv1(x)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv2(out_backbone)
        out_backbone = self.relu(out_backbone)
        out = out_backbone.reshape(-1,32*9*9)
        out = self.fc1(out)
        out = self.relu(out)
        #lstm cell
        hx, cx = self.lstm_cell(out, (hx, cx))
        out = hx
        #actor
        actor = self.out_actor(out)
        #critic
        critic = self.out_critic(out)
        
        return actor,critic,(hx, cx)
