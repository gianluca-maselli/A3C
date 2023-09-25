import torch
import numpy as np

#weight init

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self,input_shape, layer1, kernel_size1, stride1, layer2, kernel_size2, stride2, layer3, kernel_size3, stride3, fc1_dim, out_actor_dim, out_critic_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_shape, out_channels=layer1, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = torch.nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=kernel_size2, stride=stride2)
        self.conv3 = torch.nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=kernel_size3, stride=stride3)

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        #self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=fc1_dim)
        self.out_actor = torch.nn.Linear(in_features=hidden_size, out_features=out_actor_dim)
        self.out_critic = torch.nn.Linear(in_features=hidden_size, out_features=out_critic_dim)

        # LSTM layer
        self.lstm_cell = torch.nn.LSTMCell(64*7*7, hidden_size=hidden_size)
        
        self.apply(weights_init)
        self.out_actor.weight.data = normalized_columns_initializer(
            self.out_actor.weight.data, 0.01)
        self.out_actor.bias.data.fill_(0)
        self.out_critic.weight.data = normalized_columns_initializer(
            self.out_critic.weight.data, 1.0)
        self.out_critic.bias.data.fill_(0)

        self.lstm_cell.bias_ih.data.fill_(0)
        self.lstm_cell.bias_hh.data.fill_(0)
        
        self.train()


    def forward(self,x):
        x, (hx, cx) = x
        out_backbone = self.conv1(x)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv2(out_backbone)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv3(out_backbone)
        out_backbone = self.relu(out_backbone)
        #flatten for lstm
        out = out_backbone.reshape(-1, 64*7*7)
        hx, cx = self.lstm_cell(out, (hx, cx))
        out_lstm = hx
        #out = self.fc1(out_lstm)
        #out = self.relu(out)
        #actor
        actor = self.out_actor(out_lstm)
        #critic
        critic = self.out_critic(out_lstm)
        return actor,critic, (hx, cx)
