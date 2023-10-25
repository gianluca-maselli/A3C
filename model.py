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
    def __init__(self,input_shape, layer1, kernel_size1, stride1, layer2, kernel_size2, stride2, layer3, kernel_size3, stride3, fc1_dim, out_actor_dim, out_critic_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_shape, out_channels=layer1, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = torch.nn.Conv2d(in_channels=layer1, out_channels=layer2, kernel_size=kernel_size2, stride=stride2)
        self.conv3 = torch.nn.Conv2d(in_channels=layer2, out_channels=layer3, kernel_size=kernel_size3, stride=stride3)
        #self.conv4 = torch.nn.Conv2d(in_channels=layer3, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(in_features=32*9*9, out_features=fc1_dim)
        self.out_actor = torch.nn.Linear(in_features=fc1_dim, out_features=out_actor_dim)
        self.out_critic = torch.nn.Linear(in_features=fc1_dim, out_features=out_critic_dim)
       
        
        self.apply(weights_init)
        self.out_actor.weight.data = normalized_columns_initializer(
            self.out_actor.weight.data, 0.01)
        self.out_actor.bias.data.fill_(0)
        self.out_critic.weight.data = normalized_columns_initializer(
            self.out_critic.weight.data, 1.0)
        self.out_critic.bias.data.fill_(0)
        
        self.train()
        

    def forward(self,x):
        out_backbone = self.conv1(x)
        out_backbone = self.relu(out_backbone)
        out_backbone = self.conv2(out_backbone)
        out_backbone = self.relu(out_backbone)
        #out_backbone = self.conv3(out_backbone)
        #out_backbone = self.relu(out_backbone)
        #out_backbone = self.conv4(out_backbone)
        #out_backbone = self.relu(out_backbone)
        out = self.flatten(out_backbone)
        #out = out_backbone.reshape(-1,32*9*9)
        #out = self.flatten(out_backbone)
        out = self.fc1(out)
        out = self.relu(out)
        #actor
        actor = self.out_actor(out)
        #critic
        critic = self.out_critic(out)
        return actor,critic
