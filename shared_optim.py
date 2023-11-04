import torch
import math

#iterate over all the parameters
#setting the steps, exponential average and exponential average squared to zeroes effectively.
#share this parameters among the different pools in our multi-threading pool.
#N.B. this is nothing more than the code for the Adam optimizer, however it thought to work with multiple threads.

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam,self).__init__(params, lr, betas, eps, weight_decay)
        #setting initial values
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)[0]
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                
    
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class SharedRMSprop(torch.optim.RMSprop):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

        # State initialisation (must be done before step, else will not be shared between threads)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)[0]
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()