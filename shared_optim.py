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
                

    #performs a single optimization step
'''            print('------------------------------------------------------')

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                #exp_avg.mul_(beta1).add_(1 - beta1, grad)
                m_ = torch.mul(exp_avg, beta1)
                exp_avg = torch.add(m_, grad, alpha=(1 - beta1))


                #exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                m_2 = torch.mul(exp_avg_sq, beta2)
                exp_avg_sq = torch.addcmul(m_2, grad, grad, value=(1 - beta2))

                #denom = exp_avg_sq.sqrt().add_(group['eps'])
                denom = torch.add(torch.sqrt(exp_avg_sq), group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data = torch.addcdiv(p.data, exp_avg, denom, value=-step_size)

        return loss
            print('------------------------------------------------------')

'''
        
class SharedRMSprop(torch.optim.RMSprop):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

        # State initialisation (must be done before step, else will not be shared between threads)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # g = αg + (1 - α)Δθ^2
                #square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                # θ ← θ - ηΔθ/√(g + ε)
                m_ = torch.mul(square_avg, alpha)
                square_avg = torch.addcmul(m_, grad, grad, value=(1 - alpha))
                # θ ← θ - ηΔθ/√(g + ε)
                #avg = square_avg.sqrt().add_(group['eps'])
                avg = torch.add(torch.sqrt(square_avg), group['eps'])
                
                p.data = torch.addcdiv(p.data, grad, avg, value=-group['lr'])
                
        return loss      
                
                
