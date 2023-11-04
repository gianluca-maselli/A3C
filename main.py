import argparse
import torch
import torch.multiprocessing as mp
from model import ActorCritic
from shared_optim import SharedAdam, SharedRMSprop
from train import train
import gym
from test import test
import os
#from torchsummary import summary
import time
import sys

parser = argparse.ArgumentParser(description='A3C')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=40,
                    help='value to clip the grads (default: 40)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--rs', type=int, default=20,
                    help='rollout size before updating (default: 20)')
parser.add_argument('--n-workers', type=int, default=os.cpu_count(),
                    help='how many training processes to use (default: os cpus)')
parser.add_argument('--ep-length', type=int, default=4e10,
                    help='maximum episode length (default: 4e10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--opt', default='adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--use-trained', type=bool, default=False,
                    help='training A3C from scratch (default: False)')

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method('spawn')
    #parser
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    #generate the environment
    env_name = args.env_name
    env = gym.make(env_name)
    #get the dimension of the env
    space = env.observation_space.shape
    print('Space dim: ', space)
    #get the available actions
    actions = env.action_space.n
    print('n. of actions: \n', actions)
    del env
          
    #training parameters
    params = {
        'seed': args.seed,
        'env_name': args.env_name,
        'max_ep_length': args.ep_length,
        'gamma': args.gamma,
        'entropy_coef':args.entropy_coef,
        'value_coeff':args.value_loss_coef,
        'lr':args.lr, 
        'n_process': args.n_workers,
        'optimizer': args.opt, 
        'max_grad_norm': args.max_grad_norm,
        'rollout_size': args.rs,
        'use_pre_trained': args.use_trained
    }
    
    if params['env_name'] == 'PongNoFrameskip-v4':
        params.update({'mean_reward': 18.0, })
    elif params['env_name'] == 'BreakoutNoFrameskip-v4':
        params.update({'mean_reward': 60.0})
    else:
        print('No available env')
        sys.exit(1)
    #A3C parameters
    layers_ = {
        'n_frames':4,
        #conv net dim
        'hidden_dim1':16, 
        'kernel_size1':8,
        'stride1':4,
        'hidden_dim2':32,
        'kernel_size2':4,
        'stride2':2,
        #fully_connected dims
        'fc1':256,
        'lstm_dim':256,
        'out_actor_dim': actions,
        'out_critic_dim':1,
    }
    if params['use_pre_trained'] == False:

        shared_ac = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                                kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], fc1_dim=layers_['fc1'], 
                                lstm_dim=layers_['lstm_dim'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim'])

        shared_ac.share_memory()
        #shared optimizer
        if params['optimizer'] == 'adam':
            optimizer = SharedAdam(shared_ac.parameters(), lr=params['lr'])
            optimizer.share_memory()
        elif params['optimizer'] == 'rmsprop':
            optimizer = SharedRMSprop(shared_ac.parameters(), lr=params['lr'])
            optimizer.share_memory()
        else:
            optimizer = None
            
        counter_updates = mp.Value('i', 0)
        lock = mp.Lock()
        
        avg_ep = mp.Value('i', 0)
        scores = mp.Manager().list()
        scores_avg = mp.Manager().list()
        flag_exit = mp.Value('i', 0)
        trained = False

        n_processes = params['n_process']
        print('----------- TRAINING INFO ------------')
        print('Optimizer: ', optimizer)
        print('n_processes: ', n_processes)
        print('rollout size: ', params['rollout_size'])
        print('--------------------------------------')
        
        processes = []

        for p_i in range(0, n_processes):
            p = mp.Process(target=train, args=(p_i, shared_ac, params, optimizer,lock, counter_updates, layers_, avg_ep, scores, scores_avg, flag_exit))
            p.start()
            processes.append(p)
        time.sleep(5)
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        
        trained = True
    
    else:
        print('load the model...')
        if params['env_name'] == 'PongNoFrameskip-v4':
            shared_ac = torch.load('./saved_model/shared_model_pong.pt')
            trained = True
        elif params['env_name'] == 'BreakoutNoFrameskip-v4':
            shared_ac = torch.load('./saved_model/shared_model_break.pt')
            trained = True
        else:
            print('No available trained model')
            sys.exit(1)
        
    #test
    if params['use_pre_trained'] == True or trained == True:
        test(params['n_process'], shared_ac, params, params['max_ep_length'], layers_)
