import torch
import multiprocessing as mp
from model import ActorCritic
from shared_optim import SharedAdam
from train import train
import gym
from test import test


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    #generate the environment
    env_name = "PongNoFrameskip-v4"
    env = gym.make(env_name)
    #get the dimension of the env
    space = env.observation_space.shape
    print('Space dim: ', space)
    #get the available actions
    actions = env.action_space.n
    print('n. of actions: \n', actions)
    actions_name = env.unwrapped.get_action_meanings()
    print('Available actions: \n', actions_name)
    
    useGPU = 0
    if torch.cuda.is_available():
        dev = "cuda:0"
        useGPU = 1
    else:
        dev = "cpu"
        useGPU = 0

    device = torch.device(dev)
    print('Device: ', device)
    
    #AC parameters
    layers_ = {
        'n_frames':4,
        #conv net dim
        'hidden_dim1':32, 
        'kernel_size1':8,
        'stride1':4,
        'hidden_dim2':64,
        'kernel_size2':4,
        'stride2':2,
        'hidden_dim3':64,
        'kernel_size3':3,
        'stride3':1,
        #fully_connected dims
        'fc1':512,
        'out_actor_dim':6, #n_actions
        'out_critic_dim':1,
        'hidden_size_lstm':256 #lstm hidden dim
    }
    
    #train parameters

    params = {
        'max_games': int(4e10),
        'gamma': 0.99,
        'lambd':1.0,
        'entropy_coef':0.01,
        'value_coeff':0.5,
        'rollout_size':5,
        'max_steps':1000000,
        'lr':0.0001
    }
    
    mp.set_start_method('spawn', force=True)
    
    shared_ac = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                        kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], layer3=layers_['hidden_dim3'], kernel_size3=layers_['kernel_size3'], stride3=layers_['stride3'],
                        fc1_dim=layers_['fc1'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim'],hidden_size=layers_['hidden_size_lstm']) #.to(device)

    shared_ac.share_memory()
    #shared optimizer
    optimizer = SharedAdam(shared_ac.parameters(), lr=params['lr'])
    optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    shared_ep, shared_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    lock = mp.Lock()

    #n_processes = mp.cpu_count() 
    n_processes = 4
    print('n_processes: ', n_processes)
    
    #p = mp.Process(target=test, args=(shared_ac, counter, env, params['max_steps'], layers_, actions_name, lock, device))
    #p.start()
    #processes.append(p)

    for p_i in range(0, n_processes):
        p = mp.Process(target=train, args=(p_i, shared_ac, env, params, optimizer,lock, counter, layers_, device, actions_name, shared_ep, shared_r, res_queue))
        p.start()
        processes.append(p)
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    for p in processes:
        p.join()
    #for p in processes:
    #    p.terminate()