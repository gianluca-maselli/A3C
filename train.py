import torch
from model import ActorCritic
from utils import *
from ac_utils import *
from test import test
import time
import gym
import numpy as np 

def train(p_i, shared_model, p, optimizer, lock, counter, lys, avg_ep, scores, scores_avg, flag_exit):
    
    params = p.copy()
    layers_ = lys.copy()
    
    seed = params['seed']
    torch.manual_seed(seed + p_i)
    np.random.seed(seed + p_i)
    
    env = gym.make(params['env_name'])
    actions_name = env.unwrapped.get_action_meanings()
    
    print(' ----- TRAIN PHASE -----')
    
    #create instance of the model
    model = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                        kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], fc1_dim=layers_['fc1'], 
                        lstm_dim=layers_['lstm_dim'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim'])

    if optimizer is None:
        optimizer = torch.optim.Adam(shared_model.parameters(), lr=params['lr'])
    
    model.train()
    
    #reset env
    queue = deque(maxlen=4)
    in_state_i = env.reset(seed=(seed + p_i))
    #initialize a queue for each env, preprocess each frame and obtain a vecotr of 84,84,4
    frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
    #stack the frames together
    input_frames = stack_frames(frame_queue)
    current_state = input_frames
    episode_length = 0
    tot_rew = 0
    
    #initialization lstm hidden state
    hx = torch.zeros(1, layers_['lstm_dim'])
    cx = torch.zeros(1, layers_['lstm_dim'])
    
    while True:
        
        #stop workers when the avg > mean reward
        if flag_exit.value == 1:
            print(f"Terminating process n. {p_i}...")
            break
        # optimizer.zero_grad(set_to_none=False) necessary for PyTorch Versions >= 2.0.0
        optimizer.zero_grad()
        #Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
                
        #rollout_step
        hx, cx, steps_array, episode_length, frame_queue, current_state, tot_rew, counter, flag_finish, scores_avg = rollout(p_i, counter, params, model, hx, cx, frame_queue, env, current_state,
                                                                                  episode_length, actions_name, layers_, tot_rew, scores, lock, avg_ep, scores_avg)
        if flag_finish == True:
            print('Save Model...')
            if params['env_name'] == 'PongNoFrameskip-v4':
                torch.save(shared_model,'./saved_model/shared_model_pong.pt')
            elif params['env_name'] == 'BreakoutNoFrameskip-v4':
                torch.save(shared_model, './saved_model/shared_model_break.pt')
            plot_avg_scores(scores_avg, 'Plot AVG Scores')
            
            with flag_exit.get_lock():
                flag_exit.value = 1
            
            break
            
        #compute expected returns
        probs, log_probs, action_log_probs, advantages, returns, values = compute_returns(steps_array, params['gamma'], model)

        # compute losses and update parameters
        a3c_loss, value_loss, policy_loss, entropy_loss = update_parameters(probs, log_probs, action_log_probs, advantages, returns, values, params['value_coeff'], params['entropy_coef'])
        
        a3c_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        
        if counter.value % 100 == 0:
            print(f'Process: {p_i} \nUpdate: {counter.value} \nPolicy_Loss: {policy_loss.item()} \nValue_Loss: {value_loss.item()} \nEntropy_Loss: {entropy_loss.item()} \nA3C loss: {a3c_loss.item()} \n')
            print('------------------------------------------------------')
                            
        with counter.get_lock():
            counter.value += 1
