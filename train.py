import torch
from model import ActorCritic
from utils import *
from ac_utils import *
from test import test
import time

def train(p_i, shared_model, env, params, optimizer, lock, counter, layers_, device, actions_name, shared_ep, shared_r, res_queue):
    
    print(' ----- TRAIN PHASE -----')
    #seed = 1
    #torch.manual_seed(seed + p_i)
    #env.seed(seed + p_i)

    #create instance of the model
    model = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                        kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], layer3=layers_['hidden_dim3'], kernel_size3=layers_['kernel_size3'], stride3=layers_['stride3'],
                        fc1_dim=layers_['fc1'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim'],hidden_size=layers_['hidden_size_lstm']) #.to(device)

    model.train()
    
    #list for gradient updating
    values = []
    log_probs = []
    rewards = []
    masks = []
    entropies = []
    actions = []
    tot_rew = 0
    steps_array = []

    #reset env
    queue = deque(maxlen=4)
    in_state_i = env.reset()
    #initialize a queue for each env, preprocess each frame and obtain a vecotr of 84,84,4
    frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
    #stack the frames together
    input_frames = stack_frames(frame_queue)
    current_state = input_frames
    #print(current_state.shape)
    done = True
    episode_length = 0
    
    update_i = 0
    
    while shared_ep.value < params['max_games']:
        #Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        #lstm hidden and cell state initialiazation
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        #rollout_step
        steps_array, entropies, entropy, episode_length, frame_queue, current_state, done, tot_rew = rollout(params['rollout_size'], model, hx, cx, frame_queue, env, current_state,
                                                                                  episode_length, params['max_steps'], steps_array, values, log_probs, rewards, masks, actions, entropies, actions_name, layers_['n_frames'], device, tot_rew)


        #assert len(log_probs) == len(rewards) == len(masks)
        #assert len(log_probs) == len(values)-1

        #compute GAE (Generalized Advantage Estimate)
        returns, gaes = GAE(steps_array, params['gamma'], params['lambd'], device)
        #print('advs', advs.shape)
        
        # compute losses and update parameters
        a3c_loss, value_loss, policy_loss = upgrade_parameters(returns, gaes, steps_array, params['value_coeff'], entropies, params['entropy_coef'])
        
        optimizer.zero_grad()
        a3c_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
            

        #empty lists
        values = []
        log_probs = []
        rewards = []
        masks = []
        actions = []
        entropies = []
        steps_array = []
        
        if update_i % 100 == 0:
            print(f'Process: {p_i} \n Update: {update_i} \n Policy_Loss: {policy_loss.item()} \n Value_Loss: {value_loss.item()} \n A3C loss: {a3c_loss.item()} \n Entropy: {entropy}')
            print('------------------------------------------------------')

        if done:
            print('\n')
            record(shared_ep, shared_r, tot_rew, res_queue, p_i)
            print('\n')
            tot_rew = 0
            #time.sleep(20)
        
        #if update_i % 500 == 0:
        #    with lock:
                #torch.save(shared_model.state_dict(), './sM_weights.pth')
        #        test(shared_model, counter, env, params['max_steps'], layers_, actions_name, lock, device)
        
        update_i += 1