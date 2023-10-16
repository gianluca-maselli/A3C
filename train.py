import torch
from model import ActorCritic
from utils import *
from ac_utils import *
from test import test
import time

def train(p_i, shared_model, env, params, optimizer, lock, counter, layers_, device, actions_name, shared_ep, shared_r, res_queue, avg_ep, scores, scores_avg):
    
    print(' ----- TRAIN PHASE -----')
   
    seed = 1
    torch.manual_seed(seed + p_i)
    env.seed(seed + p_i)

    #create instance of the model
    model = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                        kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], layer3=layers_['hidden_dim3'], kernel_size3=layers_['kernel_size3'], stride3=layers_['stride3'],
                        fc1_dim=layers_['fc1'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim']) #.to(device)

    model.train()
    
    #list for gradient updating
    values = []
    log_probs = []
    rewards = []
    masks = []
    entropies = []
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
    tot_rew = 0
    mean_reward = 18.0
    flag_finish = False
    
    while counter.value < params['updates']:
        #Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        #rollout_step
        steps_array, entropies, entropy, episode_length, frame_queue, current_state, done, tot_rew = rollout(params['rollout_size'], model, frame_queue, env, current_state,
                                                                                  episode_length, params['max_steps'], steps_array, values, log_probs, rewards, masks, entropies, actions_name, layers_['n_frames'], device, tot_rew)
        #assert len(log_probs) == len(rewards) == len(masks)
        #assert len(log_probs) == len(values)-1

        #compute GAE (Generalized Advantage Estimate)
        returns, gaes = GAE(steps_array, params['gamma'], params['lambd'], device)
        #print('advs', advs.shape)
        
        # compute losses and update parameters
        a3c_loss, value_loss, policy_loss, entropy_loss = upgrade_parameters(returns, gaes, steps_array, params['value_coeff'], entropies, params['entropy_coef'])
        
        optimizer.zero_grad()
        a3c_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
        #for  shared_param in shared_model.parameters():
        #    print('shared_param.grad', shared_param.grad)
        #    print('\n')
        #    print('grads:', shared_param._grad)
        #    break
        #break
            #if shared_param.grad is not None:
        #    return
        #    print('grads:', shared_param._grad)
        #    break

        #empty lists
        values = []
        log_probs = []
        rewards = []
        masks = []
        entropies = []
        steps_array = []
        
        if counter.value % 100 == 0 or done:
            print(f'Process: {p_i} \n Update: {counter.value} \n Policy_Loss: {policy_loss.item()} \n Value_Loss: {value_loss.item()} \n A3C loss: {a3c_loss.item()} \n Entropy: {entropy_loss.item()}')
            print('------------------------------------------------------')

        if done:
            flag_finish, scores_avg = print_avg(scores, p_i, tot_rew, lock, avg_ep, params, mean_reward, flag_finish, scores_avg)                        
            print('\n')
            record(shared_ep, shared_r, tot_rew, res_queue, p_i)
            #print_avg(avg_ep, list_scores, lock, dict_scores)
            print('\n')
            tot_rew = 0
            #time.sleep(20)
            
            if flag_finish == True:
                print('Save Model...')
                torch.save(shared_model,'./shared_model.pt')
                plot_avg_scores(scores_avg, 'Plot AVG Scores')
                break
                
            
        with counter.get_lock():
            counter.value += 1
    
        #if update_i % 500 == 0:
        #    with lock:
                #torch.save(shared_model.state_dict(), './sM_weights.pth')
        #        test(shared_model, counter, env, params['max_steps'], layers_, actions_name, lock, device)
        
    res_queue.put(None)