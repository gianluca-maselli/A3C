import torch
import torch.nn.functional as F
from utils import *

def compute_log_prob_actions(logits):
    prob_v = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=prob_v)
    action = dist.sample().detach()
    return action.numpy()[0]


def rollout(p_i, counter, params, model, hx, cx, frame_queue, env, current_state, episode_length, actions_name, layers_, tot_rew, scores, lock, avg_ep, scores_avg):
    
    #empty lists
    states = []
    actions = []
    rewards = []
    masks = []
    hx_s = []
    cx_s = []
    steps_array = []
    
    flag_finish = False
    
    for _ in range(params['rollout_size']):
        episode_length +=1
        
        current_state = current_state.unsqueeze(0).permute(0,3,1,2)
        with torch.no_grad():
            #compute logits, values and hidden and cell states from the current state
            logits, _ , (hx_, cx_)  = model((current_state,(hx, cx)))
            #get action
            action = compute_log_prob_actions(logits)
        
        #permorm step in the env
        next_frame, reward, done, _ = skip_frames(action,env,skip_frame=4)
        #reward = max(min(reward, 1), -1)
        
        states.append(current_state)
        actions.append(action)
        rewards.append(np.sign(reward).astype(np.int8))
        masks.append(done)
        hx_s.append(hx)
        cx_s.append(cx)
        
        tot_rew +=reward
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)
        current_state = next_state
        hx, cx = hx_, cx_
        
        if episode_length > params['max_ep_length']:
            break
        
        if done:
            #reset env
            in_state_i = env.reset()
            frame_queue = initialize_queue(frame_queue, layers_['n_frames'], in_state_i, env, actions_name)
            #stack the frames together
            input_frames = stack_frames(frame_queue)
            current_state = input_frames
            episode_length = 0
            print(
                "Process: ", p_i,
                "Update:", counter.value,
                "| Ep_r: %.0f" % tot_rew,
            )
            print('------------------------------------------------------')
            flag_finish, scores_avg = print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, scores_avg)                        
            print('\n')
            if flag_finish == True:
                break
            
            tot_rew = 0
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])
        
    #bootstrapping
    with torch.no_grad():
        _, f_value , _  = model((current_state.unsqueeze(0).permute(0,3,1,2),(hx_, cx_)))
    
    steps_array.append((states, actions, rewards, masks, hx_s, cx_s, f_value))
    
    return hx, cx, steps_array, episode_length, frame_queue, current_state, tot_rew, counter, flag_finish, scores_avg


def compute_returns(steps_array, gamma, model):
    states, actions, rewards, masks, hx_s, cx_s, f_value = steps_array[0]
    
    R = f_value
    returns  = torch.zeros(len(rewards),1)
    for j in reversed(range(len(rewards))):
        R = rewards[j] + R * gamma * (1-masks[j])
        returns[j] = R
    
    #batch of states
    s = torch.concat(states, dim=0)
    a = torch.tensor(actions).unsqueeze(1)
    hxs = torch.cat(hx_s)
    cxs = torch.cat(cx_s)
    
    #compute probs and logproba
    logits, values, _ = model((s,(hxs, cxs)))
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    #gather logprobs with respect the chosen actions
    action_log_probs = log_probs.gather(1, a)
    #advantages
    advantages = returns-values
    
    return probs, log_probs, action_log_probs, advantages, returns, values
    
    
def ensure_shared_grads(local_model, shared_model):
    for param, shared_param in zip(local_model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.grad = param.grad 
    

def update_parameters(probs, log_probs, action_log_probs, advantages, returns, values, value_coeff, entropy_coef):
    #policy loss
    policy_loss = -(action_log_probs * advantages.detach()).mean() 
    #value loss
    value_loss = torch.nn.functional.mse_loss(values, returns)
    #entropy loss
    entropy_loss = (probs * log_probs).sum(dim=1).mean()
    
    a3c_loss = policy_loss + value_coeff * value_loss + entropy_coef * entropy_loss
    
    return a3c_loss, value_loss, policy_loss, entropy_loss
    
def print_avg(scores, p_i, tot_rew, lock, avg_ep, params, flag_finish, array_avgs):
    print('\n')
    with lock:
        scores.append([p_i, tot_rew])
        #print('scores', scores)
        all_found = 0
        #check if all process present
        for p_k in range(0, params['n_process']):
            ff = False
            for s_k in scores:
                if p_k == s_k[0] and ff==False:
                    all_found+=1
                    ff = True
                
        if all_found == params['n_process']:
            avg = 0
            for p_j in range(0, params['n_process']):
                idx = 0
                found = False
                for s_i in scores:
                    if p_j == s_i[0] and found==False:
                        avg += s_i[1]
                        found=True
                        scores.pop(idx)
                    idx+=1
                    
            with avg_ep.get_lock():
                avg_ep.value +=1
                print('\n')
                print('------------ AVG-------------')
                print(f"Ep: {avg_ep.value} | AVG: {avg/params['n_process']}")
                print('-----------------------------')
                array_avgs.append(avg/params['n_process'])
                
                if len(array_avgs)>100:
                    avg = np.mean(np.array(array_avgs[-100:]))
                    print('\n')
                    print('------------------------------')
                    print('AVG last 100 scores: ', avg)
                    print('------------------------------')
                    print('\n')
                    if avg >= params['mean_reward']:
                        flag_finish = True
                        print('------------------------')
                        print('GAME FINISHED')
                        print('------------------------')
                else:
                    flag_finish = False
        else:
            print('Not enough process completed to compute AVG...')
            flag_finish = False
        
        return flag_finish, array_avgs
