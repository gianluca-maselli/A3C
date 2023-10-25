import torch
import torch.nn.functional as F
from utils import *

def compute_log_prob_actions(logits):
    prob_v = F.softmax(logits, dim=-1)
    log_prob_v = F.log_softmax(logits, dim=-1)
    action_pd = torch.distributions.Categorical(probs=prob_v)
    action = action_pd.sample().detach()
    action_log_probs = log_prob_v.gather(1, action.unsqueeze(1)).squeeze()
    entropy = -(prob_v * log_prob_v).sum(dim=1)

    return action.numpy()[0], action_log_probs, entropy


def rollout(rollout_size, model, hx, cx, frame_queue, env, current_state, episode_length, max_steps, steps_array, values, log_probs, rewards, masks, entropies, actions_name, n_frames, device, tot_rew):
    
    for _ in range(rollout_size):
        episode_length +=1
        #print('current_state', current_state.shape)
        current_state = current_state.unsqueeze(0).permute(0,3,1,2).to(device)
        #print('current_state', current_state.shape)

        #compute logits, values and hidden and cell states from the current state
        logits, value, (hx,cx) = model((current_state,(hx,cx)))
        #print('logits', logits.shape)
        #print('values', values.shape)

        #get action
        action, action_log_probs, entropy = compute_log_prob_actions(logits)
        #print('action', type(action))
        #print('action_log_probs', action_log_probs)
        #print('entropy', entropy)

        #permorm step in the env
        next_frame, reward, done, _ = skip_frames(action,env,skip_frame=4)
        reward = max(min(reward, 1), -1)
        tot_rew +=reward
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)

        #condition of episode end
        if episode_length > max_steps:
            done = True
        
        values.append(value)
        log_probs.append(action_log_probs)
        rewards.append(reward)
        masks.append(done)
        entropies.append(entropy)

        current_state = next_state

        if done:
            #reset env
            #print('tot_reward: ', tot_rew)
            in_state_i = env.reset()
            frame_queue = initialize_queue(frame_queue, n_frames, in_state_i, env, actions_name)
            #stack the frames together
            input_frames = stack_frames(frame_queue)
            current_state = input_frames
            episode_length = 0
            break
    
    #bootstrapping
    next_value = torch.zeros(1, 1)
    if not done:
        #with torch.no_grad():
        final_state = current_state.unsqueeze(0).permute(0,3,1,2).to(device)
        _, f_value, _ = model((final_state,(hx,cx)))
            #in the case the game is not done we bootstrap
        next_value = f_value.detach()
    #add the last reward value to the array
    values.append(next_value)
    
    steps_array.append((rewards, masks, log_probs, values))

    return hx, cx, steps_array, entropies, episode_length, frame_queue, current_state, done, tot_rew

def GAE(steps, gamma, lambd, device, val_coeff, entropy_coef, entropies):
       
    #loop in reverse mode excluding the last element (i.e. next value)
    rewards, dones, log_probs , values = steps[0]
    
    advantages = torch.zeros(len(rewards),1)
    gaes = torch.zeros(len(rewards),1)
    
    R = values[-1]
    gae = 0.0
     
    for j in reversed(range(len(rewards))):
        R = rewards[j] + R * gamma
        advantages[j] = R - values[j]

        td_error = rewards[j] + gamma * values[j+1] - values[j]
        gae = gae * gamma * lambd  + td_error
        gaes[j] = gae
    
    return advantages, gaes
    


def ensure_shared_grads(local_model, shared_model):
    for param, shared_param in zip(local_model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.grad = param.grad #.clone()
    #print('Grads updating....')
    



def upgrade_parameters(advantages, gaes, steps, val_coeff, entropies, entropy_coef):
    
    
    #policy_loss = torch.zeros(advantages.shape[0],1)
    policy_loss = torch.zeros(1,1)
    #value_loss = torch.zeros(advantages.shape[0],1)
    value_loss = torch.zeros(1,1)
    #entropy_loss = torch.zeros(returns.shape[0],1)
    
    _, _, log_probs, _ = steps[0]
    
    for i in range(advantages.shape[0]):
        policy_loss = policy_loss - log_probs[i] * gaes[i].detach() - entropy_coef * entropies[i]
        #policy_loss[i] = -log_probs[i] * gaes[i].detach() + entropy_coef * entropies[i]
        #entropy_loss[i] = entropies[i]
        value_loss = value_loss + 0.5 * advantages[i].pow(2)
        #value_loss[i] = 0.5 * advantages[i].pow(2)
    #overall loss
    #policy_loss = policy_loss.sum()
    #value_loss = value_loss.sum()
    #entropy_loss = entropy_loss.mean()
    
    #policy_loss = (- torch.tensor(log_probs) * gaes.detach()).mean() 
    #value_loss = (torch.mul(advantages.pow(2), 0.5)).mean()
    #entropy_loss = torch.mul(torch.tensor(entropies), entropy_coef).mean()
    
    a3c_loss = policy_loss + val_coeff * value_loss
     
    '''
    for i in range(returns.shape[0]):
        policy_loss = policy_loss - log_probs[i] * gaes[i].detach() - entropy_coef * entropies[i]
        value_loss = value_loss + 0.5 * (returns[i]- values[i]).pow(2)
        #value_loss = value_loss + (returns[i]- values[i]).pow(2)
    '''
        

    return a3c_loss, value_loss, policy_loss


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    
    with global_ep.get_lock():
        global_ep.value += 1
    
    with global_ep_r.get_lock():
        #if global_ep_r.value == 0.:
        #    global_ep_r.value = ep_r
        #else:
            #global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
        global_ep_r.value  = ep_r
    
    #res_queue.put(global_ep_r.value)
    print(
        "Process: ", name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )
    
def print_avg(scores, p_i, tot_rew, lock, avg_ep, params, mean_reward, flag_finish, array_avgs):
    print('\n')
    with lock:
        scores.append([p_i, tot_rew])
        print('scores', scores)
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
                
                if avg/params['n_process'] >= mean_reward:
                    flag_finish = True
                    print('------------------------')
                    print('GAME FINISHED')
                    print('------------------------')
        else:
            print('Not enough process completed to compute AVG...')
            flag_finish = False
        
        return flag_finish, array_avgs
