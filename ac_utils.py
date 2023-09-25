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

    return action.numpy()[0],action_log_probs, entropy


def rollout(rollout_size, model, hx, cx, frame_queue, env, current_state, episode_length, max_steps, steps_array, values, log_probs, rewards, masks, actions, entropies, actions_name, n_frames, device, tot_rew):
    
    for i in range(rollout_size):
        episode_length +=1
        #print('current_state', current_state.shape)
        current_state = current_state.unsqueeze(0).permute(0,3,1,2).to(device)
        #print('current_state', current_state.shape)

        #compute logits, values and hidden and cell states from the current state
        logits, value, (hx,cx) = model((current_state, (hx,cx)))
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
        #log_probs.append(logits)
        rewards.append(reward)
        masks.append(done)
        actions.append(action)
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
    next_value = 0
    if not done:
        with torch.no_grad():
            final_state = next_state.unsqueeze(0).permute(0,3,1,2).to(device)
            _, f_value, _ = model((final_state, (hx,cx)))
            #in the case the game is not done we bootstrap
            next_value = f_value.detach()
    #add the last reward value to the array
    values.append(next_value)
    steps_array.append((rewards, masks, log_probs, values))

    return steps_array, entropies, entropy, episode_length, frame_queue, current_state, done, tot_rew

def GAE(steps, gamma, lambd, device):
       
    '''
    for j in reversed(range(len(rewards))):
        R = gamma * R + rewards[j]
        advantage = R - values[j]
        advs[j] = advantage
        #generalzied advantage estimation
        td_error = rewards[j] + gamma * values[j + 1] * (1-masks[j]) - values[j]
        #gae = gae * gamma * lambd * (1-masks[j]) + td_error
        gae = gae * gamma * lambd * (1-masks[j]) + td_error
        gaes[j] = gae
    '''
     #loop in reverse mode excluding the last element (i.e. next value)
    rewards, dones, _, values = steps[0]
    
    returns = torch.zeros(len(rewards),1, device=device)
    gaes = torch.zeros(len(rewards),1, device=device)
    
    R = values[-1]
    gae = 0.0
    
    for j in reversed(range(len(rewards))):
        R = rewards[j] + R * gamma * (1-dones[j])
        returns[j] = R
        td_error = rewards[j] + gamma * values[j+1] * (1-dones[j]) - values[j]
        gae = gae * gamma * lambd *  (1-dones[j]) + td_error
        gaes[j] = gae
    
    #print('returns size: ', returns.shape) 
    #print('gae size: ', gaes.shape) 
    
    return returns, gaes

    '''
    for j in reversed(range(len(values)-1)):
        #compute expected returns
        R = rewards[j] + R * gamma * (1-masks[j])
        returns[j] = R
        #extract next value recurrently
        next_value = values[j + 1]
        vals[j] =  values[j]
        #compute td error
        td_error = rewards[j] + gamma * next_value * (1-masks[j]) - values[j]
        #GAE
        gae = gae * gamma * lambd * (1-masks[j]) + td_error
        gaes[j] = gae
    
    assert len(returns) == len(values)-1
    
    '''
    #return returns, gaes, vals



def ensure_shared_grads(local_model, shared_model):
    for param, shared_param in zip(local_model.parameters(),shared_model.parameters()):
        #if shared_param.grad is not None:
        #    return
        shared_param._grad = param.grad
    #print('Grads updating....')
    



def upgrade_parameters(returns, gaes, steps, val_coeff, entropies, entropy_coef):
    
    
    policy_loss = 0
    value_loss = 0
    '''
    #critic loss
    #value_loss = advantages.pow(2).mean()
    #value_loss = advantages.pow(2).sum()
    value_loss = F.mse_loss(values, returns)
    #policy loss
    a_log_probs = torch.stack(action_log_probs).unsqueeze(1)

    #policy_loss = (-a_log_probs * gaes.detach()).mean() - entropy_coef * torch.tensor(entropies).mean()
    policy_loss = (-a_log_probs * gaes.detach()).mean() + entropy_coef * torch.tensor(entropies).mean()
    #overall loss
    a3c_loss = policy_loss + val_coeff * value_loss
    '''
    _, _, log_probs, values = steps[0]
    
    for i in range(returns.shape[0]):
        policy_loss = policy_loss - log_probs[i] * gaes[i].detach() - entropy_coef * entropies[i]
        #value_loss = value_loss + 0.5 * (returns[i]- values[i]).pow(2)
        value_loss = value_loss + (returns[i]- values[i]).pow(2)
        
    a3c_loss = policy_loss + val_coeff * value_loss

    return a3c_loss, value_loss, policy_loss


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    
    with global_ep.get_lock():
        global_ep.value += 1
    
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            #global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
            global_ep_r.value  = ep_r
    
    res_queue.put(global_ep_r.value)
    print(
        "Process: ", name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )