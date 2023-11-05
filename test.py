import os
import imageio
from collections import deque
import torch
from utils import *
import torch.nn.functional as F
import gym
import numpy as np

def test(p_i, shared_model, params, max_steps, layers_):
    
    seed = params['seed']
    torch.manual_seed(seed + p_i)
    np.random.seed(seed + p_i)
    
    env = gym.make(params['env_name'])
    actions_name = env.unwrapped.get_action_meanings()
    
    print('------ TEST PHASE -------')
    
    shared_model.eval()

    queue = deque(maxlen=4)
    #reset env
    in_state_i = env.reset(seed=(seed + p_i))
    #initialize a queue for each env, preprocess each frame and obtain a vecotr of 84,84,4
    frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
    #stack the frames together
    input_frames = stack_frames(frame_queue)
    current_state = input_frames

    done = True
    episode_length = 0
    tot_reward = 0
    render = []
    fps = 30
    g_i = 0
    tot_games = 3
    
    #start game
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            hx = torch.zeros(1, layers_['lstm_dim'])
            cx = torch.zeros(1, layers_['lstm_dim'])
        else:
            hx = hx.detach()
            cx = cx.detach()

        current_state = current_state.unsqueeze(0).permute(0,3,1,2)
        with torch.no_grad():
            #compute logits, values and hidden and cell states from the current state
            logits, _ , (hx, cx)  = shared_model((current_state,(hx, cx)))
        #get the most probable action
        probs = F.softmax(logits, dim=-1)
        action = probs.max(1, keepdim=True)[1].numpy()
        #perform step in the env
        next_frame, reward, done, _ = skip_frames(action[0, 0],env,skip_frame=4)
        render.append(next_frame)
        tot_reward+=reward

        #stack frames
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)


        if done or (episode_length >=max_steps):
            g_i +=1
            print('-------------------------------------------')
            print(f'Test Game: {g_i}, Score: {tot_reward}, episode_length: {episode_length}')
            print('-------------------------------------------')
            name = './replay_test'+str(g_i)+'.gif'
            imageio.mimsave(name, [np.array(img_i) for img_i in render], fps = fps)
            if g_i == tot_games:
                break
            tot_reward = 0
            episode_length = 0
            #reset env
            in_state_i = env.reset()
            frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
            next_state = stack_frames(frame_queue) 
            render = []
        
        current_state = next_state
