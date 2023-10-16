import os
import imageio
from collections import deque
import torch
from utils import *
from model import ActorCritic
import time
import torch.nn.functional as F
from IPython.display import display

def test(p_i, shared_model, counter, env, max_steps, layers_, actions_name, lock, device):
    seed = 1
    torch.manual_seed(seed + p_i)
    env.seed(seed + p_i)
    
    print('------ TEST PHASE -------')
    #create instance of the model
    model = ActorCritic(input_shape=layers_['n_frames'], layer1=layers_['hidden_dim1'], kernel_size1=layers_['kernel_size1'], stride1=layers_['stride1'], layer2=layers_['hidden_dim2'],
                        kernel_size2=layers_['kernel_size2'], stride2=layers_['stride2'], layer3=layers_['hidden_dim3'], kernel_size3=layers_['kernel_size3'], stride3=layers_['stride3'],
                        fc1_dim=layers_['fc1'], out_actor_dim=layers_['out_actor_dim'], out_critic_dim=layers_['out_critic_dim']) #.to(device)

    # Verify that model is indeed an instance of nn.Module
    
    model.eval()

    queue = deque(maxlen=4)
    #reset env
    in_state_i = env.reset()
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
    mean_reward = 18.0
    actions = deque(maxlen=1000) #to prevent the agent from stucking
    flag_render = False
    #counter_dones = 0
    
    #start game
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        current_state = current_state.unsqueeze(0).permute(0,3,1,2).to(device)
        #print('current_state', current_state.shape)
        with torch.no_grad():
            #compute logits, values and hidden and cell states from the current state
            logits, value  = model(current_state)
        #get the most probable action
        probs = F.softmax(logits, dim=-1)
        action = probs.max(1, keepdim=True)[1].numpy()
        #print('test action', action)
        #perform step in the env
        next_frame, reward, done, _ = skip_frames(action[0, 0],env,skip_frame=4)
        render.append(next_frame)
        tot_reward+=reward
        
        if tot_reward >= mean_reward:
            print('\n')
            print('---- GAME FINISHED ----')
            break
        
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        #stack frames
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue)


        if done or (episode_length >=max_steps):
            print('probs: ', probs)
            if counter.value % 10 == 0:
                if flag_render == True:
                    os.remove("./replay_test.gif")
                imageio.mimsave('./replay_test.gif', [np.array(img_i) for img_i in render], fps = fps)
                flag_render = True
                #display.display(display.Image("/content/replay_test.gif"))
                

            counter.value += 1

            print('\n')
            print('-------------------------------------------')
            print(f'Test Game: {counter.value}, Score: {tot_reward}, episode_length: {episode_length}')
            print('-------------------------------------------')
            print('\n')
            tot_reward = 0
            episode_length = 0
            actions.clear()
            #reset env
            in_state_i = env.reset()
            frame_queue = initialize_queue(queue, layers_['n_frames'], in_state_i, env, actions_name)
            next_state = stack_frames(frame_queue)
            time.sleep(60)       
        
        current_state = next_state