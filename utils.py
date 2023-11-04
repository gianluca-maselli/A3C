import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import deque
import matplotlib.image as mpimg


#this function is used to capture easy temporal relationship by stacking n frames together
#for more complex temporal relationships 3D convolutions may be required
#following the DeepMind approach we stacked together the last 4 frames to produce a single image
#Each channel in the stacked image corresponds to a single frame at a different point in time, 
#so the CNN can learn to extract features that represent the changes that occur between frames.
def frame_preprocessing(frame):
    if frame.size == 210 * 160 * 3:
        img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
        img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    else:
        assert False, "Unknown resolution."
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    x_t = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = x_t[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

#Sample initial states by taking random number of no-ops on reset. No-op is assumed to be action 0.
def Noop(env, actions_name, noop_max):
    _ = env.reset()
    assert actions_name[0] == 'NOOP'
    noops = np.random.randint(1, noop_max + 1)
    action = 0
    init_state = None
    for _ in range(noops):
        init_state, _, done, _, _ = env.step(action)
        if done:
            init_state = env.reset()
    return init_state

def initialize_queue(queue, n_frames, init_frame,env, actions_name): 
    queue.clear()
    init_frame = Noop(env, actions_name, noop_max=30)
    for i in range(n_frames):
        queue.append(frame_preprocessing(init_frame))
    return queue

#since we are using Pong without skipping frame integrated, we built a function to skip 4 frames at each step.
#in addition max-pooling is performed by stacking every k frames together and taking the element-wise maximum.
#then we return the resulting frame as the current observation.
def skip_frames(action,env, skip_frame=4):
    skipped_frame = deque(maxlen=2)
    skipped_frame.clear()
    total_reward = 0.0
    done = None
    for _ in range(skip_frame):
        n_state, reward, done, _, info = env.step(action)
        skipped_frame.append(n_state)
        total_reward += reward
        if done:
            break
    max_frame = np.max(np.stack(skipped_frame), axis=0)

    return max_frame, total_reward, done, info

#stack frames together to form a single one
def stack_frames(stacked_frames):
    #concatenate the frames 
    frames_stack = np.concatenate(stacked_frames, axis=-1)
    frames_stack = frames_stack.astype(np.float32) / 255.0
    return torch.tensor(frames_stack, dtype=torch.float32)

def running_mean(x,N=100):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

def plot_avg_scores(array_avg, title):
    scores_avg = running_mean(np.array(array_avg), N=100)
    plt.title(title)
    plt.ylabel('Scores')
    plt.xlabel('Episode')
    plt.plot(scores_avg)
    plt.yticks(np.arange(np.min(scores_avg), np.max(scores_avg)+1, 3.0))
    plt.savefig('./plot_avg_scores.png')
