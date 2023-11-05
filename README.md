# A3C
The repository contains the PyTorch implementation of the **Asynchronous Advantage Actor Critic (A3C)** introduced in ["Asynchronous Methods for Deep Reinforcement Learning"](https://arxiv.org/pdf/1602.01783v1.pdf).

A3C is based on the following key principles :
  * **Asynchronous Training**: The algorithm relies on multiple agents with their own networks and a copy of the environment. Each of them acts independently and Asynchronously with its environment, continually learning with each interaction. This approach speeds up the training process by gathering more experience.
  * **Global Network**: A3C uses a global model that is accessed and updated by all agents simultaneously. The idea is that as each agent gains more knowledge, it contributes to the total knowledge of the global model.
  * **Actor-Critic Architecture**: A3C employs an Actor-Critic architecture consisting of:
    *   **Actor**: The task of this policy network is to select actions based on the current state. The choice is driven by the goal of maximizing the expected rewards.
    *   **Critic**: A value network whose task is to evaluate the quality of the state-action pairs. Its output is the value estimate that helps the actor to learn which actions are more favorable.
  * **Global and Local Steps**: A total number of interactions across all agents is defined to be the maximum number of global steps. By contrast within each agent training process, a number of local steps (e.g. 5 or 20) are used to periodically update the global model with the gradients computed by the local Model. Moreover, periodically, the local models are updated with the global model's weights.

## Results and Comparison

The following results were obtained by carrying out experiments with both **4** and **8** parallel agents. In the original implementation, the authors used a total of 16 parallel agents. Note that to achieve the results presented below, we employed a moving average over the final 100 rewards as the stopping criterion in the two experimental environments:
  * **PongNoFrameskip-v4**: 18
  * **BreakoutNoFrameskip-v4** : 60

PongNoFrameskip-v4  | BreakoutNoFrameskip-v4
:-------------------------:|:-------------------------:
![](https://github.com/gianluca-maselli/A3C/blob/main/gifs/replay_test_pong.gif) | ![](https://github.com/gianluca-maselli/A3C/blob/main/gifs/replay_test_break.gif)
Running AVG reward with 8 parallel agents  | Running AVG reward with 8 parallel agents
![](https://github.com/gianluca-maselli/A3C/blob/main/plots/plot_avg_scores_pong_8pr.png)  | ![](https://github.com/gianluca-maselli/A3C/blob/main/plots/plot_avg_scores_break_8_pr.png)
Running AVG reward with 4 parallel agents  | Running AVG reward with 4 parallel agents
![](https://github.com/gianluca-maselli/A3C/blob/main/plots/plot_avg_scores_pong_4pr.png)  | ![](https://github.com/gianluca-maselli/A3C/blob/main/plots/plot_avg_scores_break_4pr.png)

## Hyperparameters (Default)
Hyperparameter  | Value
:-------------------------:|:-------------------------:
learning rate | 0.0001
gamma | 0.99
critic loss coefficient | 0.5
entropy coefficient | 0.001
rollout size | 20
max grad norm | 40

## Requirements
Library  | Version
:-------------------------:|:-------------------------:
pytorch |  1.12.1
gym | 0.26.2
numpy | 1.23.4
opencv-python | 4.6.0
matplotlib | 3.6.1

### Hardware Requirements
All runs with both 8 and 4 parallel agents were performed on [paperspace](https://www.paperspace.com/).

## Usage
 ```
usage: main.py [-h] [--lr LR] [--gamma GAMMA] [--entropy-coef ENTROPY_COEF]
               [--value-loss-coef VALUE_LOSS_COEF]
               [--max-grad-norm MAX_GRAD_NORM] [--seed SEED] [--rs RS]
               [--n-workers N_WORKERS] [--ep-length EP_LENGTH]
               [--env-name ENV_NAME] [--opt OPT] [--use-trained USE_TRAINED]

optional arguments:
  -h, --help                         show this help message and exit
  --lr LR                            learning rate (default: 0.0001)
  --gamma GAMMA                      discount factor for rewards (default: 0.99)
  --entropy-coef ENTROPY_COEF        entropy term coefficient (default: 0.01)
  --value-loss-coef VALUE_LOSS_COEF  value loss coefficient (default: 0.5)
  --max-grad-norm MAX_GRAD_NORM      value to clip the grads (default: 40)
  --seed SEED                        random seed (default: 1)
  --rs RS                            rollout size before updating (default: 20)
  --n-workers N_WORKERS              how many training processes to use (default: os cpus)
  --ep-length EP_LENGTH              maximum episode length (default: 4e10)
  --env-name ENV_NAME                environment to train on (default: PongNoFrameskip-v4)
  --opt OPT                          optimizer to use (default: Adam)
  --use-trained USE_TRAINED          training A3C from scratch (default: False)

```
By running: 
 ```
python main.py
```
The program will be executed with the default parameters, considering a number of workers equal to the available system CPUs. 
In addition, the A3C will be trained from scratch. To use a pre-trained model it is sufficient to switch the  flag   ```--use-trained=TRUE```.

## Acknowledgement
The code is inspired by the following implementation:
1) [A3C-ACER-PyTorch](https://github.com/alirezakazemipour/A3C-ACER-PyTorch) by [alirezakazemipour](https://github.com/alirezakazemipour)
2) [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c) by [ikostrikov](https://github.com/ikostrikov)
