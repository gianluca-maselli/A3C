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

## Usage


