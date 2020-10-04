# Report

## Implementation
The environment was solved using a deep reinforcement learning agent. The implementation can be found in the `collab_and_comp`-directory.
`agent.py` contains the rl-agent and `model.py` contains the neural networks used as the estimators. Some of the code was taken
from the Udacity ddpg-pendulum exercise and adapted to the needs of this problem.

### Learning algorithm
[MADDPG](https://arxiv.org/abs/1706.02275) which is an actor-critic approach was used as the learning algorithm for the agent.
This algorithm is derived from [DDPG](https://arxiv.org/abs/1509.02971), but manages to solve tasks with multiple agents.
As an off-policy algorithm MADDPG utilizes four neural networks: a local actor, a target actor, a local critic and a target critic
Each training step the experience (state, action, action_other_agent, reward, next state, next_state_other_agent) the two agents gained was stored.
Then every training step the agent learned from a random sample from the stored experience. The actor tries to estimate the
optimal policy by using the estimated state-action values from the critic while critic tries to estimate the optimal q-value function
and learns by using a normal q-learning approach. Using this approach one gains the benefits of value based and policy based
methods at the same time. By giving the critic access to the action of the other player the learning process gets stabilized
without requiring to give the additional information to the actor which is the only network required for acting after the
agent was trained successfully.

### Hyperparameters
The following hyperparameters were used:
* replay buffer size: 1e5
* max timesteps: 10000 (all episodes get shutdown after 10000 timesteps)
* minibatch size: 128
* discount factor: 0.99
* tau (soft update for target networks factor): 1e-3
* learning rate: 1e-4 (actor) and 1e-3 (critic)
* update interval (how often to learn): 1
* beta start (factor for the noise added to the actions selected by the actor): 1.0
* beta decay factor: 0.995
* min beta: 0.01

### Neural networks
The actor model is a simple feedforward network:
* Batch normalization
* Input layer: 24 (input) neurons (the state size)
* 1st hidden layer: 128 neurons (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 2 neurons (1 for each action) (tanh)

The critic model:
* Batch normalization
* Input layer: 24 (input) neurons (the state size)
* 1st hidden layer: 132 neurons (action with 2 * action_size 2 added) (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 1 neuron

## Results
The agent was able to solve the environment after 467 episodes achieving an average maximum score of 0.51 over the last 100 episodes
of the training process.

The average maximum scores of the 2 agents during the training process:
![scores](https://user-images.githubusercontent.com/9535190/78504705-33104180-776f-11ea-99ea-861ef0bb299a.png)

## Possible future improvements
The algorithm could be improved in many ways. For example one could implement some DQN improvements such as Prioritized Experience Replays
which would improve the learning effect gained from the saved experience. Also one could try to optimize the various hyperparameters
or change the neural network architectures.
