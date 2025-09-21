#Some SARSA implementations
import gymnasium as gym
import numpy as np


def update_q_table(state,action,reward,next_state,next_action):
    old_value=Q[state,action]
    next_value=Q[next_state,next_action]
    Q[state,action]= (1-alpha)*old_value + alpha*(reward + gamma*next_value)

def get_policy(env):
    policy={state: np.argmax(Q[state]) for state in range(env.observation_space.n)}
    return policy

env=gym.make("FrozenLake",is_slippery=False)

num_states=env.observation_space.n
num_actions=env.action_space.n

Q=np.zeros((num_states,num_actions))

alpha=0.1
gamma=1
num_episodes=1000

for episode in range(num_episodes):
    state,info=env.reset()
    action=env.action_space.sample()
    terminated=False
    while not terminated:
        next_state,reward,terminated,truncated,info=env.step(action)
        next_action=env.action_space.sample()
        update_q_table(state,action,reward,next_state,next_action)
        state,action=next_state,next_action

policy=get_policy(env)
print(policy)
