#Some Q-learning implementations
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(state,action,new_state):
    old_value=Q[state,action]
    next_value=np.max(Q[new_state])
    Q[state,action]= (1-alpha)*old_value + alpha*(reward + gamma*next_value)

def get_policy(env):
    policy={state: np.argmax(Q[state]) for state in range(env.observation_space.n)}
    return policy

env=gym.make("FrozenLake",is_slippery=False)

num_episodes=1000
alpha=0.1
gamma=1

num_states,num_actions=env.observation_space.n, env.action_space.n
Q=np.zeros((num_states,num_actions))

reward_per_random_episode=[]
for episode in range(num_episodes):
    state,info=env.reset()
    terminated=False
    episode_reward=0
    while not terminated:
        action=env.action_space.sample()
        new_state,reward,terminated,truncated,info=env.step(action)
        update_q_table(state,action,new_state)
        episode_reward+=reward
        state=new_state
    reward_per_random_episode.append(episode_reward)

reward_per_learned_episode=[]
policy=get_policy(env)
for episode in range(num_episodes):
    state,info=env.reset()
    terminated=False
    episode_reward=0
    while not terminated:
        action=policy[state]
        new_state,reward,terminated,truncated,info=env.step(action)
        episode_reward+=reward
        state=new_state
    reward_per_learned_episode.append(episode_reward)

avg_random_reward=np.mean(reward_per_random_episode)
avg_learned_reward=np.mean(reward_per_learned_episode)

plt.bar(['Rnandom Policy','Learned Policy'],[avg_random_reward,avg_learned_reward], color=['blue','orange'])
plt.title('Average Reward per Episode')
plt.ylabel('Average Reward')
plt.show()