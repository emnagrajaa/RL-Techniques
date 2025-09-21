import gymnasium as gym
import numpy as np

def generate_episode(env):
    episode=[]
    state,_=env.reset()
    terminated=False
    while not terminated:
        action=env.action_space.sample()
        next_state,reward,terminated,truncated,info=env.step(action)
        episode.append((state,action,reward))
        state=next_state
    return episode

def first_visit_mc(num_episodes,env):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    returns_sum=np.zeros((env.observation_space.n,env.action_space.n))
    returns_count=np.zeros((env.observation_space.n,env.action_space.n))
    for i in range(num_episodes):
        episode=generate_episode(env)
        visited_states_actions=set()
        for j, (state,action,reward) in enumerate(episode):
            if (state, action) not in visited_states:
                returns_sum[state,action] += sum(x[2] for x in episode[j:])
                returns_count[state,action] += 1
                visited_states_actions.add((state,action))
    nonzero_counts= returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]
    return Q

def every_visit_mc(num_episodes,env):
    Q=np.zeros((env.observation_space.n,env.action_space.n))
    returns_sum=np.zeros((env.observation_space.n,env.action_space.n))
    returns_count=np.zeros((env.observation_space.n,env.action_space.n))
    for i in range(num_episodes):
        episode=generate_episode(env)
        for j, (state,action,reward) in enumerate(episode):
            returns_sum[state,action] += sum(x[2] for x in episode[j:])
            returns_count[state,action] += 1
    nonzero_counts= returns_count != 0
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]
    return Q

def get_policy(env):
    policy={state: np.argmax(Q[state]) for state in range(env.observation_space.n)}
    return policy
