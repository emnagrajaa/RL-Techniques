import gymnasium as gym

def compute_state_value(state, policy, env, terminal_state,gamma):
    if state == terminal_state:
        return 0
    action = policy[state]
    _,next_state, reward, _, _ = env.step(action)
    return reward + gamma * compute_state_value(next_state, policy, env, terminal_state,gamma)

def compute_q_value(state, action, policy, env, terminal_state,gamma):
    if state == terminal_state:
        return None
    _,next_state, reward, _ = env.unwrapped.P[state][action][0]
    return reward + gamma * compute_state_value(next_state, policy, env, terminal_state,gamma)

def policy_evaluation(policy, env, terminal_state, gamma):
    V={state : compute_state_value(state, policy, env, terminal_state,gamma) for state in range(env.observation_space.n)}
    return V

def policy_improvement(V, env, terminal_state,gamma):
    improved_policy={s: 0 for s in range(env.observation_space.n - 1)}
    Q={(state, action): compute_q_value(state, action, improved_policy, env, terminal_state,gamma) for state in range(env.observation_space.n) for action in range(env.action_space.n)}
    for state in range(env.observation_space.n - 1):
        max_action = max(range(env.action_space.n),key= lambda action: Q[(state,action)])
        improved_policy[state] = max_action
    return improved_policy

def policy_iteration():
    policy={0:1, 1:2, 2:1 , 3:1, 4:3, 5:1, 6:2, 7:3}
    while True:
        V=policy_evaluation(policy)
        improved_policy=policy_improvement(V)
        if improved_policy == policy:
            break
        policy = improved_policy
    return policy, V

def get_max_action_and_value(state,V, env, terminal_state,gamma):
    Q_values=[compute_q_value(state,action,V)for action in range(env.action_space.n)]
    max_action = max(range(env.action_space.n), key=lambda action: Q_values[action])
    return max_action, Q_values[max_action]
