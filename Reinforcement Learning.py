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