state = env.reset()

while not done:

    action = agent.select_action(state)

    next_state, reward, done, info = env.step(action)

    agent.store_transition(
        state,
        action,
        reward,
        next_state,
        done
    )

    agent.learn()

    state = next_state