


action_list = []
reward_list = []
queue_list = []
episodes = 1
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        # action = 0
        state, reward, done, _ = env.step(action)
        # env.render()
        total_reward += reward

        action_list.append(action)
        reward_list.append(reward)
        queue_list.append(state[-1])

    print(f"Test Episode: {episode+1}, Reward: {total_reward}")
    print(f"Test Episode: {episode + 1}, TTS: {0.1 * sum(action_list) - total_reward}")

plt.figure()
plt.plot(action_list)
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Action on {}'.format(env_name))

plt.figure()
plt.plot(reward_list)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward on {}'.format(env_name))

plt.figure()
plt.plot(queue_list)
plt.xlabel('Step')
plt.ylabel('Queue')
plt.title('Queue on {}'.format(env_name))

plt.show()