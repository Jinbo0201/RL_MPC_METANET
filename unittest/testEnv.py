
from env.metanetEnv import MetanetEnv

if __name__ == "__main__":
    env = MetanetEnv()
    observation = env.reset()
    print(observation)

    done = False
    while not done:
        action = env.action_space.sample()  # 示例：随机选择动作
        observation, reward, done, _ = env.step(action)
        print(action, observation, reward, done)