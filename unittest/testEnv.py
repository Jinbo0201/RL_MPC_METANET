import matplotlib.pyplot as plt
from metanetGym.metanetEnv import MetanetEnv
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    env = MetanetEnv()
    observation = env.reset()

    df_flow = pd.DataFrame(columns=['1', '2', '3'])
    df_v = pd.DataFrame(columns=['1', '2', '3'])
    df_queue_length_onramp = pd.DataFrame(columns=['onramp'])
    df_demand = pd.DataFrame(columns=['origin', 'onramp'])
    df_density = pd.DataFrame(columns=['1', '2', '3', '4'])

    reward_list = []
    action_list = []

    done = False
    while not done:
        # action = env.action_space.sample()  # 示例：随机选择动作
        action = 4  # 示例：随机选择动作
        observation, reward, done, _ = env.step(action)
        reward_list.append(reward)
        action_list.append(action)
        # print(env.metanet.state_density)
        df_density.loc[len(df_density)] = env.metanet.state_density
        df_flow.loc[len(df_flow)] = [num * env.metanet.FLOW_MAX for num in observation[:3]]
        df_v.loc[len(df_v)] = [num * env.metanet.V_MAX for num in observation[3:6]]
        df_queue_length_onramp.loc[len(df_queue_length_onramp)] = observation[-1] * env.metanet.QUEUE_LENGTH_ONRAMP_MAX
        df_demand.loc[len(df_demand)] = [env.metanet.input_demand_origin, env.metanet.input_demand_onramp]
        env.render()

    print(f"Reward: {sum(reward_list)}")
    print(f"TTS: {0.1 * sum(action_list) - sum(reward_list)}")

    # plt.plot(df_density['1'], label='1')
    # plt.plot(df_density['2'], label='2')
    # plt.plot(df_density['3'], label='3')
    # plt.plot(df_density['4'], label='4')
    # plt.legend()
    plt.figure()
    sns.heatmap(df_density.transpose())
    plt.title('df_density')
    plt.figure()
    sns.heatmap(df_flow.transpose())
    plt.title('df_flow')
    plt.figure()
    sns.heatmap(df_v.transpose())
    plt.title('df_v')
    plt.figure()
    plt.plot(df_queue_length_onramp)
    plt.title('df_queue_length_onramp')
    plt.figure()
    plt.plot(df_demand['origin'])
    plt.title('demand_origin')
    plt.figure()
    plt.plot(df_demand['onramp'])
    plt.title('demand_onramp')
    plt.figure()
    plt.plot(reward_list)
    plt.title('reward')

    plt.show()
