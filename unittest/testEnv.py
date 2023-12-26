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

    done = False
    while not done:
        # action = env.action_space.sample()  # 示例：随机选择动作
        action = 4  # 示例：随机选择动作
        observation, reward, done, _ = env.step(action)
        df_flow.loc[len(df_flow)] = observation[:3]
        df_v.loc[len(df_v)] = observation[3:6]
        df_queue_length_onramp.loc[len(df_queue_length_onramp)] = observation[-1]
        df_demand.loc[len(df_demand)] = [env.metanet.input_demand_origin, env.metanet.input_demand_onramp]
        env.render()


    # plt.plot(df_density['1'], label='1')
    # plt.plot(df_density['2'], label='2')
    # plt.plot(df_density['3'], label='3')
    # plt.plot(df_density['4'], label='4')
    # plt.legend()
    plt.figure()
    sns.heatmap(df_flow.transpose())
    plt.figure()
    sns.heatmap(df_v.transpose())
    plt.figure()
    plt.plot(df_queue_length_onramp)
    plt.figure()
    plt.plot(df_demand['origin'])
    plt.figure()
    plt.plot(df_demand['onramp'])


    plt.show()