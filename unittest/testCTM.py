from ctmGym.ctmEnv import *
import pandas as pd

if __name__ == '__main__':

    numEachGame = 100  # 每次游戏的步数
    env = CTMEnv(numEachGame)

    demand_file_path = '../resources/demand_CTM.csv'
    demandDF = pd.read_csv(demand_file_path)
    env.setDemand(demandDF)

    rAll = 0
    tttALL = 0
    twtALL = 0

    for j in range(numEachGame):

        a = 0

        s1, r, d = env.step(a)

        ttt, twt = env.getResult()

        tttALL += ttt
        twtALL += twt
        rAll += r  # rAll累加当前的收获。

        if d == True:  # 如果已经到达最终状态，就跳出for循环。(开始下一次迭代)
            break


    print("sum reward is %f, sum TTT is %f, sum TWT is %f" %(rAll, tttALL, twtALL))

    env.ctm.show()