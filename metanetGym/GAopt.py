import pandas as pd
import copy
from metanetGym.cost import *

# 遗传算法参数
population_size = 50  # 种群大小
chromosome_length = 6  # 染色体长度
mutation_rate = 0.1  # 变异率
generations = 100  # 迭代次数

TIME_LENGTH_STEP = 5 / 60  # 每个控制步长的时长，小时


# 生成初始种群
def generate_population():
    population = []
    for _ in range(population_size):
        chromosome = [random.randint(0, 4) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population


# 计算适应度
def fitness(chromosome, state, id_step):
    # 这里以列表中元素的和作为适应度
    return 1 / cost.cal_cost(chromosome, copy.deepcopy(state), id_step)


# 选择操作（轮盘赌选择）
def selection(population, state, id_step):
    fitness_values = [fitness(chromosome, state, id_step) for chromosome in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness_value / total_fitness for fitness_value in fitness_values]

    selected_population = []
    for _ in range(population_size):
        selected_chromosome = random.choices(population, probabilities)[0]
        selected_population.append(selected_chromosome)
    return selected_population


# 交叉操作（单点交叉）
def crossover(parent1, parent2):
    crossover_point = random.randint(1, chromosome_length - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# 变异操作
def mutate(chromosome):
    mutated_chromosome = chromosome.copy()
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = random.randint(0, 4)
    return mutated_chromosome


# 遗传算法主函数
def genetic_algorithm(state, id_step):
    population = generate_population()
    best_fitness = -float('inf')
    best_chromosome = None

    for generation in range(generations):
        population = selection(population, state, id_step)

        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            child1, child2 = crossover(parent1, parent2)
            mutated_child1 = mutate(child1)
            mutated_child2 = mutate(child2)
            next_generation.extend([mutated_child1, mutated_child2])

        population = next_generation

        # 计算最佳适应度和染色体
        for chromosome in population:
            chromosome_fitness = fitness(chromosome, state, id_step)
            if chromosome_fitness > best_fitness:
                best_fitness = chromosome_fitness
                best_chromosome = chromosome

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Chromosome = {best_chromosome}")

    env_cost = MetanetEnv()
    env_cost.reset()
    env_cost.set_state(state, id_step)
    total_reward = 0
    action_list = best_chromosome
    for action in action_list:
        for _ in range(int(TIME_LENGTH_STEP / env_cost.metanet.DELTA_T)):
            observation, reward, done, _ = env_cost.step(action)
            total_reward += reward
    # self.env_cost.render()
    # return total_reward
    env_cost.render()

    return best_chromosome


# 运行遗传算法
# best_solution = genetic_algorithm()
# print("Best Solution:", best_solution)

cost = CostMetanet()

env = MetanetEnv()
observation = env.reset()

df_flow = pd.DataFrame(columns=['1', '2', '3'])
df_v = pd.DataFrame(columns=['1', '2', '3'])
df_queue_length_onramp = pd.DataFrame(columns=['onramp'])
df_demand = pd.DataFrame(columns=['origin', 'onramp'])
df_density = pd.DataFrame(columns=['1', '2', '3'])

reward_list = []
action_list = []

done = False

action_best = [4, 4]
while not done:
    # action = env.action_space.sample()  # 示例：随机选择动作

    if env.metanet.step_id % 30 == 0 and env.metanet.step_id > 0:
        action_best = genetic_algorithm(copy.deepcopy((env.state)), env.metanet.step_id)  # 示例：随机选择动作
        print('print(action_best) ', action_best)

    observation, reward, done, _ = env.step(action_best[0])
    reward_list.append(reward)
    action_list.append(action_best)
    # print(env.metanet.state_density)
    df_density.loc[len(df_density)] = env.metanet.state_density
    df_flow.loc[len(df_flow)] = [num * env.metanet.FLOW_MAX for num in observation[:3]]
    df_v.loc[len(df_v)] = [num * env.metanet.V_MAX for num in observation[3:6]]
    df_queue_length_onramp.loc[len(df_queue_length_onramp)] = observation[-1] * env.metanet.QUEUE_LENGTH_ONRAMP_MAX
    df_demand.loc[len(df_demand)] = [env.metanet.input_demand_origin, env.metanet.input_demand_onramp]
    env.render()
