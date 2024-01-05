import random

def encode_variable(value):
    # 将变量值编码为二进制字符串
    binary = bin(value)[2:].zfill(3)  # 转换为二进制并填充到3位
    return binary

def decode_variable(binary):
    # 将二进制字符串解码为变量值
    value = int(binary, 2)
    return value

def mutate(chromosome):
    # 随机选择一个基因进行变异
    gene_index = random.randint(0, len(chromosome) - 1)
    gene = chromosome[gene_index]

    # 随机选择一个新的取值
    new_value = random.randint(0, 4)
    new_gene = encode_variable(new_value)

    # 替换基因
    mutated_chromosome = chromosome[:gene_index] + new_gene + chromosome[gene_index+3:]

    return mutated_chromosome

# 示例用法
chromosome = "000001001010011"  # 染色体序列
variable = decode_variable(chromosome[6:9])  # 解码基因序列为变量值
print("变量值:", variable)

print("变异前的染色体:", chromosome)
mutated_chromosome = mutate(chromosome)  # 变异操作
print("变异后的染色体:", mutated_chromosome)