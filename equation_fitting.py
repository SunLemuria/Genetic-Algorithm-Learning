# http://wiki.swarma.net/index.php/%E4%BD%BF%E7%94%A8pythony%E5%AE%9E%E7%8E%B0%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95

from random import random, randint, choice
from copy import deepcopy
import numpy as np


# 树的基本结构
class FWrapper:
    def __init__(self, function, childcount, name):
        self.function = function  # 函数定义
        self.childcount = childcount  # 函数参数个数
        self.name = name  # 函数名字


class Node:  # 一个节点
    def __init__(self, fw, children):
        self.function = fw.function
        self.name = fw.name
        self.children = children

    def evaluate(self, inp):
        # 接受子节点的输入，把FWrapper定义的函数运用在子节点上，得到输出结果
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)

    def display(self, indent=0):
        # 打印节点以下的树的形式
        print((' ' * indent) + self.name)
        for c in self.children:
            c.display(indent + 1)


class ParamNode:
    # 参数节点，返回输入的参数idx
    def __init__(self, idx):
        self.idx = idx

    def evaluate(self, inp):
        return inp[self.idx]

    def display(self, indent=0):
        print('%sp%d' % (' ' * indent, self.idx))


class ConstNode:
    # 常数节点，不管输入是什么，总是返回一个常数
    def __init__(self, v):
        self.v = v

    def evaluate(self, inp):
        return self.v

    def display(self, indent=0):
        print('%s%d' % (' ' * indent, self.v))


# 定义基本的运算
addw = FWrapper(lambda l: l[0] + l[1], 2, 'add')
subw = FWrapper(lambda l: l[0] - l[1], 2, 'subtract')
mulw = FWrapper(lambda l: l[0] * l[1], 2, 'multiply')


def iffunc(l):
    if l[0] > 0:
        return l[1]
    else:
        return l[2]


ifw = FWrapper(iffunc, 3, 'if')


def isgreater(l):
    if l[0] > l[1]:
        return 1
    else:
        return 0


gtw = FWrapper(isgreater, 2, 'isgreater')

flist = [addw, mulw, ifw, gtw, subw]


# # 构造示例树
# def exampletree():
#     return Node(
#         ifw,
#         [
#             Node(gtw, [ParamNode(0), ConstNode(3)]),
#             Node(addw, [ParamNode(1), ConstNode(5)]),
#             Node(subw, [ParamNode(1), ConstNode(2)]),
#         ]
#     )
#
#
# exampletree = exampletree()
# exampletree.evaluate([2, 3])
# exampletree.evaluate([5, 3])
# exampletree.display()


# 构造随机树，作为演化的初始值
def makerandomtree(pc, maxdepth=4, fpr=0.5, ppr=0.6):
    if random() < fpr and maxdepth > 0:
        f = choice(flist)
        children = [makerandomtree(pc, maxdepth - 1, fpr, ppr)
                    for i in range(f.childcount)]
        return Node(f, children)

    elif random() < ppr:
        return ParamNode(randint(0, pc - 1))

    else:
        return ConstNode(randint(0, 10))


random1 = makerandomtree(2)
random1.evaluate([7, 1])
random1.evaluate([2, 4])

random2 = makerandomtree(2)
random2.evaluate([5, 3])
random2.evaluate([5, 20])
random1.display()
random2.display()


# 定义变异和交配函数对树结构进行操作
def mutate(t, pc, probchange=0.1):
    if random() < probchange:
        return makerandomtree(pc)
    else:
        result = deepcopy(t)
        if isinstance(t, Node):
            result.children = [mutate(c, pc, probchange) for c in t.children]
        return result


def crossover(t1, t2, probswap=0.7, top=1):
    if random() < probswap and not top:
        return deepcopy(t2)
    else:
        result = deepcopy(t1)
        if isinstance(t1, Node) and isinstance(t2, Node):
            result.children = [crossover(c, choice(t2.children), probswap, 0) for c in t1.children]
    return result


random2.display()
muttree = mutate(random2, 2)
muttree.display()
cross = crossover(random1, random2)
cross.display()


# 使用事先给定的函数制造一个数据
# 准备相应的衡量一棵树拟合数据的结果的函数
def hiddenfunction(x, y):
    return x ** 2 + 2 * y + 3 * x + 5


def buildhissdeset():
    rows = []
    for i in range(200):
        x = randint(0, 10)
        y = randint(0, 10)
        rows.append([x, y, hiddenfunction(x, y)])
    return rows


def scorefunction(tree, s):
    dif = 0
    for data in s:
        v = tree.evaluate([data[0], data[1]])
        dif += abs(v - data[2])
    return dif


def getrankfunction(dataset):
    def rankfunction(population):
        scores = [(scorefunction(t, dataset), t) for t in population]
        scores.sort()
        return scores

    return rankfunction


hiddenset = buildhissdeset()
scorefunction(random2, hiddenset)
scorefunction(random1, hiddenset)


# 准备就绪，开始迭代
def evolve(pc, popsize, rankfunction, maxgen=500, mutationrate=0.1, breedingrate=0.4, pexp=0.7, pnew=0.05):
    # returns a random number, tending towards lower numbers.the lower pexp is more lower numbers you will get
    def selectindex():
        return int(np.log(random()) / np.log(pexp))

    # create a random initial population
    population = [makerandomtree(pc) for i in range(popsize)]
    for i in range(maxgen):
        scores = rankfunction(population)
        print(scores[0][0])
        if scores[0][0] == 0:
            break
        # the two best always make it
        newpop = [scores[0][1], scores[1][1]]
        # build the next generation
        while len(newpop) < popsize:
            if random() > pnew:
                newpop.append(mutate(
                    crossover(scores[selectindex()][1],
                              scores[selectindex()][1],
                              probswap=breedingrate),
                    pc, probchange=mutationrate
                ))
            else:
                # add a random node to mix things up
                newpop.append(makerandomtree(pc))
        population = newpop
    scores[0][1].display()
    return scores[0][1]


rf = getrankfunction(buildhissdeset())
final = evolve(2, 500, rf, mutationrate=0.2, breedingrate=0.1, pexp=0.7, pnew=0.1)
