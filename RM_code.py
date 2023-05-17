import numpy as np
import random
#from prettytable import PrettyTable
import pandas as pd
import torch

class Marix:
    def __init__(self):
        #self.G = []  # 满秩二元矩阵
        self.G_R: int  # 满秩二元矩阵的秩
        self.E = []  # 由满秩拆分的秩相同的矩阵
        self.I = []  # 由满秩拆分的单位矩阵
        self.X = []  # 存放随机生成的二进制矩阵

    def X_create(self, r): # 生成v_1,v_2,...,v_m
        self.X = np.array([[0 for i in range(2 ** r)] for i in range(r)])

        for i in range(2 ** r):
            now = "{:b}".format(i).zfill(r)
            self.X[:, i] = np.array(list(now))
        return self.X.tolist()

    def X_create2(self, X, r): #生成r(2,m)的基：1,v_1,...,v_m,v_1v_2,...,v_{m-1}v_m
        X_len = len(X)
        for i in range(X_len - 1):
            Count = X_len
            for number in range(Count):
                if number <= i:
                    continue
                X.append(self.martix_multiply(X[i], X[number]))
            Count -= 1
        return np.array([[1 for i in range(2 ** r)]] + X)

    def matrix_rank(self, matrix):
        # 返回矩阵的秩
        return np.linalg.matrix_rank(matrix)

    def matrix_same(self, matrix1, matrix2):
        # 矩阵的比较，相同为一
        return np.array(np.array(matrix1) == np.array(matrix2)).astype(int)

    def matrix_notsame(self, matrix1, matrix2):
        # 矩阵的比较，相同为零
        return np.array(np.array(matrix1) != np.array(matrix2)).astype(int)

    def martix_multiply(self, matrix1, matrix2):
        # 矩阵相乘
        return np.multiply(matrix1, matrix2)

    def martix_matmul(self,matrix1,matrix2):
        return np.matul(matrix1, matrix2)

    def martix_solve(self,matrix):
        #return np.where(matrix>1,0,matrix) #np.where(condition,x,y) 满足条件(condition)，输出x，不满足输出y。
        return np.where(matrix%2==0,0,1)


def G_Create(m, r):
    while True:
        G = np.zeros((m, m), int)
        rows = []
        for i in range(r):
            cur_row = np.random.randint(0, m, 1)
            while cur_row in rows:
                cur_row = np.random.randint(0, m, 1)
            rows.append(cur_row)
            G_cur_row = np.random.randint(0, 2, m)
            G[cur_row, :] = G_cur_row
        for i in range(m):
            G[i, i] = 1
        E = (G - np.identity(m)).astype(int)
        r_cur = np.linalg.matrix_rank(E)

        # Check if rank matches with desired rank
        if r_cur == r:
            break
    return G


if __name__ == "__main__":
    def run(m, i):
        G = G_Create(m, i + 1)
        #print("G:\n", G)
        b1 = marix.X_create2(d.copy(), m)
        #print(d)
        #print("二进制拓展:\n", b1)
        b = np.array(d.copy())
        #print('b:',b)
        c = b1.copy()[m + 1:]
        #print('b1:',b1)
        #print("c:\n", c)
        #print("G乘X:\n", marix.martix_solve(np.matmul(G, b))) # b就是X：v_1,...,v_m
        # numpy.matmul(a, b, out=None) 两个numpy数组的矩阵相乘
        G = marix.martix_solve(np.matmul(G, b.copy())) # 此时G=G*X
        #print('G:',G)
        #now = marix.martix_compare(G, b, c)
        now = marix.X_create2(G.tolist(), m)
        #print("现在生成的:\n", now)
        #print("之前的:\n", b1)
        compare = marix.matrix_notsame(b1, now)
        #print("Id+G = ",compare)
        #print("k' = ",marix.matrix_rank(compare))
        #print("不同：\n", compare, "\nk'：\n", marix.matrix_rank(compare))
        comparetorch = torch.tensor(compare)
        comparetorch = comparetorch.to(torch.double)
        res = int(torch.linalg.matrix_rank(comparetorch))
        return res
    
    #file_name = input("Enter the output file name: ")
    #out_file = open(file_name, 'w+')

    while True:
        m = int(input("m = "))
        r = int(input("r(E) = "))
        val = 4 * r * (m - r) + 1
        #print("val: ",val)
        iteration = int(input("iteration = "))
        #out_file.write('m = {0}, r(E) = {1}, 4r(m-r)+1 = {2}, iteration = {3}\n'.format(m, r, val, iteration))
        #r = 1
        if r > m:
            print("error")
            continue
        break

    marix = Marix()
    # 生成二进制
    d = marix.X_create(m)
    #print("二进制数据:\n", d)
    print("k'",run(m,r-1))
    #k_rank_max = int(1 + m + m*(m-1)/2)
    #k_new = [0 for i in range(k_rank_max)]
    #val = 4 * r * (m - r) + 1
    #print(val)
    '''for i in range(iteration):
      kn = run(m,r-1)
      k_new[kn-1] += 1'''
    
    '''for i in range(k_rank_max):
      print("k' = ",i+1," : ","times = ", k_new[i])'''
      #out_file.write('{0}\t'.format(i))
      #out_file.write('{0}\t'.format(k_new[j]))
    '''k_rank = []
    k_rank = [0 for i in range(m)]
    for i in range(r):
      k_rank[i] = run(i+1,m)  
    print(k_rank)'''
    #out_file.close()
    #data = pd.DataFrame(k_new).T
    #print(data)
    