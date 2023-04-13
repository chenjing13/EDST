import pandas as pd
import numpy as np
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def read_data(name):
    address1 = "20220210测试新细胞系数据/{}/drugdrug_extract.csv".format(name)
    address2 = "20220210测试新细胞系数据/{}/drugfeature_sig_extract.csv".format(name)

    with open(address1) as f:
        drugdrug = np.loadtxt(f, str, delimiter=",", skiprows=1, usecols=(3, 4, 9, 10))
    with open(address2) as f:
        drugfeature = np.loadtxt(f, str, delimiter=",")
        drugfeature = drugfeature[:, 1:]  # remove gene id

    data = np.zeros([1, (drugfeature.shape[0]-1)*2+1], dtype=float)
    for i in range(0, drugdrug.shape[0]):
        drug1ID = drugdrug[i][0]
        drug2ID = drugdrug[i][1]
        # S=drugdrug[i][2]
        L = drugdrug[i][3]
        drug1data = drugfeature[1:, list(drugfeature[0]).index(drug1ID)].astype(np.float32)
        drug2data = drugfeature[1:, list(drugfeature[0]).index(drug2ID)].astype(np.float32)

        # case1 drug1+drug2
        if L == "NA" or L == "Additive":
            L = 0
        if L == "antagonism" or L == "Antagonism":
            L = 1
        if L == "synergy" or L == "Synergy":
            L = 2
        union = np.hstack((drug1data, drug2data, L)).reshape(1, (drugfeature.shape[0]-1)*2+1)
        data = np.vstack((data, union))

    data = data[1:, :]

    print(data.shape)
    print("=================data loading completed================")
    return data
