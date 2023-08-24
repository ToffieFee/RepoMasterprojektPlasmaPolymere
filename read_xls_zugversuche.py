import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_xls_zugversuch(path, filename):
    file_ = path + filename + ".xls"
    df = pd.read_excel(file_, sheet_name="Statistik", header=[0,1])
    df.columns = df.columns.map(" in ".join)
    Fmax = df["Fmax in N"]
    eRm = df["e-Rm in mm"]
    tmax = df["tmax in MPa"]
    S0 = df["S0 in mm²"]
    # 0 = Mittelwert
    # 1 = Standardabweichung absolut
    # 2 = Standardabweichung relativ in %

    # print(df["Fmax in N"].iloc[0])
    # print(Fmax[0])




if __name__ == "__main__":
    path = "/Users/toffiefee/Desktop/"
    filename = "HDPE_rezykliert_oT_ND2449h_20m-min_2"
    read_xls_zugversuch(path, filename)