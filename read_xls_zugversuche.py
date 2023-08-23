import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import xlsxwriter as xls


def read_xls_zugversuch(path, filename):
    
    file = path + filename + ".xls"
    excel = pd.read_excel
    print(excel)




if __name__ == "__main__":
    path = "/Users/toffiefee/Desktop/"
    filename = "HDPE_rezykliert_oT_ND2449h_20m-min_2"
    read_xls_zugversuch(path, filename)