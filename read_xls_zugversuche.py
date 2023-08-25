import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


def read_xls_zugversuch_statistik(path, filename):
    file_ = path + filename + ".xls"
    df = pd.read_excel(file_, sheet_name="Statistik", header=[0,1])
    df.columns = df.columns.map(" in ".join)
    Fmax = df["Fmax in N"]
    eF = df["e-F max 2 in mm"]
    tmax = df["tmax in MPa"]
    S0 = df["S0 in mm²"]
    # 0 = Mittelwert
    # 1 = Standardabweichung absolut
    # 2 = Standardabweichung relativ in %

    # print(df)
    # print(df["Fmax in N"].iloc[0])
    # print(Fmax[0])

    return Fmax, eF, tmax, S0


def read_xls_zugversuch_verlaeufe(path, filename, n=5):
    file_ = path + filename + ".xls"
    proben = [pd.read_excel(file_, sheet_name=str(i+1), header=[1,2]) for i in range(n)]
    for p in proben:
        p.columns = p.columns.map(" in ".join)
    
    return proben


def plot_zugversuch_verlaeufe(proben):
    
    plt.figure()
    cmap = matplotlib.cm.get_cmap('plasma')
    colors = np.asarray([cmap(0.01), cmap(0.2), cmap(0.5), cmap(0.7), cmap(0.9)])
    for i, p in enumerate(proben):
        plt.scatter(p["Dehnung in mm"], p["Standardkraft in N"], marker=".", s=0.5, c=colors[i])
    
    plt.show()



if __name__ == "__main__":
    path = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/Zugversuch/14.08.2023/"
    filename = "HDPE_rezykliert_oT_ND2449h_20m-min_1"
    # Fmax, eF, tmax, S0 = read_xls_zugversuch_statistik(path, filename)
    proben = read_xls_zugversuch_verlaeufe(path, filename)
    plot_zugversuch_verlaeufe(proben)
