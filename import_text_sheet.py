import numpy as np
from os.path import exists

def import_KW_txtfile(path, samplename, no_of_samples=3):
    """ Importieren der Messergebnisse der Kontaktwinkelmessung.

    Parameters
    ----------
    path : string
        path, where the sample-txt-files are stored
    samplename : string
        name of sample (as stored in DataFrame)
    no_of_samples : int (default = 3)
        number of samples of the measurement
    
    Returns
    -------
    KW_Wasser : numpy array (shape=(2,0))
        mean contact angle of water and its derivation
    KW_Diodmethan : numpy array (shape=(2,0))
        mean contact angle of Diodmethan and its derivation
    KW_Ethylglykol : numpy array (shape=(2,0))
        mean contact angle of Ethylglykol and its derivation
    OE_total : numpy array (shape=(2,0))
        mean total surface energy and its derivation
    OE_dispers : numpy array (shape=(2,0))
        mean dispersive surface energy and its derivation
    OE_polar : numpy array (shape=(2,0))
        mean polar surface energy and its derivation

    """

    iterate_samples = ["_Probe" + str(n+1) for n in range(no_of_samples)]

    f = path + samplename + iterate_samples[0] + ".txt"

    if exists(f):

        # get data
        KW_Wasser = []
        KW_Diodmethan = []
        KW_Ethylglykol = []
        OE_total = []
        OE_dispers = []
        OE_polar = []

        for i in iterate_samples:
            check = np.loadtxt(path + samplename + i + ".txt", dtype=str, encoding="cp1252", delimiter="\t", skiprows=7, max_rows=1)
            line = None
            if check[0] == 'Remarks':
                line = 1
            elif check[0] == 'Liquid':
                line = 0
            else:
                print("There is something unusual in file ", samplename, i)
            
            if line == 0 or line == 1:
                KW_Wasser_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, encoding="cp1252", delimiter="\t", skiprows=(9+line), max_rows=1)
                KW_Wasser.append([float(KW_Wasser_[7]), float(KW_Wasser_[8])])
                KW_Diodmethan_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, delimiter="\t", encoding="cp1252", skiprows=(10+line), max_rows=1)
                KW_Diodmethan.append([float(KW_Diodmethan_[7]), float(KW_Diodmethan_[8])])
                KW_Ethylglykol_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, delimiter="\t", encoding="cp1252", skiprows=(11+line), max_rows=1)
                KW_Ethylglykol.append([float(KW_Ethylglykol_[7]), float(KW_Ethylglykol_[8])])
                OE_total_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, delimiter="\t", encoding="cp1252", skiprows=(16+line), max_rows=1)
                OE_total.append(float(list(np.char.split(OE_total_[1]).flatten())[0][0]))
                OE_dispers_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, delimiter="\t", encoding="cp1252", skiprows=(17+line), max_rows=1)
                OE_dispers.append(float(list(np.char.split(OE_dispers_[1]).flatten())[0][0]))
                OE_polar_ = np.loadtxt(path + samplename + i + ".txt", dtype=str, delimiter="\t", encoding="cp1252", skiprows=(18+line), max_rows=1)
                OE_polar.append(float(list(np.char.split(OE_polar_[1]).flatten())[0][0]))

        # print(KW_Wasser)
        # print(OE_polar)

        KW_Wasser = np.asarray(KW_Wasser)
        KW_Diodmethan = np.asarray(KW_Diodmethan)
        KW_Ethylglykol = np.asarray(KW_Ethylglykol)
        KW_Wasser = [np.mean(KW_Wasser.T[0]), np.std(KW_Wasser.T[0])]
        KW_Diodmethan = [np.mean(KW_Diodmethan.T[0]), np.std(KW_Diodmethan.T[1])]
        KW_Ethylglykol = [np.mean(KW_Ethylglykol.T[0]), np.std(KW_Ethylglykol.T[1])]
        OE_total = [np.mean(OE_total), np.std(OE_total)]
        OE_dispers = [np.mean(OE_dispers), np.std(OE_dispers)]
        OE_polar = [np.mean(OE_polar), np.std(OE_polar)]

        # print(KW_Wasser)
        # print(KW_Diodmethan)
        # print(KW_Ethylglykol)
        # print(OE_dispers)
        # print(OE_polar)

        return KW_Wasser, KW_Diodmethan, KW_Ethylglykol, OE_total, OE_dispers, OE_polar
    
    else:
        return False, False, False, False, False, False


if __name__ == '__main__':

    path = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/Kontaktwinkelmessungen/AD/"
    # samplename = "Altech_PA66ECO_1000-561_Referenz"
    # samplename = "Altech_PA66ECO_1000-561_Vfg5"
    # samplename = "DYMID_R_PA66GF15_Referenz"
    samplename = "AltechPA66A_1000-109_Referenz"
    no_of_KW_samples = 3

    KW_Wasser, KW_Diodmethan, KW_Ethylglykol, OE_total, OE_dispers, OE_polar = import_KW_txtfile(path, samplename, no_of_samples=no_of_KW_samples)

    # print(KW_Wasser)
    # print(OE_dispers)