import numpy as np
import pandas as pd
import math
from os.path import exists
import openpyxl
import matplotlib
import matplotlib.pyplot as plt
import operator
from read_xls_zugversuche import read_xls_zugversuch_statistik, read_xls_zugversuch_verlaeufe, plot_zugversuch_verlaeufe
from functools import reduce
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 12})
#Locale settings
import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "de_DE")
# Tell matplotlib to use the locale we set above
plt.rcParams['axes.formatter.use_locale'] = True


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
    # print("*****F*****:", f)

    if exists(f):

        # get data
        KW_Wasser = []
        KW_Diodmethan = []
        KW_Ethylglykol = []
        OE_total = []
        OE_dispers = []
        OE_polar = []

        for i in iterate_samples:

            # if exists(i) == False:
            #     break

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

        # if exists(iterate_samples[1]):
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

def load_KW_data(df, path, no_of_KW_samples=3):

    df['KW_Wasser_mean'] = ''
    df['KW_Wasser_std'] = ''
    df['KW_Diodmethan_mean'] = ''
    df['KW_Diodmethan_std'] = ''
    df['KW_Ethylglykol_mean'] = ''
    df['KW_Ethylglykol_std'] = ''
    df['OE_total_mean'] = ''
    df['OE_total_std'] = ''
    df['OE_dispers_mean'] = ''
    df['OE_dispers_std'] = ''
    df['OE_polar_mean'] = ''
    df['OE_polar_std'] = ''

    for samplename, _ in df.iterrows():
        
        # print("Current sample: ", samplename)
        KW_Wasser, KW_Diodmethan, KW_Ethylglykol, OE_total, OE_dispers, OE_polar = import_KW_txtfile(path, samplename, no_of_samples=no_of_KW_samples)

        if KW_Wasser == False:
            print("Couldn't find a KW-resultfile for sample ", samplename)
            df.at[samplename, 'KW_Wasser_mean'] = np.NaN
            df.at[samplename, 'KW_Diodmethan_mean'] = np.NaN
            df.at[samplename, 'KW_Ethylglykol_mean'] = np.NaN
            df.at[samplename, 'OE_total_mean'] = np.NaN
            df.at[samplename, 'OE_dispers_mean'] = np.NaN
            df.at[samplename, 'OE_polar_mean'] = np.NaN
            df.at[samplename, 'KW_Wasser_std'] = np.NaN
            df.at[samplename, 'KW_Diodmethan_std'] = np.NaN
            df.at[samplename, 'KW_Ethylglykol_std'] = np.NaN
            df.at[samplename, 'OE_total_std'] = np.NaN
            df.at[samplename, 'OE_dispers_std'] = np.NaN
            df.at[samplename, 'OE_polar_std'] = np.NaN
        
        else:
            df.at[samplename, 'KW_Wasser_mean'] = KW_Wasser[0]
            df.at[samplename, 'KW_Diodmethan_mean'] = KW_Diodmethan[0]
            df.at[samplename, 'KW_Ethylglykol_mean'] = KW_Ethylglykol[0]
            df.at[samplename, 'OE_total_mean'] = OE_total[0]
            df.at[samplename, 'OE_dispers_mean'] = OE_dispers[0]
            df.at[samplename, 'OE_polar_mean'] = OE_polar[0]
            df.at[samplename, 'KW_Wasser_std'] = KW_Wasser[1]
            df.at[samplename, 'KW_Diodmethan_std'] = KW_Diodmethan[1]
            df.at[samplename, 'KW_Ethylglykol_std'] = KW_Ethylglykol[1]
            df.at[samplename, 'OE_total_std'] = OE_total[1]
            df.at[samplename, 'OE_dispers_std'] = OE_dispers[1]
            df.at[samplename, 'OE_polar_std'] = OE_polar[1]
        
    df = df.astype({"KW_Wasser_mean": float, "KW_Wasser_std": float, "KW_Diodmethan_mean": float, "KW_Diodmethan_std": float, "KW_Ethylglykol_mean": float, "KW_Ethylglykol_std": float, "OE_total_mean": float, "OE_total_std": float,"OE_dispers_mean": float, "OE_dispers_std": float, "OE_polar_mean": float, "OE_polar_std": float})

    # delete all non-KW-value rows
    OE_vals_where = df['OE_polar_mean'].notna()
    indices_OE = np.where(OE_vals_where)[0] # indices
    df = df.iloc[indices_OE]

    print("\n")
    
    return df

def read_spreadsheet_tsv(path, filename):
    
    df = pd.read_csv(path + filename, sep="\t", skiprows=3)
    df.drop(['Datum', 'Material_alt', 'Stichprobenanzahl', 'Soll-Spannung [V]', 'Ist-Spannung [V]'], axis=1, inplace=True)
    df = df.iloc[:, :-19]

    df = df.set_index('Probe')

    # ND-Plasmaparameter müssen noch angepasst werden

    return df

def read_spreadsheet_xls(path, filename):
    file_ = path + filename
    df = pd.read_excel(file_ + ".xlsx", engine='openpyxl', skiprows=3)
    df = df.set_index('Probe')
    df = df[df.index.notna()]
    df = df[df.index != "KW"]
    df = df[df.index != "Kleben"]
    
    # print(df.index)

    return df


def reduce_df(df, one_mat, material, one_condition, condition, one_gf, gfamount, one_type, type_, one_time, time, del_columns, columns_of_interest):

    df_list = []

    # MATERIAL
    # one_mat = input("Plot one material? Enter 'y' or 'n' : ")

    if one_mat == "y":
        # materialtypes = df['Material'].unique()
        # material = input(f'Which material is of interest? Enter one of following: {materialtypes} ')

        material_where = df['Material'] == material
        indices_mat = np.where(material_where)[0] # indices
        if len(indices_mat) == 0:
            indices_mat = [0]
        df = df.iloc[indices_mat]

        label = material
    else:

        materialtypes = df['Material'].unique()
        label_ = []

        for mat in materialtypes:
            material_where = df['Material'] == mat
            indices_mat = np.where(material_where)[0] # indices
            if len(indices_mat) == 0:
                indices_mat = [0]
            df_ = df.iloc[indices_mat]
            df_list.append(df_)
            label.append(mat)


    # ZUSTAND
    # one_condition = input("Plot one condition? Enter 'y' or 'n' : ")

    if one_condition == "y":
        # conditiontypes = df['Zustand'].unique()
        # condition = input(f'Which condition is of interest? Enter one of following: {conditiontypes} ')

        if type(label) != list:
            condition_where = df['Zustand'] == condition
            indices_cond = np.where(condition_where)[0] # indices
            if len(indices_cond) == 0:
                indices_cond = [0]
            df = df.iloc[indices_cond]
            label = label + "_" + condition

        else:
            df_list_ = []
            for d in df_list:
                condition_where = d['Zustand'] == condition
                indices_cond = np.where(condition_where)[0] # indices
                if len(indices_cond) == 0:
                    indices_cond = [0]
                df_list_.append(d.iloc[indices_cond])
            df_list = df_list_
            label = [l + "_" + condition for l in label]

    else:

        label_ = []
        
        if type(label) != list:
            conditiontypes = df['Zustand'].unique()
            for cond in conditiontypes:
                condition_where = df['Zustand'] == cond
                indices_cond = np.where(condition_where)[0] # indices
                if len(indices_cond) == 0:
                    indices_cond = [0]
                df_ = df.iloc[indices_cond]
                df_list.append(df_)
                label_.append(cond)
            label = [label + "_" + l for l in label_]
        else:
            df_list_ = []
            for i, d in enumerate(df_list):
                conditiontypes = d['Zustand'].unique()
                lab = []
                for cond in conditiontypes:
                    condition_where = d['Zustand'] == cond
                    indices_cond = np.where(condition_where)[0] # indices
                    if len(indices_cond) == 0:
                        indices_cond = [0]
                    df_list_.append(d.iloc[indices_cond])
                    lab.append(cond)
                label_.append([label[i] + "_" + l for l in lab])
            df_list = df_list_
            label = reduce(operator.add, label_)


    # Glasfaseranteil
    # one_gf = input("Plot samples with the same amount of glassfibre? Enter 'y' or 'n' : ")

    if one_gf == "y":
        # gfamount_types = df['Glasfaseranteil'].unique()
        # gfamount = input(f'Which glassfibre content is of interest? Please enter one of following: {gfamount_types} ')

        if type(label) != list:
            gfamount_where = df['Glasfaseranteil'] == gfamount
            indices_gf = np.where(gfamount_where)[0] # indices
            if len(indices_gf) == 0:
                indices_gf = [0]
            df = df.iloc[indices_gf]
            label = label + "_gf" + str(int(gfamount))

        else:
            df_list_ = []
            for d in df_list:
                gfamount_where = d['Glasfaseranteil'] == gfamount
                indices_gf = np.where(gfamount_where)[0] # indices
                if len(indices_gf) == 0:
                    indices_gf = [0]
                df_list_.append(d.iloc[indices_gf])
            df_list = df_list_
            label = [l + "_gf" + str(int(gfamount)) for l in label]

    else:

        label_ = []
        
        if type(label) != list:
            gfamount_types = df['Glasfaseranteil'].unique()
            for amount in gfamount_types:
                gfamount_where = df['Glasfaseranteil'] == amount
                indices_gf = np.where(gfamount_where)[0] # indices
                if len(indices_gf) == 0:
                    indices_gf = [0]
                df_ = df.iloc[indices_gf]
                df_list.append(df_)
                label_.append(amount)
            label = [label + "_gf" + str(int(l)) for l in label_]

        else:
            df_list_ = []
            for i, d in enumerate(df_list):
                gfamount_types = d['Glasfaseranteil'].unique()
                lab = []
                for amount in gfamount_types:
                    gfamount_where = d['Glasfaseranteil'] == amount
                    indices_gf = np.where(gfamount_where)[0] # indices
                    if len(indices_gf) == 0:
                        indices_gf = [0]
                    df_list_.append(d.iloc[indices_gf])
                    lab.append(amount)
                label_.append([label[i] + "_gf" + str(int(l)) for l in lab])
            df_list = df_list_
            label = reduce(operator.add, label_)


    # Typ
    # one_type = input("Plot samples with same type? Enter 'y' or 'n' : ")

    if one_type == "y":
        # types_ = df['Typ'].unique()
        # type_ = input(f'Which type content is of interest? Please enter one of following: {types_} ')

        if type(label) != list:
            type_where = df['Typ'] == type_
            indices_type = np.where(type_where)[0] # indices
            if len(indices_type) == 0:
                indices_type = [0]
            df = df.iloc[indices_type]
            label = label + "_" + type_
        else:
            df_list_ = []
            for d in df_list:
                type_where = d['Typ'] == type_
                indices_type = np.where(type_where)[0] # indices
                if len(indices_type) == 0:
                    indices_type = [0]
                df_list_.append(d.iloc[indices_type])
            df_list = df_list_
            label = [l + "_" + type_ for l in label]

    else:

        label_ = []
        
        if type(label) != list:
            types_ = df['Typ'].unique()
            for t in types_:
                type_where = df['Typ'] == t
                indices_type = np.where(type_where)[0] # indices
                if len(indices_type) == 0:
                    indices_type = [0]
                df_ = df.iloc[indices_type]
                df_list.append(df_)
                label_.append(t)
            label = [label + "_" + l for l in label_]
        else:
            df_list_ = []
            for i, d in enumerate(df_list):
                types_ = d['Typ'].unique()
                lab = []
                for t in types_:
                    type_where = d['Typ'] == t
                    indices_type = np.where(type_where)[0] # indices
                    if len(indices_type) == 0:
                        indices_type = [0]
                    df_list_.append(d.iloc[indices_type])
                    lab.append(t)
                label_.append([label[i] + "_" + l for l in lab])
            # print(df_list_)
            df_list = df_list_
            label = reduce(operator.add, label_)


    # Zeit
    # one_time = input("Plot samples with same timedelay? Enter 'y' or 'n' : ")
    # print(df_list)
    if one_time == "y":
        # times = df['Zeit'].unique()
        # time = input(f'Which timedelay is of interest? Please enter one of following: {types_} ')

        if type(label) != list:
            time_where = df['Zeit'] == time
            indices_time = np.where(time_where)[0] # indices
            if len(indices_time) == 0:
                indices_time = [0]
            df = df.iloc[indices_time]
            label = label + "_t" + str(int(time)) + "min"
        else:
            df_list_ = []
            for d in df_list:
                time_where = d['Zeit'] == time
                indices_time = np.where(time_where)[0] # indices
                if len(indices_time) == 0:
                    indices_time = [0]
                df_list_.append(d.iloc[indices_time])
            df_list = df_list_
            label = [l + "_t" + str(int(time)) + "min" for l in label]
            # print(df_list)
    else:

        label_ = []
        
        if type(label) != list:
            times = df['Zeit'].unique()
            for tim in times:
                time_where = df['Zeit'] == tim
                indices_time = np.where(time_where)[0] # indices
                if len(indices_time) == 0:
                    indices_time = [0]
                df_ = df.iloc[indices_time]
                df_list.append(df_)
                label_.append(tim)
            label = [label + "_t" + str(int(l)) + "min" for l in label_]
        else:
            df_list_ = []
            for i, d in enumerate(df_list):
                times = d['Zeit'].unique()
                lab = []
                for tim in times:
                    time_where = d['Zeit'] == tim
                    indices_time = np.where(time_where)[0] # indices
                    if len(indices_time) == 0:
                        indices_time = [0]
                    df_list_.append(d.iloc[indices_time])
                    lab.append(tim)
                label_.append([label[i] + "_t" + str(int(l)) + "min" for l in lab])
            df_list = df_list_
            label = reduce(operator.add, label_)


    # print(df_list)

    if del_columns == "y":

        if type(df) != list:
            df = df[columns_of_interest]
        
        else:
            for i, d in enumerate(df_list):
                df_list[i] = d[columns_of_interest]

    # print(df_list)

    # for i, d in enumerate(df_list):
    #     print("label:", label[i])
    #     print(d[['Verfahrgeschwindigkeit [m/min]', 'Glasfaseranteil', 'Typ', 'Zeit']])

    if len(df_list) == 0:
        return df, label
    else:
        return df_list, label



if __name__ == '__main__':

    path_spreadsheet = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/"
    filename_spreadsheet = "Proben_Uebersichts_Tabelle-Dokumentation"

    path_KW_txtfiles = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/Kontaktwinkelmessungen/AD/"
    no_of_KW_samples = 3

    path_Zug_xlsfiles = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/Zugversuch/"

    one_mat = 'n'       # 'y' or 'n'
    material = 'PA66'

    one_condition = 'n' # 'y' or 'n'
    condition = 'rezykliert'

    one_gf = 'n'        # 'y' or 'n'
    gfamount = 0.0

    one_type = 'n'      # 'y' or 'n'
    type_ = 'Dymid'

    one_time = 'n'      # 'y' or 'n'
    time = 5.0

    del_columns = 'n'   # 'y' or 'n'
    columns_of_interest = ['Material', 'Glasfaseranteil', 'Typ', 'Verfahrgeschwindigkeit [m/min]', 'Zeit', 'KW_Wasser_mean', 'KW_Wasser_std', 'KW_Diodmethan_mean', 'KW_Diodmethan_std', 'KW_Ethylglykol_mean', 'KW_Ethylglykol_std', 'OE_total_mean', 'OE_total_std', 'OE_dispers_mean', 'OE_dispers_std', 'OE_polar_mean', 'OE_polar_std']

    # plotting settings
    x = 'Verfahrgeschwindigkeit [m/min]'
    y = 'OE_polar_mean' #'OE_total_mean'
    yerr = 'OE_polar_std'

    save_plot_path = "/Users/toffiefee/Documents/Uni_Bremen/Masterprojekt_2.0/Ergebnisse/Abbildungen/"
    save_plot_name = "" #"PA66_t5_polar"

    figsize = [8,4] # [x, y]


##################################################################################################

    df = read_spreadsheet_xls(path_spreadsheet, filename_spreadsheet)

    # df = load_KW_data(df, path_KW_txtfiles, no_of_KW_samples=no_of_KW_samples)

    # datatypes = df.dtypes
    # print(datatypes)

    # df, label = reduce_df(df, one_mat, material, one_condition, condition, one_gf, gfamount, one_type, type_, one_time, time, del_columns, columns_of_interest)

    # x = input(f'Enter X : {df.columns} \n')
    # y = input(f'Enter Y : {df.columns} \n')

    # print(df)


    # cmap = matplotlib.cm.get_cmap('turbo_r')
    # plt.figure(figsize=figsize)


    # if type(df) != list:

    #     # print(df[['Verfahrgeschwindigkeit [m/min]', 'Glasfaseranteil', 'Typ', 'Zeit']])

    #     color = cmap(0.3)

    #     plt.scatter(df[x], df[y], label=label, marker='x')
    #     if yerr != '' or yerr != None:
    #         plt.errorbar(df[x], df[y], yerr=df[yerr], xerr=None, fmt='none', capsize=3.0, elinewidth=0.5, ecolor=color)
    #     plt.xlabel(str(x))
    #     plt.ylabel(str(y))
    #     plt.legend()
    #     plt.show()

    # else:

    #     amount_colors = len(df)
    #     colors = [cmap(c) for c in np.arange(0.05, 0.95, 1/amount_colors)]

    #     # print(df)
    #     for i, d in enumerate(df):

    #         # print(d[['Verfahrgeschwindigkeit [m/min]', 'Glasfaseranteil', 'Typ', 'Zeit']])
    #         print(d['Verfahrgeschwindigkeit [m/min]'])
    #         if d['Verfahrgeschwindigkeit [m/min]'] == 0.0:
    #             plt.hline(d[y],0,1)
    #         else:
    #             plt.scatter(d[x], d[y], label=label[i], marker='x', color=colors[i])
    #             if yerr != '' or yerr != None:
    #                 plt.errorbar(d[x], d[y], yerr=d[yerr], xerr=None, fmt='none', capsize=3.0, elinewidth=0.5, ecolor=colors[i])


    #     plt.xlabel(str(x))
    #     plt.ylabel(str(y))
    #     plt.legend(fontsize='small', loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    

    # plt.tight_layout()
    # # if save_plot_path != "" or save_plot_path != None and save_plot_name != "" or save_plot_name != None:
    # #     plt.savefig(save_plot_path + save_plot_name + ".png", dpi=300)
    # plt.show()