import collections
import pandas as pd
import glob
import os
import fileinput as fi
import numpy as np
import matplotlib.pyplot as plt

# Changing max size for printing pandas df.
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 20000)

def fetch_p2d(fp, set_target):
    """
        Read Visual3D P2D export (timenormalized timeseries data).
        A function that takes a list of filepaths, creates a pd.Dataframe containing the info from the each file. If the
        file contain more than one row of samples all the rows below the first data row is removed before appending it to
        the final df that gets retuned.

        :param fp: A list of the filepaths we want to read in.
        :param set_target: The target we want for the files in the folder. 0 for acl, 1 for controll, 2 for athletes.
        :return: A dataframe containing the data from the files we just read in.
        """
    # Iterating over all filepaths given.
    dict_list_of_list = collections.defaultdict(list)
    unique_columns_list = list('')

    for filepath in fp:
        print(f'IMPORTING FILE: {filepath}')
        # Loading data as df and extracting the rows needed.
        temp_df = pd.read_csv(filepath, delimiter="\t")
        temp_df.drop(temp_df.columns[0], axis=1, inplace=True)
        file_name = os.path.basename(filepath).removesuffix(".txt")

        # Combining top rows containing info to get unique variable names.
        # temp_df.iloc[0] = Variable, temp_df.iloc[2] = Folder, temp_df.iloc[3] = Item (X,Y,Z)
        categories_of_data = temp_df.iloc[0] + '_' + temp_df.iloc[2]
        list_of_columns_to_create = categories_of_data.values

        # Getting the amount of unique columns on first iteration before target and name is added.
        if len(dict_list_of_list) == 0:
            #unique_columns_list = list(set(list_of_columns_to_create))
            seen = set()
            unique_columns_list = []
            for item in list_of_columns_to_create:
                if item in seen:
                    break
                seen.add(item)
                unique_columns_list.append(item)
            number_of_unique_cols = len(unique_columns_list)
        
        number_of_repetitions = len(list_of_columns_to_create) / number_of_unique_cols
        temp_df_cols = [i for i in temp_df.columns]
        # Slicing df width so that that repetitions don't get added as columns.
        for i in range(1, int(number_of_repetitions) + 1):
            # Creating df with the slice of the columns containing one repetitions data.
            new_df = temp_df[temp_df_cols[(i - 1) * int(number_of_unique_cols):(i * int(number_of_unique_cols))]]
            # Dropping all values but the actual datapoints.
            new_df.drop([0, 1, 2, 3], inplace=True)
            # Checking for nan values and only adding if the set is intact.
            if not new_df.isna().values.any():
                # Making the datapoints into a list of lists.
                new_df.columns=unique_columns_list
                actual_data = new_df.values.tolist()

                # Filling dictionary to save each repetitions data.
                if 'patient_name' in dict_list_of_list:
                    dict_list_of_list['data'].append(actual_data)
                    dict_list_of_list['patient_name'].append(file_name)
                    dict_list_of_list['target'].append(set_target)
                else:
                    dict_list_of_list['data'] = [actual_data]
                    dict_list_of_list['patient_name'] = [file_name]
                    dict_list_of_list['target'] = [set_target]
                    dict_list_of_list['variable_name'] = unique_columns_list

        # Creating df for the patient.
        # saknas värden för vissa sensorer så df blir olika lång -.- fix it.

    folder_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_list_of_list.items()]))

    folder_df.dropna(inplace=True)

    return folder_df

def build_sets(folder_path):
    """
    A function that gets all patient files on the affected or non dominant leg from each folderpath specified below.
    In order for function to work you will have to change the paths to be the location where you store the data locally.
    The dataframes created contain the data points with set targets (0 for acl, 1 for control and 2 for athlete) and
    the patient name in the form of one sample filename.

    """
    # Setting up variables containing the paths to the 3 groups.
    ########## ACL GROUP #############

    # SH data paths
    sh_acl_p2d_fp = glob.glob(f"{folder_path}/SH AI export/ACL Injured/anonymized/**/*_A_p2d.txt", recursive=True)

    ######### ATHLETE GROUP ##############

    # SH data paths
    sh_athletes_p2d_fp = glob.glob(f"{folder_path}/SH AI export/Athletes/anonymized/**/*_ND_p2d.txt", recursive=True)

    ######### Controls group #############

    # SH data paths
    sh_controls_p2d_fp = glob.glob(f"{folder_path}/SH AI export/Controls/anonymized/**/*_ND_p2d.txt", recursive=True)

    ########## IMPORTING ############

    sh_acl_p2d_df = fetch_p2d(sh_acl_p2d_fp, 0)  
    sh_controls_p2d_df = fetch_p2d(sh_controls_p2d_fp, 1)
    sh_athletes_p2d_df = fetch_p2d(sh_athletes_p2d_fp, 2)
    sh_p2d_df = pd.concat([sh_acl_p2d_df, sh_controls_p2d_df, sh_athletes_p2d_df], ignore_index=True)

    return sh_p2d_df


def testplot(sh_p2d_df, p_name, p_rep, p_variable_name):
    patient_data = sh_p2d_df.loc[sh_p2d_df['patient_name'] == p_name]
    unique_columns_list = list(sh_p2d_df['variable_name'].values)
    v_index = unique_columns_list.index(p_variable_name)
    patient_data_np = np.array(patient_data['data'][p_rep-1]).T   # this is a mess GHAAA!!#(/%)
    patient_data_nps = patient_data_np.astype(float)            # AAA...
    p_variable = patient_data_nps[v_index]
    x = np.linspace(0, 100, len(p_variable))
    plt.plot(x, p_variable)
    plt.xlabel('0-100%')
    plt.ylabel(p_variable_name)
    plt.title(p_name + ' rep' + str(p_rep))
    plt.show()
    return


folder_path = 'D:\DevFileStorage\SideHop p2d export\01_joint'
sh_p2d_df = build_sets(folder_path)

# JS test
p_name = '103_P_T2_sh_A_p2d'
p_rep = 2
p_variable_name = 'knee_Y_lat'
testplot(sh_p2d_df, p_name, p_rep, p_variable_name)