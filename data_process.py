import os
import numpy as np


def data_process(data_path):
    demographic_data = []
    diagnosis_data = []
    idx_list = []

    for id_name in os.listdir(data_path):
        for demo_file in os.listdir(os.path.join(data_path, id_name)):
            if demo_file[0:7] != 'episode':
                continue
            cur_file = data_path + id_name + '/' + demo_file
            with open(cur_file, "r") as tsfile:
                header = tsfile.readline().strip().split(',')
                if header[0] != "Icustay":
                    continue
                cur_data = tsfile.readline().strip().split(',')
                if len(cur_data) == 1:
                    cur_demo = np.zeros(4).tolist()
                    cur_diag = np.zeros(128).tolist()
                else:
                    if cur_data[3] == '':
                        cur_data[3] = 60.0
                    if cur_data[4] == '':
                        cur_data[4] = 160
                    if cur_data[5] == '':
                        cur_data[5] = 60
                    cur_demo = [int(cur_data[2]),float(cur_data[3]),float(cur_data[4]),float(cur_data[5])]
                    cur_diag = [int(cur_data[i+8]) for i in range(len(cur_data[8:len(cur_data)+1]))]

                demographic_data.append(cur_demo)
                diagnosis_data.append(cur_diag)
                idx_list.append(id_name + '_' + demo_file[0:8])

    return demographic_data, diagnosis_data, idx_list
