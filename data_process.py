import os
import torch
from torch.utils import data
import numpy as np
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer


class Dataset(data.Dataset):
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.name[index]

    def __len__(self):
        return len(self.x)


def data_process_x(data_x_path):
    small_part = False
    arg_timestep = 0.8
    batch_size = 256
    time_length = 48.0

    # Build readers, discretizers, normalizers
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_x_path, 'train'),
                                             listfile=os.path.join(data_x_path, 'train_listfile.csv'),
                                             period_length=time_length)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_x_path, 'train'),
                                           listfile=os.path.join(data_x_path, 'val_listfile.csv'),
                                           period_length=time_length)

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(data_path, 'test'),
                                            listfile=os.path.join(data_path, 'test_listfile.csv'),
                                            period_length=time_length)

    discretizer = Discretizer(timestep=arg_timestep,
                              store_masks=True,
                              impute_strategy='previous',
                              start_time='zero')
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)
    normalizer_state = 'ihm_ts0.8.input_str_previous.start_time_zero.normalizer'
    normalizer_state = os.path.join(os.path.dirname(data_x_path), normalizer_state)
    normalizer.load_params(normalizer_state)

    n_trained_chunks = 0
    train_raw = utils.load_data(train_reader, discretizer, normalizer, small_part, return_names=True)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, small_part, return_names=True)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, small_part, return_names=True)

    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def data_process_s(data_s_path):
    demographic_data = []
    diagnosis_data = []
    idx_list = []

    for id_name in os.listdir(data_s_path):
        for demo_file in os.listdir(os.path.join(data_s_path, id_name)):
            if demo_file[0:7] != 'episode':
                continue
            cur_file = data_s_path + id_name + '/' + demo_file
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
