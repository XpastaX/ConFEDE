import os
import random
import numpy as np
import torch


def check_dir(path, make_dir=True):
    if not os.path.exists(path):  #
        os.makedirs(path)


def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def check_and_save(model, result, check, save_model=True, name=None, train_type=None):
    file_name = 'best_'
    if name is not None:
        file_name = name + '_' + file_name
    if train_type is None:
        for key in check.keys():
            if key == 'MAE' or key == 'Loss':
                if check[key] > result[key]:
                    if save_model:
                        model.save_model(file_name + key)
                    check[key] = result[key]
            else:
                if check[key] < result[key]:
                    if save_model:
                        model.save_model(file_name + key)
                    check[key] = result[key]
    else:
        for key in check.keys():
            if key == 'MAE' or key == 'Loss':
                if check[key] > result[key]:
                    if save_model:
                        model.save_model(file_name + key, train_type=train_type)
                    check[key] = result[key]
            else:
                if check[key] < result[key]:
                    if save_model:
                        model.save_model(file_name + key, train_type=train_type)
                    check[key] = result[key]
    return check


class best_result:
    def __init__(self):
        self.result_save = {'Mult_acc_2': {},
                            'F1_score': {},
                            'Mult_acc_3': {},
                            'Mult_acc_5': {},
                            'MAE': {},
                            'Corr': {},
                            'Loss': {}}
        self.result_check = {'Mult_acc_2': 0,
                             'F1_score': 0,
                             'Mult_acc_3': 0,
                             'Mult_acc_5': 0,
                             'MAE': 100,
                             'Corr': 0,
                             'Loss': 10000}

    def check(self, result, result_test):
        for key in result.keys():
            if key == 'MAE' or key == 'Loss':
                if self.result_check[key] > result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test
            else:
                if self.result_check[key] < result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test

    def print(self):
        for key in self.result_save.keys():
            print('==========================')
            print('Best test based on %s' % key)
            for key2 in self.result_save[key].keys():
                print('\t%s: %s' % (key2, self.result_save[key][key2]))
