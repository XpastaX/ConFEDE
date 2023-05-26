import config
from train.constrastive.Ttrain import Ttrain, Ttest
from train.constrastive.Vtrain import Vtrain, Vtest
from train.constrastive.Atrain import Atest, Atrain

import torch
if __name__ == '__main__':
    config.DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # follow below performance
    load_metric = config.MOSI.downStream.load_metric
    check_list = config.MOSI.downStream.check_list
    metric = config.MOSI.downStream.metric
    # select which model to save
    check = config.MOSI.downStream.check
    result_path = config.MOSI.path.result_path
    seed = config.seed

    print('text pretrain')
    Ttrain(check={'MAE':10000}, config=config)
    Ttest(check_list=['MAE'], config=config)

    print('vision pretrain')
    Vtrain(check={'MAE':10000}, config=config)
    Vtest(check_list=['MAE'], config=config)

    print('audio pretrain')
    Atrain(check={'MAE':10000}, config=config)
    Atest(check_list=['MAE'], config=config)

