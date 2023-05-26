import config
from train.constrastive.Ttrain import Ttrain, Ttest
from train.constrastive.Vtrain import Vtrain, Vtest
from train.constrastive.Atrain import Atest, Atrain


if __name__ == '__main__':
    # follow below performance
    load_metric = config.MOSEI.downStream.load_metric
    check_list = config.MOSEI.downStream.check_list
    metric = config.MOSEI.downStream.metric
    # select which model to save
    check = config.MOSEI.downStream.check
    result_path = config.MOSEI.path.result_path
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

