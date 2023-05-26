import config
from train.constrastive.Ttrain import Ttrain, Ttest
from train.constrastive.Vtrain import Vtrain, Vtest
from train.constrastive.Atrain import Atest, Atrain


if __name__ == '__main__':
    # follow below performance
    load_metric = config.SIMS.downStream.load_metric
    check_list = config.SIMS.downStream.check_list
    metric = config.SIMS.downStream.metric
    # select which model to save
    check = config.SIMS.downStream.check
    result_path = config.SIMS.path.result_path
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
