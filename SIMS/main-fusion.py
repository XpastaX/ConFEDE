import config
from train.constrastive.TVA_fusion_train import TVA_train_fusion, TVA_test_fusion
import datetime
from util.common import set_random_seed

if __name__ == '__main__':
    # follow below performance
    load_metric = config.SIMS.downStream.load_metric
    check_list = config.SIMS.downStream.check_list
    metric = config.SIMS.downStream.metric
    # select which model to save
    check = config.SIMS.downStream.check
    result_path = config.SIMS.path.result_path
    seed = config.seed
    result_M = {}
    for s in seed:
        config.seed = s

        set_random_seed(s)
        print('TVA_fusion')
        TVA_train_fusion('TVA_fusion', check=check,
                         load_model=load_metric,
                         load_pretrain=True,
                         config=config
                         )
        result_M[s] = TVA_test_fusion('TVA_fusion', check_list=check_list, config=config)

    result_path = result_path + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '-fusion' + '.csv'
    with open(result_path, 'w') as file:
        for index, result in enumerate([
            result_M,
        ]):
            file.write('Method%s\n' % index)
            file.write('seed,acc_2,F1,acc_3,acc_5,MAE,Corr\n')
            for s in seed:
                log = '%s,%s,%s,%s,%s,%s,%s\n' % (s,
                                                  result[s][metric]['Mult_acc_2'],
                                                  result[s][metric]['F1_score'],
                                                  result[s][metric]['Mult_acc_3'],
                                                  result[s][metric]['Mult_acc_5'],
                                                  result[s][metric]['MAE'],
                                                  result[s][metric]['Corr'])
                file.write(log)
