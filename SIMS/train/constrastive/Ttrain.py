import torch
import config as default_config
from model.net.constrastive.text_encoder_finetune import TextPretrain
from dataloader.SIMS import SIMSDataloader
from tqdm import tqdm
import transformers as trans
from util.metrics import Metrics
from util.common import write_log, check_and_save
import datetime


def Ttrain(exp_type=None, load_model=None, check=None, config=default_config):
    print('---------------TextPretrain---------------')
    if check is None:
        check = {'Non0_F1_score': 0, 'MAE': 100, 'Loss': 10000}
    else:
        check = check.copy()

    log_path = config.LOGPATH + "SIMS_TextPretrain." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'

    train_data = SIMSDataloader('train', batch_size=config.SIMS.downStream.textPretrain.batch_size)
    valid_data = SIMSDataloader('valid', shuffle=False, batch_size=config.SIMS.downStream.textPretrain.batch_size,
                                num_workers=0)
    metrics = Metrics()

    model = TextPretrain(config=config)

    device = config.DEVICE
    batch_size = config.SIMS.downStream.textPretrain.batch_size
    lr = config.SIMS.downStream.textPretrain.lr
    total_epoch = config.SIMS.downStream.textPretrain.epoch
    decay = config.SIMS.downStream.textPretrain.decay
    num_warm_up = config.SIMS.downStream.textPretrain.num_warm_up

    optimizer = trans.optimization.AdamW(params=model.parameters(), lr=lr, weight_decay=decay)

    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=int(
                                                                       num_warm_up * (len(train_data))),
                                                                   num_training_steps=total_epoch * len(train_data), )
    model.to(device)
    if load_model is not None:
        model.load_model(load_model, module='encoder')
    model.train()
    loss = 0

    train_all_epoch = int(total_epoch / 3)
    for epoch in range(1, total_epoch + 1):
        if epoch < train_all_epoch:
            model.set_train([False, True, True])
        else:
            model.set_train([True, True, True])
        bar = tqdm(train_data, disable=False)
        for index, sample in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:[%s]|" % (epoch, loss))
            optimizer.zero_grad()
            text = sample['raw_text']
            label = sample['regression_labels_T'].clone().detach().to(device)
            pred, fea, loss = model(text, label.float().squeeze(), return_loss=True)
            loss.backward()

            optimizer.step()
            scheduler.step()

        result, result_loss = eval(model, metrics, valid_data, device, config)

        log = 'visionPretarin_TrainAcc\n\tEpoch:%d\n\tacc_2:%s\n\tF1_score:%s\n\tacc_3"%s\n\t' \
              'acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  epoch, result['Mult_acc_2'], result['F1_score'], result['Mult_acc_3'], result['Mult_acc_5'],
                  result['MAE'], result['Corr'], result_loss)
        print(log)
        write_log(log, path=log_path)
        if epoch > train_all_epoch:
            check = check_and_save(model, result, check)
    print(check)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


def eval(model, metrics=None, eval_data=None, device=None, config=default_config):
    if device is None: device = config.DEVICE
    if eval_data is None: eval_data = SIMSDataloader('test', shuffle=False, num_workers=0,
                                                     batch_size=config.SIMS.downStream.textPretrain.batch_size)
    if metrics is None: metrics = Metrics()

    model.eval()
    with torch.no_grad():
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            text = sample['raw_text']
            label = sample['regression_labels_T'].clone().detach().to(device)
            _pred, fea, _loss = model(text, label.float().squeeze(), return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * config.SIMS.downStream.textPretrain.batch_size
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_sims_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
    model.train()
    return eval_results, loss / len(eval_data)


def Ttest(check_list=None, config=default_config):
    if check_list is None: check_list = ['Has0_acc_2', 'Has0_F1_score', 'Mult_acc_7']
    if not isinstance(check_list, list): check_list = [check_list]
    log_path = config.LOGPATH + "SIMS_TextPretrain_Test." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    model = TextPretrain(config=config)
    device = config.DEVICE
    model.to(device)
    check = {}
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name='best_' + metric)
        result, loss = eval(model=model, device=device, config=config)
        check[metric] = result[metric]

        log = '\tacc_2:%s\n\tF1_score:%s\n\t' \
              'Mult_acc_3:%s\n\tMult_acc_5:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  result['Mult_acc_2'], result['F1_score'], result['Mult_acc_3'], result['Mult_acc_5'], result['MAE'],
                  result['Corr'], loss)
        print(log)
        write_log(metric + '\n' + log, log_path)
