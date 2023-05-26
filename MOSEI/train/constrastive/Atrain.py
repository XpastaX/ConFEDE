import torch
import config as default_config
from model.net.constrastive.audio_encoder_fintune import AudioPretrain
from dataloader.MOSEI import MOSEIDataloader
from tqdm import tqdm
import transformers as trans
from util.metrics import Metrics
from util.common import write_log, check_and_save
import datetime


def Atrain(exp_type=None, load_model=None, check=None, config=default_config):
    print('---------------AudioPretrain---------------')
    if check is None:
        check = {'Non0_F1_score': 0, 'MAE': 100, 'Loss': 10000}
    else:
        check = check.copy()
    log_path = config.LOGPATH + "MOSEI_AudioPretrain." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    metrics = Metrics()

    model = AudioPretrain(config=config)

    device = config.DEVICE
    batch_size = config.MOSEI.downStream.audioPretrain.batch_size
    lr = config.MOSEI.downStream.audioPretrain.lr
    total_epoch = config.MOSEI.downStream.audioPretrain.epoch
    decay = config.MOSEI.downStream.audioPretrain.decay
    num_warm_up = config.MOSEI.downStream.audioPretrain.num_warm_up

    train_data = MOSEIDataloader('train', batch_size=batch_size)
    valid_data = MOSEIDataloader('valid', shuffle=False, num_workers=0, batch_size=batch_size)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, amsgrad=False, )
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=int(
                                                                       num_warm_up * (len(train_data))),
                                                                   num_training_steps=total_epoch * len(train_data), )
    model.to(device)
    if load_model is not None:
        model.load_model(load_model, module='encoder')
    loss = 0
    start_save = 5
    for epoch in range(1, total_epoch + 1):
        model.set_train([True, True])
        model.train()
        bar = tqdm(train_data, disable=False)
        for index, sample in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:[%s]|" % (epoch, loss))
            optimizer.zero_grad()
            audio = sample['audio'].clone().detach().to(device).float()
            label = sample['regression_labels'].clone().detach().to(device).float()
            mask = sample['audio_padding_mask'].clone().detach().to(device)
            pred, fea, loss = model(audio, label.float().squeeze(), mask, return_loss=True)
            loss.backward()

            optimizer.step()
            scheduler.step()

        result, result_loss = eval(model, metrics, valid_data, device, config)

        log = 'audioPretarin_ValidAcc\n\tEpoch:%d\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  epoch, result['Has0_acc_2'], result['Has0_F1_score'],
                  result['Non0_acc_2'], result['Non0_F1_score'], result['Mult_acc_5'],
                  result['Mult_acc_7'], result['MAE'], result['Corr'], result_loss)

        print(log)
        write_log(log, path=log_path)
        if epoch > start_save:
            check = check_and_save(model, result, check, save_model=True)
    print(check)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


def eval(model, metrics=None, eval_data=None, device=None, config=default_config):
    if device is None: device = config.DEVICE
    if eval_data is None: eval_data = MOSEIDataloader('test', shuffle=False, num_workers=0,
                                                      batch_size=config.MOSEI.downStream.audioPretrain.batch_size)
    if metrics is None: metrics = Metrics()

    model.eval()
    with torch.no_grad():
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            audio = sample['audio'].clone().detach().to(device).float()
            label = sample['regression_labels'].clone().detach().to(device).float()
            mask = sample['audio_padding_mask'].clone().detach().to(device)
            _pred, fea, _loss = model(audio, label.float().squeeze(), mask, return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * config.MOSEI.downStream.audioPretrain.batch_size
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
    model.train()
    return eval_results, loss / len(eval_data)


def Atest(check_list=None, config=default_config):
    if check_list is None: check_list = ['Has0_F1_score', 'MAE']
    if not isinstance(check_list, list): check_list = [check_list]
    log_path = config.LOGPATH + "MOSEI_AudioPretrain_Test." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    model = AudioPretrain(config=config)
    device = config.DEVICE
    model.to(device)
    check = {}
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name='best_' + metric)
        result, loss = eval(model=model, device=device, config=config)
        check[metric] = result[metric]
        log = '\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
            result['Has0_acc_2'], result['Has0_F1_score'],
            result['Non0_acc_2'], result['Non0_F1_score'], result['Mult_acc_5'],
            result['Mult_acc_7'], result['MAE'],
            result['Corr'], loss)

        print(log)
        write_log(metric + '\n' + log, log_path)
