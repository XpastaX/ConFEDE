import torch
import os
from util.common import check_dir

seed = [1, 12, 123, 1234, 1235]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGPATH = 'log/'
check_dir(LOGPATH)
USEROBERTA = False

# collect positive and negative pairs
keys = ['T{}s', 'V{}s', 'A{}s', 'T{}d', 'V{}d', 'A{}d']
num_sample = 7
all_keys = {}
idx = 0
for item in keys:
    for i in range(num_sample):
        all_keys[(item.replace('{}', str(i)))] = idx
        idx += 1
positive_pairs = [
    # inter-sample pairing
    'T0s,T1s',
    'T0s,T2s',
    'V0s,V1s',
    'V0s,V2s',
    'A0s,A1s',
    'A0s,A2s',
    # intra-sample pairing
    'T0s,V0s',
    'T0s,A0s',
    'T1s,V1s',
    'T1s,A1s',
    'T2s,V2s',
    'T2s,A2s',
    'T3s,V3s',
    'T3s,A3s',
    'T4s,V4s',
    'T4s,A4s',
    'T5s,V5s',
    'T5s,A5s',
    'T6s,V6s',
    'T6s,A6s',
]
negative_pairs = [
    # inter-sample pairing
    'T0s,T3s',
    'T0s,T4s',
    'T0s,T5s',
    'T0s,T6s',
    'V0s,V3s',
    'V0s,V4s',
    'V0s,V5s',
    'V0s,V6s',
    'A0s,A3s',
    'A0s,A4s',
    'A0s,A5s',
    'A0s,A6s',
    # intra-sample pairing
    'T0s,T0d',
    'T0s,V0d',
    'T0s,A0d',
    'T1s,T1d',
    'T1s,V1d',
    'T1s,A1d',
    'T2s,T2d',
    'T2s,V2d',
    'T2s,A2d',
    'T3s,T3d',
    'T3s,V3d',
    'T3s,A3d',
    'T4s,T4d',
    'T4s,V4d',
    'T4s,A4d',
    'T5s,T5d',
    'T5s,V5d',
    'T5s,A5d',
    'T6s,T6d',
    'T6s,V6d',
    'T6s,A6d',
]

t1, p, t2, n = [], [], [], []
for pair in positive_pairs:
    eA, eB = pair.split(',')
    eA_idx = all_keys[eA]
    eB_idx = all_keys[eB]
    t1.append(eA_idx)
    p.append(eB_idx)
for pair in negative_pairs:
    eA, eB = pair.split(',')
    eA_idx = all_keys[eA]
    eB_idx = all_keys[eB]
    t2.append(eA_idx)
    n.append(eB_idx)


class SIMS:
    class path:
        raw_data_path = 'data/SIMS/unaligned_39.pkl'
        model_path = 'ckpt/fusion-best'
        encoder_path = 'ckpt/fea_encoder/'

        if USEROBERTA:
            model_path = model_path + '/roberta/'
        else:
            model_path = model_path + '/bert/'
        check_dir(model_path)
        result_path = 'result/'
        check_dir(result_path)

    class downStream:
        # follow below performance
        metric = 'Mult_acc_2'
        load_metric = 'best_' + 'MAE'
        check_list = [metric]

        # select which model to save
        check = {metric: 10000 if metric == 'Loss' or metric == 'MAE' else 0}

        # parameters
        use_reg = True
        proj_fea_dim = 256
        encoder_fea_dim = 768
        text_fea_dim = 768
        vision_fea_dim = 709
        video_seq_len = 55
        audio_fea_dim = 33
        audio_seq_len = 400
        text_drop_out = 0.5
        vision_drop_out = 0.5
        audio_drop_out = 0.5
        vision_nhead = 8
        audio_nhead = 8
        vision_dim_feedforward = vision_fea_dim
        audio_dim_feedforward = audio_fea_dim
        vision_tf_num_layers = 1
        audio_tf_num_layers = 2

        sds_heat = 0.5
        const_heat = 0.5

        class textPretrain:
            batch_size = 64
            lr = 1e-5
            epoch = 150
            decay = 1e-3
            num_warm_up = 5

        class visionPretrain:
            batch_size = 64
            lr = 1e-4
            epoch = 300
            decay = 1e-3
            num_warm_up = 10

        class audioPretrain:
            batch_size = 64
            lr = 1e-4
            epoch = 300
            decay = 1e-3
            num_warm_up = 10

        class TVAExp_fusion:
            batch_size = 32
            lr = 1e-4
            epoch = 50
            decay = 1e-3
            num_warm_up = 1
            finetune_epoch = 200

