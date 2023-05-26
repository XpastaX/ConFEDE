import torch
import os
from util.common import check_dir

seed = [1, 12, 123, 1234, 1235]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGPATH = 'log/'
check_dir(LOGPATH)
USEROBERTA = False


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
