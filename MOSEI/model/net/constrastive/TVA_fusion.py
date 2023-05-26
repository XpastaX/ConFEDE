from model.net.constrastive.text_encoder_finetune import TextEncoder
from model.net.constrastive.vision_encoder_finetune import VisionEncoder
from model.net.constrastive.audio_encoder_fintune import AudioEncoder
import torch
import config as default_config
from torch import nn
from model.decoder.classifier import BaseClassifier
from util.metrics import weighted_NTXentLoss
import numpy as np
from util.common import check_dir


class projector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(projector, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TVA_fusion(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, drop_out=None, config=default_config):
        super(TVA_fusion, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(name=name, with_projector=False, config=config)
        self.vision_encoder = VisionEncoder(config=config)
        self.audio_encoder = AudioEncoder(config=config)
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.MOSEI.downStream.text_drop_out

        uni_fea_dim = int(encoder_fea_dim/2)

        self.T_simi_proj = projector(encoder_fea_dim, uni_fea_dim)
        self.V_simi_proj = projector(encoder_fea_dim, uni_fea_dim)
        self.A_simi_proj = projector(encoder_fea_dim, uni_fea_dim)

        self.T_dissimi_proj = projector(encoder_fea_dim, uni_fea_dim)
        self.V_dissimi_proj = projector(encoder_fea_dim, uni_fea_dim)
        self.A_dissimi_proj = projector(encoder_fea_dim, uni_fea_dim)

        hidden_size = [uni_fea_dim * 2, uni_fea_dim, int(uni_fea_dim / 2), int(uni_fea_dim / 4),
                       ]

        self.TVA_decoder = BaseClassifier(input_size=uni_fea_dim * 6,
                                          hidden_size=hidden_size,
                                          output_size=1, drop_out=drop_out,
                                          name='TVARegClassifier', )

        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )

        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

        self.batch_size = config.MOSEI.downStream.TVAExp_fusion.batch_size
        self.heat = config.MOSEI.downStream.const_heat

        self.ntxent_loss = weighted_NTXentLoss(temperature=self.heat)
        self.set_train()

    def forward(self, sample1, sample2, return_loss=True, return_emb=False, device=None):
        if device is None:
            device = self.device

        text1 = sample1['raw_text']
        vision1 = sample1['vision'].clone().detach().to(device).float()
        audio1 = sample1['audio'].clone().detach().to(device).float()
        label1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_T1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_V1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_A1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        key_padding_mask_V1, key_padding_mask_A1 = (sample1['vision_padding_mask'].clone().detach().to(device),
                                                    sample1['audio_padding_mask'].clone().detach().to(device))

        x_t_embed = self.text_encoder(text1, device=device).squeeze()
        x_v_embed = self.vision_encoder(vision1, key_padding_mask=key_padding_mask_V1, device=device).squeeze()
        x_a_embed = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1, device=device).squeeze()

        x_t_simi1 = self.T_simi_proj(x_t_embed)
        x_v_simi1 = self.V_simi_proj(x_v_embed)
        x_a_simi1 = self.A_simi_proj(x_a_embed)
        x_t_dissimi1 = self.T_dissimi_proj(x_t_embed)
        x_v_dissimi1 = self.V_dissimi_proj(x_v_embed)
        x_a_dissimi1 = self.A_dissimi_proj(x_a_embed)

        x1_s = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1), dim=-1)
        x1_ds = torch.cat((x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=-1)
        x1_all = torch.cat((x1_s, x1_ds), dim=-1)
        x1_sds = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1,
                            ), dim=0)
        label1_sds = torch.cat((label1, label1, label1, label_T1, label_V1, label_A1,), dim=0)
        x_sds = x1_sds
        label_sds = label1_sds
        x2 = None
        x = x1_all
        label_all = label1
        if sample2 is not None:
            text2 = sample2['raw_text']
            vision2 = sample2['vision'].clone().detach().to(device).float()
            audio2 = sample2['audio'].clone().detach().to(device).float()
            label2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_T2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_V2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_A2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach().to(device),
                                                        sample2['audio_padding_mask'].clone().detach().to(device))

            x_t_embed2 = self.text_encoder(text2, device=device).squeeze()
            x_v_embed2 = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2, device=device).squeeze()
            x_a_embed2 = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2, device=device).squeeze()

            x_t_simi2 = self.T_simi_proj(x_t_embed2)
            x_v_simi2 = self.V_simi_proj(x_v_embed2)
            x_a_simi2 = self.A_simi_proj(x_a_embed2)
            x_t_dissimi2 = self.T_dissimi_proj(x_t_embed2)
            x_v_dissimi2 = self.V_dissimi_proj(x_v_embed2)
            x_a_dissimi2 = self.A_dissimi_proj(x_a_embed2)

            x2_s = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2), dim=-1)
            x2_ds = torch.cat((x_t_dissimi2, x_v_dissimi2, x_a_dissimi2), dim=-1)
            x2_all = torch.cat((x2_s, x2_ds), dim=-1)
            x2_sds = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2, x_a_dissimi2,
                                ), dim=0)
            label2_sds = torch.cat((label2, label2, label2, label_T2, label_V2, label_A2,), dim=0)
            x = torch.cat((x1_all, x2_all), dim=0)
            label_all = torch.cat((label1.squeeze(), label2.squeeze()), dim=0)
            x_sds = torch.cat((x1_sds, x2_sds), dim=0)
            label_sds = torch.cat((label1_sds, label2_sds), dim=0)

        if return_loss:
            pred = self.TVA_decoder(x)
            pred_mono = self.mono_decoder(x_sds)
            sup_const_loss = 0
            # sds_loss = 0
            if sample2 is not None:
                # [Ts,T1s,T2s,T3s,T4s,T5s,T6s,V1s,V2s,V3s,....]
                t1, p, t2, n = torch.tensor([0, 0, 7, 7, 14, 14,
                                             0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                            device=device), \
                               torch.tensor([1, 2, 8, 9, 15, 16,
                                             7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20],
                                            device=device), \
                               torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
                                             0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                            device=device), \
                               torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
                                             21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27,
                                             34, 41], device=device)

                indices_tuple = (t1, p, t2, n)
                pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
                                                 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, ])
                for i in range(len(x1_all)):
                    pre_sample_x = []
                    for fea1, fea2 in zip([x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1, ],
                                          [x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2,
                                           x_a_dissimi2, ]):
                        pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))

                    sup_const_loss += self.ntxent_loss(torch.cat(pre_sample_x, dim=0), pre_sample_label,
                                                       indices_tuple=indices_tuple)

                sup_const_loss /= len(x1_all)

            pred_loss = self.criterion(pred.squeeze(), label_all)
            mono_task_loss = self.criterion(pred_mono.squeeze(), label_sds)

            loss = pred_loss + 0.1 * sup_const_loss + 0.01 * mono_task_loss
            if return_emb:
                return pred, x1_all, loss, pred_loss, sup_const_loss
            else:
                return pred, (x_t_embed, x_v_embed, x_a_embed), loss, pred_loss, sup_const_loss
        else:
            if return_emb:
                return x1_all
            else:
                return (x_t_embed, x_v_embed, x_a_embed)

    def save_model(self, name):
        # save all modules
        mode_path = self.model_path + 'TVA_fusion' + '_model.ckpt'

        print('model saved at:')
        print(mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name, load_pretrain=False):
        if load_pretrain:
            text_encoder_path = self.config.MOSEI.path.encoder_path + name + '_text_encoder.ckpt'
            vision_encoder_path = self.config.MOSEI.path.encoder_path + name + '_vision_encoder.ckpt'
            audio_encoder_path = self.config.MOSEI.path.encoder_path +name + '_audio_encoder.ckpt'

            print('model loaded from:')
            print(text_encoder_path)
            print(vision_encoder_path)
            print(audio_encoder_path)
            self.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=self.device))
            # self.text_encoder.tokenizer.from_pretrained(self.config.SIMS.path.bert_en,do_lower_case=True)
            # self.text_encoder.extractor.from_pretrained(self.config.SIMS.path.bert_en)
            self.vision_encoder.load_state_dict(torch.load(vision_encoder_path, map_location=self.device))
            self.audio_encoder.load_state_dict(torch.load(audio_encoder_path, map_location=self.device))

        else:
            mode_path = self.model_path + 'TVA_fusion' + '_model.ckpt'

            print('model loaded from:')
            print(mode_path)
            self.load_state_dict(torch.load(mode_path, map_location=self.device))

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [False, False, True, True]

        for param in self.parameters():
            param.requires_grad = train_module[3]
        self.text_encoder.set_train(train_module=train_module[0:2])
        self.vision_encoder.set_train(train_module=train_module[2])
        self.audio_encoder.set_train(train_module=train_module[2])
