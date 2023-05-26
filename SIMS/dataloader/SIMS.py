import torch
import numpy as np
import random
import _pickle as pk
import config as default_config
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import samplers
import math


class SIMSDataset(Dataset):
    def __init__(self, type, use_similarity=False, simi_return_mono=False, config=default_config):
        raw_data_path = config.SIMS.path.raw_data_path
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[type]

        self.simi_return_mono = simi_return_mono
        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.data['id'] = np.array(self.data['id'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
        self.scaled_embedding_averaged = False
        # self.__normalize()
        self.__gen_mask()
        if type == 'train' and use_similarity:
            self.__scale()
            self.__gen_cos_matrix()
        self.use_similarity = use_similarity

    def __gen_mask(self):
        vision_tmp = torch.sum(torch.tensor(self.data['vision']), dim=-1)
        vision_mask = (vision_tmp == 0)

        for i in range(self.size):
            vision_mask[i][0] = False
        vision_mask = torch.cat((vision_mask[:, 0:1], vision_mask), dim=-1)

        self.data['vision_padding_mask'] = vision_mask
        audio_tmp = torch.sum(torch.tensor(self.data['audio']), dim=-1)
        audio_mask = (audio_tmp == 0)
        for i in range(self.size):
            audio_mask[i][0] = False
        audio_mask = torch.cat((audio_mask[:, 0:1], audio_mask), dim=-1)
        self.data['audio_padding_mask'] = audio_mask
        # self.data['vision'] = self.data['vision'][:, 1:, :]
        # self.data['audio'] = self.data['audio'][:, 1:, :]

    def __pad(self):

        PAD = torch.zeros(self.data['vision'].shape[0], 1, self.data['vision'].shape[2])
        self.data['vision'] = np.concatenate((self.data['vision'], PAD), axis=1)
        Ones = torch.ones(self.data['vision'].shape[0], self.data['vision'].shape[2])
        for i in range(len(self.data['vision'])):
            self.data['vision'][i, self.data['vision_lengths'], :] = Ones

        PAD = torch.zeros(self.data['audio'].shape[0], 1, self.data['audio'].shape[2])
        self.data['audio'] = np.concatenate((self.data['audio'], PAD), axis=1)
        Ones = torch.ones(self.data['audio'].shape[0], self.data['audio'].shape[2])
        for i in range(len(self.data['audio'])):
            self.data['audio'][i, self.data['audio_lengths'], :] = Ones

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.data['vision'] = np.mean(self.data['vision'], axis=0, keepdims=True)
        self.data['audio'] = np.mean(self.data['audio'], axis=0, keepdims=True)

        # remove possible NaN values
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0

        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))

    def __scale(self):
        self.scaled_audio = self.data['audio'].copy()
        self.scaled_vision = self.data['vision'].copy()
        self.scaled_text = self.data['text'].copy()
        for i in range(self.audio_fea_size[-1]):
            max_num = np.max(self.data['audio'][:, :, i])
            min_num = np.min(self.data['audio'][:, :, i])
            self.scaled_audio[:, :, i] = (self.data['audio'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        for i in range(self.vision_fea_size[-1]):
            max_num = np.max(self.data['vision'][:, :, i])
            min_num = np.min(self.data['vision'][:, :, i])
            self.scaled_vision[:, :, i] = (self.data['vision'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        # for i in range(len(self.data['text'][0])):
        #     max_num = np.max(self.data['text'][:, :, i])
        #     min_num = np.min(self.data['text'][:, :, i])
        #     self.scaled_text[:, :, i] = (self.data['text'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        self.scaled_audio = torch.tensor(self.scaled_audio)
        self.scaled_vision = torch.tensor(self.scaled_vision)
        self.scaled_text = torch.tensor(self.scaled_text)

    def __gen_cos_matrix(self, model=None):
        self.cos_matrix_M = torch.zeros((self.size, self.size))
        # self.cos_matrix_T = torch.zeros((self.size, self.size))
        # self.cos_matrix_V = torch.zeros((self.size, self.size))
        # self.cos_matrix_A = torch.zeros((self.size, self.size))

        self.text_fea = torch.zeros((self.size, self.size))

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        if not self.scaled_embedding_averaged:
            # calculated mean
            audio_mean = torch.sum(self.scaled_audio, dim=-2)
            vision_mean = torch.sum(self.scaled_vision, dim=-2)
            text_mean = torch.sum(self.scaled_text, dim=-2)

            for i in range(len(audio_mean)):
                audio_mean[i, :] /= self.data['audio_lengths'][i]
            for i in range(len(vision_mean)):
                vision_mean[i, :] /= self.data['vision_lengths'][i]
            for i in range(len(text_mean)):
                text_mean[i, :] /= 39
        else:
            text_mean, vision_mean, audio_mean = self.scaled_text, self.scaled_vision, self.scaled_audio
        # self.cos_matrix_T = cos(text_mean.unsqueeze(1), text_mean.unsqueeze(0))
        # self.cos_matrix_V = cos(vision_mean.unsqueeze(1), vision_mean.unsqueeze(0))
        # self.cos_matrix_A = cos(audio_mean.unsqueeze(1), audio_mean.unsqueeze(0))

        self.cos_matrix_M = cos(torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(1),
                                torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(0))

        self.rank_M = torch.zeros(self.cos_matrix_M.shape)
        # self.rank_T = torch.zeros(self.cos_matrix_T.shape)
        # self.rank_V = torch.zeros(self.cos_matrix_V.shape)
        # self.rank_A = torch.zeros(self.cos_matrix_A.shape)

        for i in range(len(self.cos_matrix_M)):
            _, self.rank_M[i, :] = torch.sort(self.cos_matrix_M[i, :], descending=True)
            # _, self.rank_T[i, :] = torch.sort(self.cos_matrix_T[i, :], descending=True)
            # _, self.rank_V[i, :] = torch.sort(self.cos_matrix_V[i, :], descending=True)
            # _, self.rank_A[i, :] = torch.sort(self.cos_matrix_A[i, :], descending=True)

        # self.T_retrieve = self.__pre_sample(self.rank_T, self.data['regression_labels_T'])
        # self.V_retrieve = self.__pre_sample(self.rank_V, self.data['regression_labels_V'])
        # self.A_retrieve = self.__pre_sample(self.rank_A, self.data['regression_labels_A'])
        self.M_retrieve = self.__pre_sample(self.rank_M, self.data['regression_labels'])

    def __pre_sample(self, _rank, _label):
        retrieve = {'ss': [],
                    'sd': [],
                    'ds': [],
                    'dd': [],
                    }
        for i in range(self.size):
            _ss = []
            _sd = []
            _ds = []
            _dd = []
            for j in range(int(self.size/2)):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    _ss.append(j)
                else:
                    _sd.append(j)
            for j in range(-1, -int(self.size/2), -1):
                if i == j: continue
                if _label[i] == _label[int(_rank[i][j])]:
                    _ds.append(j)
                else:
                    _dd.append(j)
            if len(_ss) < 2 or len(_sd) < 2 or len(_ds) < 2 or len(_dd) < 2:
                print('Unique sample detected, may cause error!')

            retrieve['ss'].append(_ss[:10])
            retrieve['sd'].append(_sd[:10])
            retrieve['ds'].append(_ds[:10])
            retrieve['dd'].append(_dd[:10])
        return retrieve

    def update_matrix(self, T, V, A):
        self.scaled_text = T
        self.scaled_vision = V
        self.scaled_audio = A
        self.scaled_embedding_averaged = True
        self.__gen_cos_matrix()

    def sample(self, _sample_idx):
        if not self.simi_return_mono:
            samples2 = {}
            idx2 = []
            for i in _sample_idx:
                idx2 += random.sample(self.M_retrieve['ss'][i], 2)
                idx2 += random.sample(self.M_retrieve['dd'][i], 2)
                idx2 += random.sample(self.M_retrieve['sd'][i], 2)
            for key in self.data:
                if type(self.data[key]) == list:
                    continue
                else:
                    if type(self.data[key][0]) == np.str_:
                        samples2[key] = self.data[key][idx2].tolist()
                    else:
                        if type(self.data[key]) is not torch.Tensor:
                            samples2[key] = torch.tensor(self.data[key][idx2])
                        else:
                            samples2[key] = self.data[key][idx2].clone().detach()
            return samples2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        samples = {}
        for key in self.data:
            samples[key] = self.data[key][idx]
        return samples


def SIMSDataloader(name, batch_size=None, use_sampler=False, use_similarity=False, simi_return_mono=False, shuffle=True,
                   num_workers=0,
                   prefetch_factor=2,
                   config=default_config):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = SIMSDataset(name, use_similarity=use_similarity, simi_return_mono=simi_return_mono)
    sampler = None
    drop_last = False
    if use_sampler:
        shuffle = False
        drop_last = True
        sampler = samplers.MPerClassSampler(labels=dataset.data['regression_labels'],
                                            m=1,
                                            batch_size=None,
                                            # length_before_new_iter=len(dataset)
                                            length_before_new_iter=batch_size * 21
                                            )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)
