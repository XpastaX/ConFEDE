from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from model.projector import FeatureProjector
from model.decoder.classifier import BaseClassifier
import torch
from torch import nn
import config as default_config


class TextEncoder(nn.Module):
    def __init__(self, name=None, fea_size=None, proj_fea_dim=None, drop_out=None, with_projector=True,
                 config=default_config):
        super(TextEncoder, self).__init__()
        self.name = name
        if fea_size is None:
            fea_size = config.MOSI.downStream.text_fea_dim
        if proj_fea_dim is None:
            proj_fea_dim = config.MOSI.downStream.proj_fea_dim
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out
        if config.USEROBERTA:
            self.tokenizer = BertTokenizer.from_pretrained("roberta-base")
            self.extractor = BertModel.from_pretrained("roberta-base")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.extractor = BertModel.from_pretrained('bert-base-uncased')
        self.with_projector = with_projector
        if with_projector:
            self.projector = FeatureProjector(fea_size, proj_fea_dim, drop_out=drop_out,
                                              name='text_projector', config=config)
        self.device = config.DEVICE

    def forward(self, text, device=None):
        if device is None:
            device = self.device

        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
            device)
        x = self.extractor(**x)['pooler_output']
        if self.with_projector:
            x = self.projector(x)
        return x

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for name, param in self.extractor.named_parameters():
                param.requires_grad = train_module[0]

        if self.with_projector:
            for param in self.projector.parameters():
                param.requires_grad = train_module[1]


class TextPretrain(nn.Module):
    def __init__(self, name=None, proj_fea_dim=768, drop_out=None, config=default_config):
        super(TextPretrain, self).__init__()
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out
        self.encoder = TextEncoder(name=name, with_projector=False)  # bert output 768
        self.classifier = BaseClassifier(input_size=proj_fea_dim,
                                         hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4),
                                                      int(proj_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out, name='RegClassifier', )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config=config

    def forward(self, text, label, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(text, device=device)
        pred = self.classifier(x)

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, x, loss
        else:
            return pred, x

    def save_model(self, name):
        # save all modules
        encoder_path = self.config.MOSI.path.encoder_path + name + '_text_encoder.ckpt'
        decoder_path = self.config.MOSI.path.encoder_path + name + '_text_decoder.ckpt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name, module=None):
        encoder_path = self.config.MOSI.path.encoder_path + name + '_text_encoder.ckpt'
        decoder_path = self.config.MOSI.path.encoder_path + name + '_text_decoder.ckpt'
        print('model loaded from:')
        if module == 'encoder':
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)
        if module == 'decoder':
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)
        if module == 'all' or module is None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True, True]
        self.encoder.set_train(train_module=train_module[0:2])
        for param in self.classifier.parameters():
            param.requires_grad = train_module[2]
