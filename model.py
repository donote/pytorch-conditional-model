# encoding: utf-8
# NERStatus Model Architecture
# Paper ACL 2019: https://arxiv.org/abs/1812.05270
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from config.config import ModelConfigNERStatus


class EmbeddingLayer(nn.Module):

    def __init__(self, config: ModelConfigNERStatus):
        super(EmbeddingLayer, self).__init__()
        self.tok_embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                          embedding_dim=config.embedding_dim, padding_idx=0)

        self.tag_embedding = nn.Embedding(num_embeddings=config.tag_size,
                                          embedding_dim=config.embedding_dim, padding_idx=0)

    def forward(self, token_tag, type='vocab'):
        if type == 'vocab':
            emb = self.tok_embedding(token_tag)

        if type == 'tag':
            emb = self.tag_embedding(token_tag)
        return emb


class EncoderLayer(nn.Module):

    def __init__(self, config: ModelConfigNERStatus):
        super(EncoderLayer, self).__init__()
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.2)
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=config.hidden_size * 2, nhead=config.n_head)
        self.transformerEncoder = nn.TransformerEncoder(self.transformerEncoderLayer, num_layers=config.n_encoder)

    def forward(self, embeddings):
        inputs, _ = self.gru(embeddings)
        outputs = self.transformerEncoder(inputs)
        return outputs


class OutputLayerTag(nn.Module):

    def __init__(self, config: ModelConfigNERStatus):
        super(OutputLayerTag, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc1 = nn.Linear(in_features=config.hidden_size * 2,
                             out_features=100)
        self.fc2 = nn.Linear(in_features=100,
                             out_features=config.tag_size)

    def forward(self, encoded_inputs):
        encoded_inputs = self.dropout(F.relu(self.fc1(encoded_inputs)))
        outputs = self.fc2(encoded_inputs)
        return outputs


class OutputLayerStatus(nn.Module):

    def __init__(self, config: ModelConfigNERStatus, input_dim):
        super(OutputLayerStatus, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.fc1 = nn.Linear(in_features=input_dim,
                             out_features=100)
        self.fc2 = nn.Linear(in_features=100,
                             out_features=config.status_size)

    def forward(self, encoded_inputs):
        encoded_inputs = self.dropout(F.relu(self.fc1(encoded_inputs)))
        outputs = self.fc2(encoded_inputs)
        return outputs


class NERStatusUniteComm(nn.Module):

    def __init__(self, config: ModelConfigNERStatus):
        super(NERStatusUniteComm, self).__init__()
        self.embedding = EmbeddingLayer(config)
        self.encoder = EncoderLayer(config)
        self.output_layer_tag = OutputLayerTag(config)
        self.output_layer_status = OutputLayerStatusCond(config, config.hidden_size*2+config.embedding_dim)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, gold_tag=None, gold_status=None, is_train=True):
        token_emb = self.embedding(tokens, type='vocab')
        encoded_inputs = self.encoder(token_emb)  # B * L * H
        tag_output = self.output_layer_tag(encoded_inputs)

        if is_train:
            assert gold_tag is not None
            assert gold_status is not None
            tag_emb = self.embedding(gold_tag, type='tag')
        else:
            pred_tag = torch.argmax(tag_output, dim=2)
            tag_emb = self.embedding(pred_tag, type='tag')

        status_inputs = torch.cat((encoded_inputs, tag_emb), dim=2)
        status_output = self.output_layer_status(status_inputs)

        tag_output_tmp = tag_output.view(-1, tag_output.size(-1))
        gold_tag = gold_tag.view(-1)
        loss_tag = self.criterion(tag_output_tmp, gold_tag)

        status_output_tmp = status_output.view(-1, status_output.size(-1))
        gold_status = gold_status.view(-1)
        loss_status = self.criterion(status_output_tmp, gold_status)

        loss = loss_tag + loss_status
        return loss, tag_output, status_output


class NERStatusUniteConditional(nn.Module):

    def __init__(self, config: ModelConfigNERStatus):
        super(NERStatusUniteConditional, self).__init__()
        self.embedding = EmbeddingLayer(config)
        self.encoder = EncoderLayer(config)
        self.output_layer_tag = OutputLayerTag(config)
        self.output_layer_status = OutputLayerStatus(config, config.hidden_size*2+config.embedding_dim+config.tag_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens, gold_tag=None, gold_status=None, is_train=True):
        token_emb = self.embedding(tokens, type='vocab')
        encoded_inputs = self.encoder(token_emb)  # B * L * H
        tag_output = self.output_layer_tag(encoded_inputs)
        tag_output_soft = F.softmax(tag_output, dim=2)

        pred_tag = torch.argmax(tag_output_soft, dim=2)
        if is_train:
            assert gold_tag is not None
            assert gold_status is not None
            tag_emb = self.embedding(gold_tag, type='tag')
        else:
            tag_emb = self.embedding(pred_tag, type='tag')

        status_inputs = torch.cat((encoded_inputs, tag_emb, tag_output_soft), dim=2)
        status_output = self.output_layer_status(status_inputs)

        tag_output_tmp = tag_output.view(-1, tag_output.size(-1))
        gold_tag_new = gold_tag.view(-1)
        loss_tag = self.criterion(tag_output_tmp, gold_tag_new)

        status_output_tmp = status_output.view(-1, status_output.size(-1))
        gold_status = gold_status.view(-1)
        loss_status = self.criterion(status_output_tmp, gold_status)
        loss = loss_tag + loss_status

        output = {}
        if not is_train:
            output['gold_tag'] = gold_tag.tolist()
            output['pred_tag'] = pred_tag.tolist()
        output['loss'] = loss
        output['description'] = partial(self.description, output=output)
        return output, tag_output, status_output

    @staticmethod
    def description(epoch, epoch_num, output):
        return "train loss: {:.2f}, epoch: {}/{}:".format(output['loss'].item(), epoch + 1, epoch_num)


def unittest():
    print('unittest')
    args = {'vocab_size': 100,
            'embedding_dim': 50,
            'hidden_size': 100,
            'num_layers': 1,
            'tag_size': 3,
            'status_size': 2,
            'fc_dim': 100,
            'n_encoder': 2,
            'positional_embedding_length': 250,
            'type_embedding_num': 2,
            'n_head': 8,
            'pos_emb': 50,
            'dropout': 0.2}

    conf = ModelConfigNERStatus(args)
    #model = NERStatusUniteComm(conf)
    model = NERStatusUniteConditional(conf)
    print(model)

    # generate test data
    def _inner_dataloader(iters):
        for i in range(iters):
            input_tokens = torch.randint(low=0, high=100, size=(8, 50))
            tags = torch.randint(low=0, high=3, size=(8, 50))
            status = torch.randint(low=0, high=2, size=(8, 50))
            yield input_tokens, tags, status

    # optim and iterator
    from tqdm import tqdm
    import torch.optim as optim
    optimzer = optim.Adam(model.parameters(), lr=0.0002)

    epoch = 10
    for i in range(epoch):
        optimzer.zero_grad()
        model.train()
        pbar = tqdm(_inner_dataloader(20), total=20)
        for input_tokens, tags, status in pbar:
            output, tags_out, status_out = model(input_tokens, tags, status)
            output['loss'].backward()
            optimzer.step()
            pbar.set_description(output['description'](i, epoch))
        print(output['loss'].data)


# todo: add mask to model

if __name__ == '__main__':
    unittest()

