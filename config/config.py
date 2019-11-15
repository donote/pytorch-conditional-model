import json

from config.configuration import BasicConfig


class ModelConfigNERStatus(BasicConfig):

    def __init__(self, kwargs):
        super(ModelConfigNERStatus, self).__init__(**kwargs)
        model_config_json = kwargs.get('model_config_json')
        if model_config_json is not None:
            with open(model_config_json, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            self.vocab_size = kwargs.get('vocab_size')
            self.embedding_dim = kwargs.get('embedding_dim')
            self.hidden_size = kwargs.get('hidden_size')
            self.num_layers = kwargs.get('num_layers')
            self.tag_size = kwargs.get('tag_size')
            self.status_size = kwargs.get('status_size')
            self.fc_dim = kwargs.get('fc_dim')
            self.n_encoder = kwargs.get('n_encoder')
            self.positional_embedding_length = kwargs.get('positional_embedding_length')
            self.type_embedding_num = kwargs.get('type_embedding_num')
            self.n_head = kwargs.get('n_head')
            self.type_emb = kwargs.get('type_emb')
            self.pos_emb = kwargs.get('pos_emb')
            self.dropout = kwargs.get('dropout')

