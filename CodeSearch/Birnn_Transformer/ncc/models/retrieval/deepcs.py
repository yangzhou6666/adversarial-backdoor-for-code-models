import torch
import torch.nn as nn

from ncc.models import register_model
from ncc.models.ncc_model import NccRetrievalModel
from ncc.modules.retrieval.deepcs_encoder import (
    NBOWEncoder,
    SeqEncoder,
)

from ncc.modules.common.layers import Linear


@register_model('deepcs')
class DeepCS(NccRetrievalModel):
    def __init__(self, args, src_encoders, tgt_encoders, fusion):
        super(DeepCS, self).__init__(src_encoders, tgt_encoders)
        self.fusion = fusion
        self.args = args
        # self.reset()

    @classmethod
    def build_model(cls, args, config, task):
        code_encoders = nn.ModuleDict({
            'name': nn.ModuleList([
                SeqEncoder(task.src_dicts['name'],
                           embed_dim=args['model']['embed_dim'],
                           hidden_dim=args['model']['hidden_size'],
                           rnn_layers=args['model']['rnn_layers'],
                           bidirectional=args['model']['bidirectional'],
                           dropout=args['model']['dropout']),
                Linear(2 * args['model']['hidden_size'], args['model']['embed_dim']),
            ]),
            'apiseq': nn.ModuleList([
                SeqEncoder(task.src_dicts['apiseq'],
                           embed_dim=args['model']['embed_dim'],
                           hidden_dim=args['model']['hidden_size'],
                           rnn_layers=args['model']['rnn_layers'],
                           bidirectional=args['model']['bidirectional'],
                           dropout=args['model']['dropout']),
                Linear(2 * args['model']['hidden_size'], args['model']['embed_dim']),
            ]),
            'tokens': nn.ModuleList([
                NBOWEncoder(task.src_dicts['tokens'],
                            embed_dim=args['model']['embed_dim'],
                            dropout=args['model']['dropout']),
                Linear(args['model']['embed_dim'], args['model']['embed_dim']),
            ]),

        })
        desc_encoders = nn.ModuleDict({
            'desc': nn.ModuleList([
                SeqEncoder(task.tgt_dicts['desc'],
                           embed_dim=args['model']['embed_dim'],
                           hidden_dim=args['model']['hidden_size'],
                           rnn_layers=args['model']['rnn_layers'],
                           bidirectional=args['model']['bidirectional'],
                           dropout=args['model']['dropout']),
                Linear(2 * args['model']['hidden_size'], args['model']['embed_dim']),
            ]),
        })
        fusion = Linear(args['model']['embed_dim'], args['model']['embed_dim'])
        return cls(args, code_encoders, desc_encoders, fusion)

    def code_forward(
        self,
        name, name_len,
        apiseq, apiseq_len,
        tokens, tokens_len,
    ):
        name_out = self.src_encoders['name'][0](name, name_len)
        name_out = self.src_encoders['name'][1](name_out)
        # name_repr = self.src_encoders['name'](name, name_len)
        apiseq_out = self.src_encoders['apiseq'][0](apiseq, apiseq_len)
        apiseq_out = self.src_encoders['apiseq'][1](apiseq_out)
        # api_repr = self.src_encoders['apiseq'](api, api_len)
        tokens_out = self.src_encoders['tokens'][0](tokens, tokens_len)
        tokens_out = self.src_encoders['tokens'][1](tokens_out)
        # tokens_repr = self.src_encoders['tokens'](tokens, tokens_len)
        code_repr = self.fusion(torch.tanh(name_out + apiseq_out + tokens_out))
        return code_repr

    def desc_forward(self, desc, desc_len, ):
        desc_out = self.tgt_encoders['desc'][0](desc, desc_len)
        desc_out = self.tgt_encoders['desc'][1](desc_out)
        # desc_repr = self.tgt_encoders['desc'](desc, desc_len)
        return desc_out

    def forward(
        self,
        name, name_len,
        apiseq, apiseq_len,
        tokens, tokens_len,
        pos_desc, pos_desc_len,
        neg_desc, neg_desc_len,
    ):
        code_repr = self.code_forward(name, name_len, apiseq, apiseq_len, tokens, tokens_len)
        pos_desc_repr = self.desc_forward(pos_desc, pos_desc_len)
        neg_desc_repr = self.desc_forward(neg_desc, neg_desc_len)
        return code_repr, pos_desc_repr, neg_desc_repr
