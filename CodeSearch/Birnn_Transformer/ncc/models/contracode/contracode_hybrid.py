# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import torch
import torch.nn as nn
from ncc.models import register_model
from ncc.models.contracode.contracode_moco import ContraCodeMoCo
from ncc.modules.code2vec.transformer_encoder import TransformerEncoder #CodeEncoderLSTM
DEFAULT_MAX_SOURCE_POSITIONS = 1e5


@register_model('contracode_hybrid')
class ContraCodeHybrid(ContraCodeMoCo):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, q_encoder, k_encoder):
        super().__init__(args, q_encoder, k_encoder)
        self.args = args

        # # We follow BERT's random weight initialization
        # self.apply(init_bert_params)

        # self.classification_heads = nn.ModuleDict()
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        torch.manual_seed(1)
        self.register_buffer("queue", torch.randn(128, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        torch.manual_seed(1)
        self.mlm_head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.LayerNorm(512))

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""
        max_source_positions = args['model']['max_source_positions'] if args['model'][
            'max_source_positions'] else DEFAULT_MAX_SOURCE_POSITIONS

        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        if args['model']['encoder_type'] == "transformer":
            encoder_q = TransformerEncoder(
                task.source_dictionary,
                project=True,
            )
            encoder_k = TransformerEncoder(
                task.source_dictionary,
                project=True,
            )

        elif args['model']['encoder_type'] == "lstm":
            # encoder_q = CodeEncoderLSTM(
            #     dictionary=task.source_dictionary,
            #     embed_dim=args['model']['encoder_embed_dim_q'],
            #     hidden_size=args['model']['encoder_hidden_size_q'],
            #     num_layers=args['model']['encoder_layers_q'],
            #     dropout_in=args['model']['encoder_dropout_in_q'],
            #     dropout_out=args['model']['encoder_dropout_out_q'],
            #     bidirectional=bool(args['model']['encoder_bidirectional_q']),
            #     # pretrained_embed=pretrained_encoder_embed,
            #     max_source_positions=max_source_positions
            # )
            # encoder_k = CodeEncoderLSTM(
            #     dictionary=task.source_dictionary,
            #     embed_dim=args['model']['encoder_embed_dim_k'],
            #     hidden_size=args['model']['encoder_hidden_size_k'],
            #     num_layers=args['model']['encoder_layers_k'],
            #     dropout_in=args['model']['encoder_dropout_in_k'],
            #     dropout_out=args['model']['encoder_dropout_out_k'],
            #     bidirectional=bool(args['model']['encoder_bidirectional_k']),
            #     # pretrained_embed=pretrained_encoder_embed,
            #     max_source_positions=max_source_positions
            # )
            pass
        return cls(args, encoder_q, encoder_k)

    def mlm_forward(self, tokens_q, lengths_q):  # predicted masked tokens
        features = self.encoder_q(tokens_q, lengths_q, no_project_override=True)  # L x B x D

        # features = features['encoder_out'][0]
        assert len(features.shape) == 3, str(features.shape)
        L, B, D = features.shape
        # assert D == self.d_model
        features = self.mlm_head(features).view(L, B, D)  # L x B x D
        logits = torch.matmul(features, self.encoder_q.embedding.weight.transpose(0, 1)).view(L, B,
                                                                                              self.encoder_q.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.encoder_q.n_tokens)  # [B, T, ntok]

    def moco_forward(self, tokens_q, tokens_k, lengths_q, lengths_k):  # logits, labels
        return super().forward(tokens_q, tokens_k, lengths_q, lengths_k)

    def forward(self, tokens_q, tokens_k, lengths_q, lengths_k):
        predicted_masked_tokens = self.mlm_forward(tokens_q, lengths_q)

        moco_logits, moco_targets = self.moco_forward(tokens_q, tokens_k, lengths_q, lengths_k)
        return predicted_masked_tokens, moco_logits, moco_targets

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args['model']['max_positions']
