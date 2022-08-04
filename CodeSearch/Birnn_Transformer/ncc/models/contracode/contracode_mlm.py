# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import torch
import torch.nn as nn
from ncc.models import NccEncoderModel, register_model
from ncc.modules.code2vec.transformer_encoder import TransformerEncoder # CodeEncoderLSTM

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


@register_model('contracode_mlm')
class ContraCodeMLM(NccEncoderModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
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
            encoder = TransformerEncoder(
                task.source_dictionary,
                project=False,
            )

        elif args['model']['encoder_type'] == "lstm":
            # encoder = CodeEncoderLSTM(
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
            pass

        return cls(args, encoder)

    def forward(self, tokens, lengths):  # predicted masked tokens
        features = self.encoder(tokens, lengths)  # L x B x D #, no_project_override=True
        assert len(features.shape) == 3, str(features.shape)
        L, B, D = features.shape
        # assert D == self.d_model
        features = self.mlm_head(features).view(L, B, D)  # L x B x D
        logits = torch.matmul(features, self.encoder.embedding.weight.transpose(0, 1)).view(L, B, self.encoder.n_tokens)  # [L, B, ntok]
        return torch.transpose(logits, 0, 1).view(B, L, self.encoder.n_tokens)  # [B, T, ntok]