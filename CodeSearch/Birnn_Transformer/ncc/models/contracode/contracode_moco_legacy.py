# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import torch
import torch.nn as nn

from ncc.models import (
    NccMoCoModel,
    register_model,
    # register_model_architecture,
)

from ncc.models.hub_interface import RobertaHubInterface
from ncc.modules.code2vec.contracode_encoder_bak import CodeEncoderLSTM, CodeEncoderTransformer

DEFAULT_MAX_SOURCE_POSITIONS = 1e5


@register_model('contracode_moco_legacy')
class ContraCodeMoCoLegacy(NccMoCoModel):

    @classmethod
    def hub_models(cls):
        return {
            'roberta.base': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz',
            'roberta.large': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz',
            'roberta.large.mnli': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz',
            'roberta.large.wsc': 'http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz',
        }

    def __init__(self, args, q_encoder, k_encoder):
        super().__init__(q_encoder, k_encoder)
        self.args = args
        self.K = args['model']['moco_num_keys']
        self.m = args['model']['moco_momentum']
        self.T = args['model']['moco_temperature']

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        torch.manual_seed(1)
        self.register_buffer("queue", torch.randn(128, self.K)) #args['model']['encoder_hidden_size_q']
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present
        # base_architecture(args)
        max_source_positions = args['model']['max_source_positions'] if args['model'][
            'max_source_positions'] else DEFAULT_MAX_SOURCE_POSITIONS

        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        if args['model']['encoder_type'] == "transformer":
            encoder_q = CodeEncoderTransformer(
                task.source_dictionary,
                project=True,
            )
            encoder_k = CodeEncoderTransformer(
                task.source_dictionary,
                project=True,
            )

        elif args['model']['encoder_type'] == "lstm":
            encoder_q = CodeEncoderLSTM(
                dictionary=task.source_dictionary,
                embed_dim=args['model']['encoder_embed_dim_q'],
                hidden_size=args['model']['encoder_hidden_size_q'],
                num_layers=args['model']['encoder_layers_q'],
                dropout_in=args['model']['encoder_dropout_in_q'],
                dropout_out=args['model']['encoder_dropout_out_q'],
                bidirectional=bool(args['model']['encoder_bidirectional_q']),
                # pretrained_embed=pretrained_encoder_embed,
                max_source_positions=max_source_positions
            )
            encoder_k = CodeEncoderLSTM(
                dictionary=task.source_dictionary,
                embed_dim=args['model']['encoder_embed_dim_k'],
                hidden_size=args['model']['encoder_hidden_size_k'],
                num_layers=args['model']['encoder_layers_k'],
                dropout_in=args['model']['encoder_dropout_in_k'],
                dropout_out=args['model']['encoder_dropout_out_k'],
                bidirectional=bool(args['model']['encoder_bidirectional_k']),
                # pretrained_embed=pretrained_encoder_embed,
                max_source_positions=max_source_positions
            )

        return cls(args, encoder_q, encoder_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def embed(self, img):
        return self.encoder_q(img)

    def forward(self, tokens_q, tokens_k, lengths_k, lengths_q): # TODO: Note: not lengths_q, lengths_k
        """
        Input:
            tokens_q: a batch of query images
            tokens_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(tokens_q, lengths_q)  # queries: NxC
        # q = output_q    # ['encoder_out'][0]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # tokens_k, idx_unshuffle = self._batch_shuffle_ddp(tokens_k)

            k = self.encoder_k(tokens_k, lengths_k)  # keys: NxC
            # k = output_k    # ['encoder_out'][0]
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        # q=k
        l_pos = torch.einsum("nc,nc->n", *[q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", *[q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None,
                 **kwargs):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2',
                        **kwargs):
        from ncc.utils import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x['args'], x['task'], x['models'][0])


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    #
    # output = torch.cat(tensors_gather, dim=0)
    output = tensor
    return output
