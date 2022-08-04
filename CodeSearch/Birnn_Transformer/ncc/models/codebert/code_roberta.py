# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ncc.utils import utils
from ncc.models import NccLanguageModel, register_model
from ncc.modules.seq2seq.ncc_decoder import NccDecoder
from ncc.modules.roberta.layer_norm import LayerNorm
# from ncc.modules.roberta.transformer_sentence_encoder import init_bert_params
from ncc.modules.embedding import Embedding
# from ncc.models.hub_interface import RobertaHubInterface
from ncc import LOGGER
from ncc.modules.code2vec.transformer_encoder import TransformerEncoder
DEFAULT_MAX_SOURCE_POSITIONS = 1e5


@register_model('code_roberta')
class CodeRobertaModel(NccLanguageModel):

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

        # # We follow BERT's random weight initialization
        # self.apply(init_bert_params)  # TODO: Warning, currently commented for debug

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, config, task):
        """Build a new model instance."""

        # make sure all arguments are present
        # base_architecture(args)

        if 'max_positions' not in args['model']:
            args['model']['max_positions'] = args['task']['tokens_per_sample']

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None,
                 **kwargs):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.decoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                LOGGER.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        # self.classification_heads[name] = RobertaClassificationHead(
        #     self.args['model']['encoder_embed_dim'],
        #     inner_dim or self.args['model']['encoder_embed_dim'],
        #     num_classes,
        #     self.args['task']['pooler_activation_fn'],
        #     self.args['model']['pooler_dropout'],
        # )

    @property
    def supported_targets(self):
        return {'self'}

    # @classmethod
    # def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2',
    #                     **kwargs):
    #     from ncc.utils import hub_utils
    #     x = hub_utils.from_pretrained(
    #         model_name_or_path,
    #         checkpoint_file,
    #         data_name_or_path,
    #         archive_map=cls.hub_models(),
    #         bpe=bpe,
    #         load_checkpoint_heads=True,
    #         **kwargs,
    #     )
    #     return RobertaHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            # if getattr(self.args, 'load_checkpoint_heads', False):
            if 'load_checkpoint_heads' in self.args['model']:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    LOGGER.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    LOGGER.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    LOGGER.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

    def upgrade_state_dict_named_for_summarization(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            # if getattr(self.args, 'load_checkpoint_heads', False):
            if 'load_checkpoint_heads' in self.args['model']:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    LOGGER.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    LOGGER.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    LOGGER.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        torch.manual_seed(1)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.ReLU() #utils.get_activation_fn(activation_fn) TODO
        self.layer_norm = nn.LayerNorm(embed_dim) # LayerNorm(embed_dim) # TODO

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        # self.bias = nn.Parameter(torch.zeros(output_dim)) # TODO

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight)    # + self.bias # TODO
        return x.transpose(0, 1)    # (B x T x C)


class RobertaEncoder(NccDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.NccDecoder` interface required
    by :class:`~fairseq.models.NccLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        # RoBERTa is a sentence encoder model, so users will intuitively trim
        # encoder layers. However, the implementation uses the fairseq decoder,
        # so we fix here.
        if args['model']['encoder_layers_to_keep']:
            args['model']['encoder_layers'] = len(args['model']['encoder_layers_to_keep'].split(","))
            args['model']['decoder_layers_to_keep'] = args['model']['encoder_layers_to_keep']
            args['model']['encoder_layers_to_keep'] = None

        # self.sentence_encoder = TransformerSentenceEncoder(
        #     padding_idx=dictionary.pad(),
        #     vocab_size=len(dictionary),
        #     num_encoder_layers=args['model']['encoder_layers'],
        #     embedding_dim=args['model']['encoder_embed_dim'],
        #     ffn_embedding_dim=args['model']['encoder_ffn_embed_dim'],
        #     num_attention_heads=args['model']['encoder_attention_heads'],
        #     dropout=args['model']['dropout'],
        #     attention_dropout=args['model']['attention_dropout'],
        #     activation_dropout=args['model']['activation_dropout'],
        #     layerdrop=args['model']['encoder_layerdrop'],
        #     max_seq_len=args['model']['max_positions'],
        #     num_segments=0,
        #     encoder_normalize_before=True,
        #     apply_bert_init=True,
        #     activation_fn=args['model']['activation_fn'],
        # )
        if args['model']['max_source_positions'] is None:
            args['model']['max_source_positions'] = DEFAULT_MAX_SOURCE_POSITIONS

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()

            # emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # # emb = nn.Embedding(num_embeddings, embed_dim)
            # # if provided, load from preloaded dictionaries
            # if path:
            #     embed_dict = utils.parse_embedding(path)
            #     utils.load_embedding(embed_dict, dictionary, emb)
            torch.manual_seed(1)
            emb = nn.Embedding(num_embeddings, embed_dim)
            return emb

        embed_tokens = build_embedding(
            dictionary, args['model']['encoder_embed_dim'], args['model']['encoder_embed_path']
        )
        self.code_encoder = TransformerEncoder(args, dictionary, embed_tokens, num_segments=0)
        # self.code_encoder = CodeEncoderTransformer(
        #         dictionary,
        #         project=False,
        #     )
        self.lm_head = RobertaLMHead(
            embed_dim=args['model']['encoder_embed_dim'],
            output_dim=len(dictionary),
            activation_fn=args['model']['activation_fn'],
            weight=self.code_encoder.embed_tokens.weight, #embed_tokens
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        # inner_states, _ = self.code_encoder(
        #     src_tokens,
        #     # last_state_only=not return_all_hiddens,
        # )
        # features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        encoder_out = self.code_encoder(
            src_tokens,
            # last_state_only=not return_all_hiddens,
        )
        features = encoder_out.encoder_out #.transpose(0, 1)
        inner_states = encoder_out.encoder_states
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args['model']['max_positions']
