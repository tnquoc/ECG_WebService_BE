import os
import math

import numpy as np
import pytorch_lightning as pl
import torch

from torch import nn
from torch.nn import Module
import torch.nn.functional as F


# vocab
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.weights = torch.tensor([])
        self.idx = 0

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def decode(self, word_idxs, listfy=False, join_words=True, skip_first=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            if skip_first:
                wis = wis[1:]
            for wi in wis:
                word = self.idx2word[int(wi)]
                if word == "<end>":
                    break
                caption.append(word)
            if join_words:
                caption = " ".join(caption)
            if listfy:
                caption = [caption]
            captions.append(caption)
        return captions


# util
# def get_next_word(logits):
#     probs = F.softmax(logits, dim=-1)
#     logprobs = F.log_softmax(logits, dim=-1)
#
#     next_probs, next_tokens = probs.topk(1)
#     next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))
#
#     return next_tokens.squeeze(1), next_logprobs


def get_next_word(logits, temp=None, k=None, p=None, greedy=None, m=None):
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    if temp is not None:
        samp_probs = F.softmax(logits.div_(temp), dim=-1)
    else:
        samp_probs = probs.clone()

    if greedy:
        next_probs, next_tokens = probs.topk(1)
        if next_tokens.shape[0] == 1:
            next_tokens = next_tokens.unsqueeze(0)
            logprobs = logprobs.unsqueeze(0)
        next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))

    elif k is not None:
        indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
        samp_probs[indices_to_remove] = 0
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1 - m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()

    elif p is not None:
        sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        if m is not None:
            sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
            sorted_samp_probs.mul_(1 - m)
            sorted_samp_probs.add_(sorted_probs.mul(m))
        sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
        next_tokens = sorted_indices.gather(1, sorted_next_indices)
        next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

    else:
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1 - m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
    return next_tokens.squeeze(1), next_logprobs


# model
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class ECGResNet(nn.Module):
    """
    This class implements the ECG-ResNet in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ResNet object can perform forward.
    """

    def __init__(
        self,
        in_channels,
        n_grps,
        N,
        num_classes,
        dropout,
        first_width,
        stride,
        dilation,
    ):
        """
        Initializes ECGResNet object.

        Args:
            in_length (int): the length of the ECG signal input.
            in_channels (int): number of channels of input (= leads).
            n_grps (int): number of ResNet groups.
            N (int): number of blocks per groups.
            num_classes (int): number of classes of the classification problem.
            stride (tuple): tuple with stride value per block per group.
            dropout (float): the dropout probability.
            first_width (int): the output width of the stem.
            dilation (int): the space between the dilated convolutions.
        """
        super().__init__()
        num_branches = 2
        first_width = first_width * num_branches
        stem = [
            nn.Conv1d(
                in_channels,
                first_width // 2,
                kernel_size=7,
                padding=3,
                stride=2,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(first_width // 2),
            nn.ReLU(),
            nn.Conv1d(
                first_width // 2,
                first_width,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(first_width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                first_width, first_width, kernel_size=5, padding=2, stride=1, bias=False
            ),
        ]

        layers = []

        # Double feature depth at each group, after the first
        widths = [first_width]
        for grp in range(n_grps):
            widths.append((first_width) * 2**grp)
        for grp in range(n_grps):
            layers += self._make_group(
                N, widths[grp], widths[grp + 1], stride, dropout, dilation
            )

        layers += [nn.BatchNorm1d(widths[-1]), nn.ReLU(inplace=True)]

        fclayers1 = [
            nn.Linear(20096, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        ]
        fclayers2 = [
            nn.Linear(5120, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        ]

        self.stem = nn.Sequential(*stem)
        aux_point = (len(layers) - 2) // 2
        self.features1 = nn.Sequential(*layers[:aux_point])
        self.features2 = nn.Sequential(*layers[aux_point:])
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(*fclayers1)
        self.fc2 = nn.Sequential(*fclayers2)

    def _make_group(self, N, in_channels, out_channels, stride, dropout, dilation):
        """
        Builds a group of blocks.

        Args:
            in_channels (int): number of channels of input
            out_channels (int): number of channels of output
            stride (tuple): tuple of strides of convolutions with length of N
            N (int): number of blocks per groups
            num_classes (int): number of classes of the classification problem
            dropout (float): the dropout probability.
        """
        group = list()
        for i in range(N):
            blk = BasicBlock(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                stride=stride[i],
                dropout=dropout,
                dilation=dilation,
            )
            group.append(blk)
        return group

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
            x (tensor): input to the block with size NxCxL
        Returns:
            out (tuple): outputs of forward pass, first the auxilliary halfway,
                then the final prediction
        """
        # print(x.shape)
        x = self.stem(x)
        x1 = self.features1(x)
        x1out = self.flatten(x1)
        x2 = self.features2(x1)
        x2out = self.flatten(x2)
        return self.fc1(x1out), self.fc2(x2out)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    """
    This class implements a residual block.
    """

    def __init__(self, in_channels, out_channels, stride, dropout, dilation):
        """
        Initializes BasicBlock object.

        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            stride (int): stride of the convolution.
            dropout (float): probability of an argument to get zeroed in the
                dropout layer.
            dilation (int): amount of dilation in the dilated branch.
        """
        super(BasicBlock, self).__init__()
        kernel_size = 5
        num_branches = 2

        self.branch0 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels,
                in_channels // num_branches,
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels // num_branches,
                out_channels // num_branches,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=stride,
                bias=False,
            ),
        )

        self.branch1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels,
                in_channels // num_branches,
                kernel_size=1,
                padding=0,
                stride=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels // num_branches),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels // num_branches,
                out_channels // num_branches,
                kernel_size=kernel_size,
                padding=((kernel_size - 1) * dilation) // 2,
                stride=stride,
                dilation=dilation,
                bias=False,
            ),
        )

        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is
        transformed through several layer transformations.

        Args:
          x (tensor): input to the block with size NxCxL
        Returns:
          out: outputs of the block
        """
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        r = self.shortcut(x)
        return out.add_(r)


class AveragePool(nn.Module):
    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(self, x):
        signal_size = x.shape[-1]
        kernel = torch.nn.AvgPool1d(signal_size)
        average_feature = kernel(x).squeeze(-1)
        return x, average_feature


class MLC(nn.Module):
    def __init__(
        self, classes=156, sementic_features_dim=512, fc_in_features=2048, k=10
    ):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(
        self, embed_size=512, hidden_size=512, visual_size=2048, k=10, momentum=0.1
    ):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(
            in_features=visual_size + hidden_size, out_features=embed_size
        )
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, avg_features, semantic_features):
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TopicTransformerModule(Module):
    def __init__(self, d_model, nhead, num_layers, mlc, attention):
        super(TopicTransformerModule, self).__init__()

        self.positional_encoding = PositionalEncoding(2 * d_model)
        self.positional_transformer = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, norm=None
        )

        self.mlc = mlc
        self.attention = attention

        decoder_layer = nn.TransformerDecoderLayer(d_model=2 * d_model, nhead=nhead)
        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers, norm=None
        )

    def forward_one_step(self, image_features, avg_feats, tgt, attended_features=None):
        if attended_features is None:
            image_features = self.positional_transformer(image_features)
            attended_features = self.transformer_encoder(image_features)

            # attended : (batch, num_features, feature_size)
            def forward_attention(mlc, co_att, avg_features):
                tags, semantic_features = mlc.forward(avg_features)
                ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)
                return tags, ctx

            tags, ctx = forward_attention(self.mlc, self.attention, avg_feats)

            contexts = ctx.unsqueeze(0).repeat(40, 1, 1)
            attended_features = torch.cat([attended_features, contexts], dim=2)

        tgt = self.positional_encoding(tgt)
        out = self.transformer_decoder(tgt, attended_features)
        return out, attended_features


class TopicTransformer(pl.LightningModule):
    def __init__(
        self,
        vocab,
        in_channels,
        n_grps,
        N,
        num_classes,
        k,
        dropout,
        first_width,
        stride,
        dilation,
        num_layers,
        d_mode,
        nhead,
    ):
        super().__init__()
        self.vocab_length = len(vocab)
        self.vocab = vocab
        self.save_hyperparameters()
        self.model = ECGResNet(
            in_channels, n_grps, N, num_classes, dropout, first_width, stride, dilation
        )

        self.model.flatten = Identity()
        self.model.fc1 = AveragePool()
        self.model.fc2 = AveragePool()

        self.feature_embedding = nn.Linear(256, d_mode)
        self.embed = nn.Embedding(len(vocab), 2 * d_mode)

        mlc = MLC(
            classes=num_classes, sementic_features_dim=d_mode, fc_in_features=256, k=k
        )
        attention = CoAttention(
            embed_size=d_mode, hidden_size=d_mode, visual_size=256, k=k
        )

        self.transformer = TopicTransformerModule(
            d_mode, nhead, num_layers, mlc, attention
        )

        self.to_vocab = nn.Sequential(nn.Linear(2 * d_mode, len(vocab)))

    def sample(self, waveforms, max_length):
        # _, (image_features, avg_feats) = self.model(waveforms.cuda())
        _, (image_features, avg_feats) = self.model(waveforms)
        image_features = image_features.transpose(1, 2).transpose(
            1, 0
        )  # ( batch, feature, number)
        image_features = self.feature_embedding(image_features)

        start_tokens = torch.tensor(
            [self.vocab("<start>")], device=image_features.device
        )
        nb_batch = waveforms.shape[0]
        start_tokens = start_tokens.repeat(nb_batch, 1)
        sent = self.embed(start_tokens).transpose(1, 0)

        attended_features = None

        tgt_mask = torch.zeros(
            sent.shape[1], sent.shape[0], device=image_features.device, dtype=bool
        )
        y_out = torch.zeros(nb_batch, max_length, device=image_features.device)

        for i in range(max_length):
            out, attended_features = self.transformer.forward_one_step(
                image_features, avg_feats, sent, attended_features=attended_features
            )
            out = self.to_vocab(out[-1, :, :]).squeeze(0)

            if len(out.shape) == 1:
                out = out[None, :]
            word_idx, props = get_next_word(out)
            y_out[:, i] = word_idx

            ended_mask = (
                tgt_mask[:, -1] | (word_idx == self.vocab("<end>"))
            ).unsqueeze(1)
            tgt_mask = torch.cat((tgt_mask, ended_mask), dim=1)

            embedded = self.embed(word_idx).unsqueeze(0)
            sent = torch.cat((sent, embedded), dim=0)

            if ended_mask.sum() == nb_batch:
                break

        return y_out
