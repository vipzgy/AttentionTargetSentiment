# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .att import Att
from torch.autograd import Variable
from torch.nn import Parameter


class Attention2(nn.Module):
    """
    这个就是正常的先过了lstm之后的，在求target的方法
    batch还是==1的
    """
    def __init__(self, args, m_embedding):
        super(Attention2, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_size,
                              dropout=args.dropout_rnn,
                              batch_first=True,
                              bidirectional=True)

        self.att0 = Att(args.hidden_size * 2, args.attention_size, args.cuda)

        self.linear_2 = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)
        nn.init.xavier_uniform(self.linear_2.weight)

    # batch=1,使用att
    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.bilstm(x)

        x = torch.squeeze(x, 0)

        start = target_start.data[0]
        end = target_end.data[0]

        if self.args.cuda:
            indices = Variable(torch.cuda.LongTensor([i for i in range(start, end + 1)]))
        else:
            indices = Variable(torch.LongTensor([i for i in range(start, end + 1)]))

        target = torch.index_select(x, 0, indices)
        average_target = torch.mul(torch.sum(target, 0), 1 / (end - start + 1))
        average_target = torch.unsqueeze(average_target, 0)

        s = self.att0(x, average_target)

        output = self.linear_2(s)

        return output
