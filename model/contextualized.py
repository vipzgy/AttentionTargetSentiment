# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable

from .attention import Attention


class Contextualized(nn.Module):
    def __init__(self, args, m_embedding):
        super(Contextualized, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.LSTM(args.embed_dim, args.hidden_size,
                              dropout=args.dropout_rnn,
                              batch_first=True,
                              bidirectional=True)
        if args.which_init == 'xavier':
            nn.init.xavier_uniform(self.bilstm.all_weights[0][0], 1)
            nn.init.xavier_uniform(self.bilstm.all_weights[0][1], 1)
            nn.init.xavier_uniform(self.bilstm.all_weights[1][0], 1)
            nn.init.xavier_uniform(self.bilstm.all_weights[1][1], 1)
        elif args.which_init == 'kaiming':
            nn.init.kaiming_uniform(self.bilstm.all_weights[0][0], 1)
            nn.init.kaiming_uniform(self.bilstm.all_weights[0][1], 1)
            nn.init.kaiming_uniform(self.bilstm.all_weights[1][0], 1)
            nn.init.kaiming_uniform(self.bilstm.all_weights[1][1], 1)

        self.attention = Attention(args.hidden_size * 2, args.attention_size, args.cuda)
        self.attention_l = Attention(args.hidden_size * 2, args.attention_size, args.cuda)
        self.attention_r = Attention(args.hidden_size * 2, args.attention_size, args.cuda)

        self.linear = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)
        self.linear_l = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)
        self.linear_r = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)

        if args.which_init == 'xavier':
            nn.init.xavier_uniform(self.linear.weight)
            nn.init.xavier_uniform(self.linear_l.weight)
            nn.init.xavier_uniform(self.linear_r.weight)
        elif args.which_init == 'kaiming':
            nn.init.kaiming_uniform(self.linear.weight)
            nn.init.kaiming_uniform(self.linear_l.weight)
            nn.init.kaiming_uniform(self.linear_r.weight)

    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = self.dropout(x)

        x, _ = self.bilstm(x)
        x = torch.squeeze(x, 0)

        start = target_start.data[0]
        end = target_end.data[0]

        if self.args.cuda:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.cuda.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.cuda.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.cuda.LongTensor([i for i in range(end + 1, x.size(0))]))
        else:
            indices_left = None
            if not start == 0:
                indices_left = Variable(torch.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if not end == x.size(0) - 1:
                indices_right = Variable(torch.LongTensor([i for i in range(end + 1, x.size(0))]))

        left = None
        if indices_left is not None:
            left = torch.index_select(x, 0, indices_left)
        target = torch.index_select(x, 0, indices_target)
        average_target = torch.mean(target, 0)
        average_target = torch.unsqueeze(average_target, 0)
        right = None
        if indices_right is not None:
            right = torch.index_select(x, 0, indices_right)

        s = self.attention(x, average_target)
        s_l = None
        if left is not None:
            s_l = self.attention_l(left, average_target)
        s_r = None
        if right is not None:
            s_r = self.attention_r(right, average_target)

        result = self.linear(s)
        if s_l is not None:
            result = torch.add(result, self.linear_l(s_l))
        if s_r is not None:
            result = torch.add(result, self.linear_r(s_r))

        return result
