# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention4(nn.Module):
    """
    这一种是加了batch的，但是算target还是要一个一个的算的，所以仍然不是多线程的，效率并没有提升多高
    """
    def __init__(self, args, m_embedding):
        super(Attention4, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)
        self.dropout = nn.Dropout(args.dropout_embed)

        self.bilstm = nn.GRU(args.embed_dim, args.hidden_size,
                              dropout=args.dropout_rnn,
                              batch_first=True,
                              bidirectional=True)
        nn.init.kaiming_uniform(self.bilstm.all_weights[0][0])
        nn.init.kaiming_uniform(self.bilstm.all_weights[0][1])
        nn.init.kaiming_uniform(self.bilstm.all_weights[1][0])
        nn.init.kaiming_uniform(self.bilstm.all_weights[1][1])

        self.linear_1 = nn.Linear(args.hidden_size * 4, args.attention_size, bias=True)
        nn.init.kaiming_uniform(self.linear_1.weight)

        self.u = Variable(torch.randn(args.attention_size, args.attention_size), requires_grad=True)
        nn.init.kaiming_uniform(self.u)

        self.linear_2 = nn.Linear(args.hidden_size * 2, args.label_num, bias=True)
        nn.init.kaiming_uniform(self.linear_2.weight)

    # batch > 1,使用attbatch
    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = self.dropout(x)

        x, _ = self.bilstm(x)

        target = None
        for idx in range(x.size(0)):
            sequence = x[idx]
            start = target_start.data[idx]
            end = target_end.data[idx]
            tar = sequence[start]
            idj = start + 1
            while idj <= end:
                if idj > sequence.size(0):
                    exit()
                tar = tar + sequence[idj]
                idj += 1
            tar = tar / (end - start + 1)
            if idx == 0:
                target = torch.unsqueeze(tar, 0)
            else:
                target = torch.cat([target, torch.unsqueeze(tar, 0)], 0)

        x = torch.transpose(x, 0, 1)
        output = None
        for idx in range(x.size(0)):
            if idx == 0:
                output = torch.unsqueeze(torch.cat([x[idx], target], 1), 0)
            else:
                output = torch.cat([output, torch.unsqueeze(torch.cat([x[idx], target], 1), 0)], 0)
        output = torch.transpose(output, 0, 1)
        x = torch.transpose(x, 0, 1)

        output = F.tanh(self.linear_1(output))
        beta = None
        for idx in range(output.size(0)):
            if idx == 0:
                beta = torch.unsqueeze(torch.mm(self.u, torch.t(output[idx])), 0)
            else:
                beta = torch.cat([beta, torch.unsqueeze(torch.mm(self.u, torch.t(output[idx])), 0)], 0)
        alfa = F.softmax(beta)

        result = None
        for idx in range(x.size(0)):
            if idx == 0:
                result = torch.mm(torch.unsqueeze(alfa[idx], 0), x[idx])
            else:
                result = torch.cat([result, torch.mm(torch.unsqueeze(alfa[idx], 0), x[idx])], 0)

        result = self.linear_2(result)
        return result
