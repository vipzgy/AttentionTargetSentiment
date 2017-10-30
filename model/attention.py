# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    """
    这一个方面，是把target写成了embedding的平均
    所以用了LSTMCell
    """
    def __init__(self, args, m_embedding):
        super(Attention, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(args.embed_num, args.embed_dim, max_norm=args.max_norm)
        if args.use_embedding:
            self.embedding.weight.data.copy_(m_embedding)

        self.lstmcell_l = nn.LSTMCell(args.embed_dim, args.hidden_size)
        self.lstmcell_r = nn.LSTMCell(args.embed_dim, args.hidden_size)

        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size, bias=True)
        self.u = Variable(torch.randn(1, args.hidden_size), requires_grad=True)

        self.linear_2 = nn.Linear(args.hidden_size, args.label_num, bias=True)

    # 试一试batch=1.not use att
    def forward(self, x, target_start, target_end):
        x = self.embedding(x)
        x = torch.squeeze(x, 0)

        ht = Variable(torch.zeros(1, self.args.hidden_size))
        ct = Variable(torch.zeros(1, self.args.hidden_size))
        if self.args.cuda:
            ht.cuda()
            ct.cuda()

        """
        这里就遇到了一个问题，因为每个句子的target的位置是不一致的，所以
        没有办法batch操作了吧。。。。
        可以试试这里调用多线程，但是其他的地方没用
        """
        start = target_start.data[0]
        end = target_end.data[0]
        target = x[target_start]
        i = start + 1
        while i <= end:
            if i > x.size(0):
                exit()
            target = target + x[i]
            i = i + 1
        target = target / (end - start + 1)

        # output就是经过lstm的向量表达
        output_l = None
        ist = True
        for idx in range(x.size(0)):
            if (idx < start or idx > end) and ist:
                ht, ct = self.lstmcell_l(x[idx], (ht, ct))
                ist = False
            else:
                ht, ct = self.lstmcell_l(target, (ht, ct))
            if idx == 0:
                output_l = ht
            else:
                output_l = torch.cat([output_l, ht], 0)

        output_r = None
        ist = True
        idx = x.size(0) - 1
        while idx >= 0:
            if (idx < start or idx > end) and ist:
                ht, ct = self.lstmcell_r(x[idx], (ht, ct))
                ist = False
            else:
                ht, ct = self.lstmcell_r(target, (ht, ct))
            if idx == x.size(0) - 1:
                output_r = ht
            else:
                output_r = torch.cat([output_r, ht], 0)
            idx -= 1

        output = torch.mul(output_l, output_r)

        t = F.tanh(self.linear_1(output))
        t = torch.t(t)
        beta = torch.mm(self.u, t)
        alfa = F.softmax(beta)
        s = torch.mm(alfa, output)

        output = self.linear_2(s)

        return output
