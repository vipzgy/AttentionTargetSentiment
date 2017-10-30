# -*- coding: utf-8 -*-

import os
import re
import random
import pickle
import numpy
import torch
from torch.autograd import Variable

random.seed(66)


class Example(object):
    def __init__(self, sequence, target_start, target_end, label):
        self.sequence = sequence
        self.target_start = target_start
        self.target_end = target_end
        self.label = label


class Examples(object):
    def __init__(self, path, shuffle=True, if_re=False):
        self.examples = []
        f = open(path, encoding='utf-8').readlines()
        sequence = []
        target_start = -1
        target_end = -1
        label = None
        count = 0
        isf = False
        for line in f:
            line = line.strip()
            if line == "" or len(line) == 0:
                if isf:
                    target_end = count - 1
                    isf = False
                self.examples.append(Example(sequence, target_start, target_end, label))
                sequence = []
                count = 0
            else:
                strs = line.split(' ')

                if if_re:
                    input = self.nom(strs[0])
                    ss = input.split(' ')
                    for idx in range(len(ss)):
                        sequence.append(self.nom(ss[idx]))
                        # print(self.nom(ss[idx]))
                        # print("sdf")
                else:
                    sequence.append(strs[0])

                if strs[1] == 'o':
                    if isf:
                        target_end = count - 1
                        isf = False
                else:
                    if strs[1][0] == 'b':
                        target_start = count
                        label = strs[1][2:]
                        isf = True
                count += 1

        if shuffle:
            random.shuffle(self.examples)

    def nom(self, m_input):
        string = m_input.lower()
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

        string = re.sub(r"^-user-$", "<user>", string)

        string = re.sub(r"^-url-$", "<url>", string)

        string = re.sub(r"^-lqt-$", "\'", string)
        string = re.sub(r"^-rqt-$", "\'", string)
        # 或者
        # string = re.sub(r"^-lqt-$", "\"", string)
        # string = re.sub(r"^-rqt-$", "\"", string)

        string = re.sub(r"^-lrb-$", "\(", string)
        string = re.sub(r"^-rrb-$", "\)", string)

        string = re.sub(r"^lol$", "<lol>", string)
        string = re.sub(r"^<3$", "<heart>", string)

        string = re.sub(r"^#.*", "<hashtag>", string)
        string = re.sub(r"^[0-9]*$", "<number>", string)

        string = re.sub(r"^\:\)$", "<smile>", string)
        string = re.sub(r"^\;\)$", "<smile>", string)
        string = re.sub(r"^\:\-\)$", "<smile>", string)
        string = re.sub(r"^\;\-\)$", "<smile>", string)
        string = re.sub(r"^\;\'\)$", "<smile>", string)
        string = re.sub(r"^\(\:$", "<smile>", string)

        string = re.sub(r"^\)\:$", "<sadface>", string)
        string = re.sub(r"^\)\;$", "<sadface>", string)
        string = re.sub(r"^\:\($", "<sadface>", string)
        return string.strip()


class Vocabulary(object):
    def __init__(self, id2word, word2id):
        self.id2word = id2word
        self.word2id = word2id

    @classmethod
    def makeVocabularyByText(cls, examplesAll):
        frequence = dict()
        id2word = {}
        word2id = {}
        for examples in examplesAll:
            for e in examples:
                for word in e.sequence:
                    if word in frequence:
                        frequence[word] += 1
                    else:
                        frequence[word] = 1
        allwords = sorted(frequence.items(), key=lambda t: t[1], reverse=True)
        id2word[0] = "<unknown>"
        word2id["<unknown>"] = 0
        id2word[1] = "<padding>"
        word2id["<padding>"] = 1
        for idx, word in enumerate(allwords):
            id2word[idx + 2] = word[0]
            word2id[word[0]] = idx + 2
        return cls(id2word, word2id)

    @classmethod
    def makeVocabularyByLable(cls, examplesAll):
        frequence = dict()
        id2word = {}
        word2id = {}
        for examples in examplesAll:
            for e in examples:
                if e.label in frequence:
                    frequence[e.label] += 1
                else:
                    frequence[e.label] = 1
        allwords = sorted(frequence.items(), key=lambda t: t[1], reverse=True)
        for idx, word in enumerate(allwords):
            id2word[idx] = word[0]
            word2id[word[0]] = idx
        return cls(id2word, word2id)


def makeMiniEmbed(save_dir, rawembed_path, name, word2id):
    assert os.path.isdir(save_dir)
    assert os.path.isfile(rawembed_path)
    output = open(os.path.join(save_dir, name), "w+", encoding='utf-8')
    with open(rawembed_path, encoding="utf-8") as f:
        hang = 0
        count = 0
        find = 0
        for line in f:
            line = line.strip()
            if hang == 0 or line == '' or len(line) == 0:
                hang = hang + 1
            else:
                strs = line.split(' ')
                if strs[0] in word2id:
                    output.write(line + '\n')
                    output.flush()
                    find += 1
                count += 1
    output.close()
    print("find:", find)
    print("word", len(word2id))
    print("all:", count)


def createEmbedding(embed_dim, plk_path, embed_path, id2word, name):
    if os.path.isfile(plk_path):
        plk_f = open(plk_path, 'rb+')
        m_embed = pickle.load(plk_f)
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)
        plk_f.close()
    else:
        assert os.path.isfile(embed_path)
        embed_f = open(embed_path, encoding="utf-8")
        m_dict = {}
        for idx, line in enumerate(embed_f.readlines()):
            if not (line == '' or len(line) == 0):
                strs = line.split(' ')
                m_dict[strs[0]] = [float(i) for idx2, i in enumerate(strs) if not idx2 == 0]
        embed_f.close()

        m_embed = []
        notfound = 0
        for idx in range(len(id2word)):
            if id2word[idx] in m_dict:
                m_embed.append(m_dict[id2word[idx]])
            else:
                notfound += 1
                if idx == 1:
                    m_embed.append([0 for i in range(embed_dim)])
                else:
                    m_embed.append([round(random.uniform(-0.25, 0.25), 6) for i in range(embed_dim)])

        print('all', len(id2word))
        print('notfound:', notfound)
        print('ratio:', notfound / len(id2word))
        m_embedding = torch.from_numpy(numpy.array(m_embed)).type(torch.DoubleTensor)

        f = open(os.path.join('./data', name), 'wb+')
        pickle.dump(m_embed, f)
        f.close()
    return m_embedding


class Batch(object):
    def __init__(self, text, label, target_start, target_end):
        self.text = text
        self.label = label
        self.target_start = target_start
        self.target_end = target_end
        self.batch_size = len(text)

    @classmethod
    def makeBatch(cls, item, shuffle=False):
        max_len = 0
        for i in item:
            if len(i[0]) > max_len:
                max_len = len(i[0])
        text = []
        label = []
        target_start = []
        target_end = []

        if shuffle:
            random.shuffle(item)

        for i in item:
            seq = []
            for idx in range(max_len):
                if idx < len(i[0]):
                    seq.append(i[0][idx])
                else:
                    seq.append(1)
            text.append(seq)
            label.append(i[1])
            target_start.append(i[2])
            target_end.append(i[3])

        text = Variable(torch.LongTensor(text))
        label = Variable(torch.LongTensor(label))
        target_start = Variable(torch.LongTensor(target_start))
        target_end = Variable(torch.LongTensor(target_end))
        return cls(text, label, target_start, target_end)


class MyIterator(object):
    def __init__(self, batch_size, examples, vocabulary_text, vocabulary_label,
                 shuffle_batch=False, shuffle_iterators=False):
        self.iterators = []
        item = []
        count = 0
        for example in examples:
            text = []
            for word in example.sequence:
                if word in vocabulary_text.word2id:
                    text.append(vocabulary_text.word2id[word])
                else:
                    text.append(0)

            item.append((text, vocabulary_label.word2id[example.label], example.target_start, example.target_end))
            count += 1
            if count % batch_size == 0 or count == len(examples):
                self.iterators.append(Batch.makeBatch(item, shuffle=shuffle_batch))
                item = []

        if shuffle_iterators:
            random.shuffle(self.iterators)


class MyDatasets(object):
    def __init__(self, args, path, train=None):

        shuffle_example = True
        shuffle_batch = True
        shuffle_iterators = True

        self.examples = Examples(path, shuffle_example, if_re=args.if_re).examples

        # 测试一下对不对
        # pos_count = 0
        # neg_count = 0
        # neu_count = 0
        # for e in self.examples:
        #     if e.label == 'positive':
        #         pos_count += 1
        #     elif e.label == 'negative':
        #         neg_count += 1
        #     elif e.label == 'neutral':
        #         neu_count += 1
        # print(pos_count)
        # print(neg_count)
        # print(neu_count)

        if train is None:
            self.vocabulary_text = Vocabulary.makeVocabularyByText([self.examples])
            self.vocabulary_label = Vocabulary.makeVocabularyByLable([self.examples])

            # 输出一下词的个数
            # print(len(self.vocabulary_text.word2id))

            # 生成小的词向量表
            if args.need_smallembed:
                if args.which_embedding == '200d':
                    makeMiniEmbed("./data",
                                  'D:/AI/embedding&corpus/glove.6B.200d.txt',
                                  'glove.6B.200d.txt',
                                  self.vocabulary_text.word2id)
                elif args.which_embedding == '300d':
                    makeMiniEmbed("./data",
                                  'D:/AI/embedding&corpus/glove.840B.300d.txt',
                                  'glove.840B.300d.txt',
                                  self.vocabulary_text.word2id)
                elif args.which_embedding == '200dt':
                    makeMiniEmbed("./data",
                                  'D:/AI/embedding&corpus/glove.twitter.27B.200d.txt',
                                  'glove.twitter.27B.200d.txt',
                                  self.vocabulary_text.word2id)

            if args.use_embedding:
                if args.which_embedding == '200d':
                    id2word = self.vocabulary_text.id2word
                    self.embedding = createEmbedding(args.embed_dim,
                                                     './data/6B.200d.pkl',
                                                     './data/glove.6B.200d.txt',
                                                     id2word,
                                                     '6B.200d.pkl')
                elif args.which_embedding == '300d':
                    id2word = self.vocabulary_text.id2word
                    self.embedding = createEmbedding(args.embed_dim,
                                                     './data/840B.300d.pkl',
                                                     './data/glove.840B.300d.txt',
                                                     id2word,
                                                     '840B.300d.pkl')
                elif args.which_embedding == '200dt':
                    id2word = self.vocabulary_text.id2word
                    self.embedding = createEmbedding(args.embed_dim,
                                                     './data/twitter.27B.200d.pkl',
                                                     './data/glove.twitter.27B.200d.txt',
                                                     id2word,
                                                     'twitter.27B.200d.pkl')

            self.iterator = MyIterator(args.batch_size, self.examples, self.vocabulary_text, self.vocabulary_label,
                                       shuffle_batch=shuffle_batch, shuffle_iterators=shuffle_iterators).iterators
        else:
            self.iterator = MyIterator(args.batch_size, self.examples, train.vocabulary_text, train.vocabulary_label,
                                       shuffle_batch=shuffle_batch, shuffle_iterators=shuffle_iterators).iterators


if __name__ == "__main__":
    args = 16
    path = "./data/T_data/train.conll"
    data = MyDatasets(args, path)
