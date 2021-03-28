__author__ = 'Daivd Guriel'
import string
from itertools import product as cartesian_product
from torch import zeros, cat, tensor
from utils import *
from  torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class FormMorphEntry:
    def __init__(self, lemma:str, form:str, feat:str):
        self.lemma = lemma
        self.form = form
        self.features = feat
    def __str__(self):
        return f"{self.lemma}   {self.form}   {self.features}" # str(dict(zip(['features','form'],[self.features,self.form])))


# def build_lemmas_from_file(path):
#     vocab = {} # vocab:[FormMorphEntry]
#     curr_lemma = ''
#     with open(path, 'r', encoding='utf8') as f:
#         for i, line in enumerate(f):
#             if line in {'', '\n'}: continue
#             lemma, form, feats = line[:-1].split('\t')
#             def assert_georgian_script(w): return 0 not in [c in kat_alphabet for c in w]
#             assert assert_georgian_script(lemma) and assert_georgian_script(form)
#             obj = FormMorphEntry(lemma, form, feats)
#             if lemma!=curr_lemma:
#                 vocab[lemma]=[obj]
#             else:
#                 vocab[lemma].append(obj)
#             curr_lemma = lemma
#     return vocab


def product(paradigm:[FormMorphEntry]):
    """
    The function takes a list of forms of the same lemma, and generates all the possible pairs ([feat1,form1,feat2],form2)
    :param paradigm: a list of FormMorphEntry objects
    :return: a list of the pairs
    """
    form_feat_list = [(e.form,e.features) for e in paradigm]
    cart_prod = list(cartesian_product(form_feat_list, form_feat_list))
    cart_prod_no_identical = list(filter(lambda e: e[0]!=e[1], cart_prod))
    samples, labels = [], []
    for pair in cart_prod_no_identical:
        samples.append((pair[0][1],pair[0][0],pair[1][1]))
        labels.append(pair[1][0])
    return samples, labels


def katword2vector(w:str or [str]):
    # vec = zeros(size,len(w),dtype=torch.long, device=device)
    # for i,c in enumerate(w):
    #     vec[enc_alph2idx[c]][i]=1
    vec = torch.tensor([[enc_alph2idx[c] for c in w]], dtype=torch.long, device=device)
    return vec

def features2vector(feature:str):
    assert feature[:2]=='V;'
    morphfeats = feature[2:].split(';')
    vec = katword2vector(morphfeats)
    # vec = zeros((size,len(morphfeats)))
    # for i,c in enumerate(morphfeats):
    #     vec[enc_alph2idx[c]][i]=1
    return vec


def sample2vector(sample):
    feat1, form, feat2 = sample
    v1, v2, v3 = features2vector(feat1), katword2vector(form), features2vector(feat2)
    res = cat((v1, v2, v3), dim=1)
    return res


def label2vector(label):
    return katword2vector(label)

def vector2sample(tensor):
    t = tensor[0].tolist()
    l = [idx2enc_alph[c] for c in t]
    return l


def encode_samples(samples): return [sample2vector(s) for s in samples]

def encode_labels(labels): return [label2vector(l) for l in labels]


class ReinflectionDataset(Dataset):
    def __init__(self, data_file, dataset_size, limit_size:bool=False):
        self.file_name = data_file
        self.data = []
        with open(self.file_name) as f: # Reading a file of indexes, not strings!!!
            for i, line in enumerate(f):
                # feat_i, form_i, feat_j, form_j = line[:-1].split('\t')
                # sample = sample2vector((feat_i, form_i, feat_j))
                # label = label2vector(form_j)
                # self.samples.append(sample)
                # self.labels.append(label)
                if limit_size and i==dataset_size: break
                sample, label = line[:-1].split('\t')
                sample, label = sample.split(), label.split()
                sample = tensor([int(c) for c in sample],dtype=torch.long, device=device)
                label = tensor([int(c) for c in label]+[EOS_token],dtype=torch.long, device=device)

                # assert 0 not in [0 <= c.item() <= 34 for c in label[0]]
                self.data.append((sample, label))
        # See unused commands that were formerly used here in utils



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # with open(self.file_name, encoding='utf8') as f:
            # for i, line in enumerate(f):
                # if i==idx-1:
                #     feat_i, form_i, feat_j, form_j = line[:-1].split('\t')
                #     sample = sample2vector((feat_i, form_i, feat_j))
                #     label = label2vector(form_j)
                #     return sample, label
        return self.data[idx] # a pair (sample, label)


def my_collate(batch):
    xx, yy = zip(*batch)

    x_lens = [len(s) for s in xx] # if this works, remove the extra [ ] in __init__
    y_lens = [len(s) for s in yy] # if this works, remove the extra [ ] in __init__

    xx_pad = pad_sequence(xx,padding_value=PAD_token)
    yy_pad = pad_sequence(yy,padding_value=PAD_token)

    # return samples, labels
    return xx_pad, yy_pad, x_lens, y_lens
