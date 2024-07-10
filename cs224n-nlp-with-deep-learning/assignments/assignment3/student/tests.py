from utils import pad_sents
from model_embeddings import ModelEmbeddings
from vocab import VocabEntry, Vocab
from torch.nn import Embedding

def test_pad_sents():
    padded = pad_sents(sents=[['a', 'b', 'c'], ['d', 'e'], ['f']], pad_token='p')
    assert padded == [['a', 'b', 'c'], ['d', 'e', 'p'], ['f', 'p', 'p']]


def test_b():
    unkown_token = "<unk>"
    src_vocab = VocabEntry({"a":0, "b":1, unkown_token:2})
    tgt_vocab = VocabEntry({"c":0, "d":1, "e": 2, unkown_token:3})
    embed_dim = 5
    m = ModelEmbeddings(embed_size=embed_dim, vocab=Vocab(src_vocab, tgt_vocab))
    assert isinstance(m.source, Embedding)
    assert m.source.weight.shape == (len(src_vocab), embed_dim)
    assert m.target.weight.shape == (len(tgt_vocab), embed_dim)
    assert isinstance(m.target, Embedding)