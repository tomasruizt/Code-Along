from utils import pad_sents
from model_embeddings import ModelEmbeddings, EmbeddingDims
from vocab import VocabEntry, Vocab
from nmt_model import NMT
from torch.nn import Embedding

def test_pad_sents():
    padded = pad_sents(sents=[['a', 'b', 'c'], ['d', 'e'], ['f']], pad_token='p')
    assert padded == [['a', 'b', 'c'], ['d', 'e', 'p'], ['f', 'p', 'p']]


def test_b():
    embed_dim = 5
    vocab = some_vocab()
    m = ModelEmbeddings(embed_size=embed_dim, vocab=vocab)
    assert isinstance(m.source, Embedding)
    assert isinstance(m.target, Embedding)
    
    expected_dims = EmbeddingDims(
        embed_size=embed_dim,
        src_vocab_size=len(vocab.src),
        tgt_vocab_size=len(vocab.tgt)
    )
    assert m.dims() == expected_dims


def some_vocab() -> Vocab:
    unkown_token = "<unk>"
    src_vocab = VocabEntry({"a":0, "b":1, unkown_token:2})
    tgt_vocab = VocabEntry({"c":0, "d":1, "e": 2, unkown_token:3})
    return Vocab(src_vocab, tgt_vocab)

def test_c_and_d():
    m = NMT(embed_size=5, hidden_size=6, vocab=some_vocab())
    source = [["a", "b", "b"], ["b", "a"]]
    target = [["c"], ["c"]]
    m(source, target)