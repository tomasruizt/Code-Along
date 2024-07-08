from utils import pad_sents

def test_pad_sents():
    padded = pad_sents(sents=[['a', 'b', 'c'], ['d', 'e'], ['f']], pad_token='p')
    assert padded == [['a', 'b', 'c'], ['d', 'e', 'p'], ['f', 'p', 'p']]