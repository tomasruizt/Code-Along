N = 2048  # sequence length, assuming prefill
d = 128  # embedding dimension


def naive_arith_intensity():
    # stage I
    loads = 2 * N * d  # loads: query, key
    stores = N**2  # store self-attn matrix
    compute = 2 * N**2 * d  # compute self-attn matrix

    # stage II
    loads2 = N**2 + N * d  # load self-attn matrix and value
    stores2 = N * d  # store output
    compute2 = 2 * N**2 * d  # compute output

    data = loads + stores + loads2 + stores2
    return (compute + compute2) / data


def flash_attn_arith_intensitiy():
    # loads: query, key, value
    loads = 3 * N * d
    stores = N * d
    
    # From FlashAttn paper Appendix C
    # https://arxiv.org/pdf/2205.14135#page=21.52
    # numtiles: N/Bm * N/Bn
    # computation per tile: Bm * Bn * d
    # total: = N^2 (Bm Bn) / (Bm Bn) * d = N^2 d
    compute = 2 * N**2 * d

    return compute / (loads + stores)

if __name__ == "__main__":
    print("=== Arithmetic Intensity ===")

    naive = naive_arith_intensity()
    print("naive =", naive)

    fa = flash_attn_arith_intensitiy()
    print("flash-attn =", fa)