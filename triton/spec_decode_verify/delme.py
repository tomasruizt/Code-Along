xs = [1, 2, 3]


def fn(x):
    return x * x


ys = map(fn, map(fn, xs))

print(list(ys))
