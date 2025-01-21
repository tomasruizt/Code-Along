from solve_mnist_lib import get_train_conf, train_and_save

conf = get_train_conf("mnist")
train_and_save(seed=0, conf=conf)
