from solve_mnist_lib import get_train_conf, train_and_save

conf = get_train_conf("cifar100")
train_and_save(seed=0, conf=conf)
