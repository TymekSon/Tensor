#include <iostream>
#include "Tensor.h"
#include "Arena.h"
#include "MNIST_loader.h"

int main() {
    MNIST_loader loader;

    Arena net_arena(2096);
    Arena data_train(60000*784 + 60000*10);
    Arena data_test(10000*784 + 10000*10);

    Tensor train_images = loader.load_images(data_train, "../data/trainImages.idx3-ubyte");
    Tensor train_labels = loader.load_labels(data_train, "../data/trainLabels.idx1-ubyte");

    Tensor test_images = loader.load_images(data_test, "../data/testImages.idx3-ubyte");
    Tensor test_labels = loader.load_labels(data_test, "../data/testLabels.idx1-ubyte");

    return 0;
}