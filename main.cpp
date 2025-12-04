#include <iostream>
#include "Tensor.h"
#include "Arena.h"
#include "MNIST_loader.h"

int main() {
    MNIST_loader loader;

    Arena net_arena(4096);
    Arena data_train(60000*784 + 60000*10);
    Arena data_test(10000*784 + 10000*10);

    Tensor train_images = loader.load_images(data_train, "../data/trainImages.idx3-ubyte");
    Tensor train_labels = loader.load_labels(data_train, "../data/trainLabels.idx1-ubyte");

    Tensor test_images = loader.load_images(data_test, "../data/testImages.idx3-ubyte");
    Tensor test_labels = loader.load_labels(data_test, "../data/testLabels.idx1-ubyte");

    Tensor image(&net_arena, {28, 28});
    Tensor out(&net_arena, {26, 26});
    Tensor kernel(&net_arena, {3, 3});

    kernel.get(0, 0) = -1; kernel.get(0, 1) = 0; kernel.get(0, 2) = 1;
    kernel.get(1, 0) = -1; kernel.get(1, 1) = 0; kernel.get(1, 2) = 1;
    kernel.get(2, 0) = -1; kernel.get(2, 1) = 0; kernel.get(2, 2) = 1;

    kernel.print("kernel", true, 2);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            image.get(i, j) = test_images.get(1111 , i, j);
        }
    }

    Tensor::conv2d(net_arena, image, kernel, 1, out);

    Tensor::batch_norm(out, out, 0, 1);

    image.save_as_png("image.png");
    out.save_as_png("out.png");

    return 0;
}