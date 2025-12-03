#include <iostream>
#include "Tensor.h"
#include "Arena.h"

int main() {
    Arena arena(1024);

    // Test conv2d
    Tensor image(&arena, {9,9});
    Tensor kernel(&arena, {4,4});
    Tensor out(&arena, {6,6});
    Tensor pooled(&arena, {3,3});
    Tensor activated(&arena, {3,3});

    image.random(0, 1);
    kernel.fill(2.0f);

    image.print("Img", true);

    Tensor::conv2d(arena, image, kernel, 1, out);
    out.print("out", true);

    Tensor::maxpool2d(arena, out, 2, PoolingType::MaxPool, pooled);
    pooled.print("pooled", true);

    pooled.activate(arena, ActivationType::ReLU, activated);
    activated.print("activated", true);

    std::cout << "Arena used: " << arena.used() << " / " << arena.capacity() << std::endl;

    return 0;
}