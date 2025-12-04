#include <iostream>
#include "Tensor.h"
#include "Arena.h"

int main() {
    Arena arena(2096);

    // Test conv2d
    Tensor image(&arena, {28,28});
    Tensor kernel(&arena, {5,5});
    Tensor out(&arena, {24,24});
    Tensor pooled(&arena, {12,12});
    Tensor normalized(&arena, {12,12});

    image.random(2, 3);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            kernel.get(i, j) = (i*i)/16;
        }
    }

    image.print("Img", true);

    Tensor::conv2d(arena, image, kernel, 1, out);
    out.print("out", true);

    Tensor::maxpool2d(arena, out, 2, PoolingType::MaxPool, pooled);
    pooled.print("pooled", true);

    Tensor::batch_norm(pooled, normalized, 1, 1);
    normalized.print("normalized", true);

    std::cout << "Arena used: " << arena.used() << " / " << arena.capacity() << std::endl;

    return 0;
}