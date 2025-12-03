#include <iostream>
#include "Tensor.h"
#include "Arena.h"

int main() {
    Arena arena(1024);

    // Test conv2d
    Tensor image(&arena, {9,9});
    Tensor kernel(&arena, {4,4});
    image.random(0, 1);
    kernel.fill(2.0f);

    image.print("Img", true);

    Tensor out = Tensor::conv2d(arena, image, kernel, 1);
    out.print("out", true);

    Tensor pooled = Tensor::maxpool2d(arena, out, 2, PoolingType::MaxPool);
    pooled.print("pooled", true);

    Tensor cube(&arena, {3,3,3});
    cube.random(5,2);

    cube.print("cube: ", true);

    float testvalue = cube.get(0,2,1);
    std::cout << testvalue << std::endl;

    std::cout << "Arena used: " << arena.used() << " / " << arena.capacity() << std::endl;

    return 0;
}