#include <iostream>
#include "Tensor.h"
#include "Arena.h"

int main() {
    Arena arena(1024*10);

    Tensor A(&arena, {2,3});
    Tensor B(&arena, {2,3});
    Tensor C(&arena, {3,2});
    Tensor G(&arena, {3,3,3});

    A.fill(1.0f);
    B.fill(2.0f);
    C.fill(2.0f);

    A.print("A");
    B.print("B");
    C.print("C");
    G.print("G");

    // Test add i sub
    Tensor D = Tensor::add(arena, A, B);
    Tensor E = Tensor::sub(arena, B, A);
    D.print("A+B");
    E.print("B-A");

    // Test matmul (2x3) * (3x2) = (2x2)
    Tensor F = Tensor::matmul(arena, A, C);
    F.print("A*C");

    // Dodatkowy test czy arena się nie przepełnia
    std::cout << "Arena used: " << arena.used() << " / " << arena.capacity() << std::endl;

    return 0;
}