// MNIST_loader.h (lub jak masz nazwane)

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include "Arena.h"
#include "Tensor.h"  // zakładam, że to jest w osobnym pliku

class MNIST_loader {
public:
  // Ładuje obrazy jako Tensor [count, 28, 28] z float w [0,1] (z normalizacją)
  static Tensor load_images(Arena& arena, const std::string& file);

  // Ładuje etykiety jako Tensor [count] z float (0-9)
  static Tensor load_labels(Arena& arena, const std::string& file);

  // Ładuje etykiety jako one-hot Tensor [count, 10] z float
  static Tensor load_one_hot_labels(Arena& arena, const std::string& file, int num_classes = 10);

private:
  static uint32_t read32(std::ifstream& file);
};

#endif // MNIST_LOADER_H