// MNIST_loader.cpp

#include "MNIST_loader.h"

// Prywatna funkcja do odczytu uint32 z big-endian
uint32_t MNIST_loader::read32(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(&bytes), 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

Tensor MNIST_loader::load_images(Arena& arena, const std::string& file) {
    std::ifstream input_file(file, std::ios::binary);
    if (!input_file) throw std::runtime_error("Nie można otworzyć pliku obrazów!");

    uint32_t magic = read32(input_file);
    if (magic != 2051) throw std::runtime_error("Błędny magiczny numer dla obrazów MNIST!");

    uint32_t count = read32(input_file);
    uint32_t rows = read32(input_file);
    uint32_t cols = read32(input_file);
    if (rows != 28 || cols != 28) throw std::runtime_error("Nieoczekiwany rozmiar obrazów MNIST!");

    // Tworzymy Tensor [count, 28, 28]
    Tensor images(&arena, {count, rows, cols});

    // Bufor na batch bajtów (cały plik, ale dla MNIST to OK, ~47 MB raw)
    std::vector<uint8_t> buffer(count * rows * cols);
    input_file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    // Wypełniamy data_ sekwencyjnie z normalizacją (szybciej niż przez get)
    float* ptr = images.data();
    for (size_t i = 0; i < buffer.size(); ++i) {
        ptr[i] = static_cast<float>(buffer[i]) / 255.0f;
    }

    return images;
}

Tensor MNIST_loader::load_labels(Arena& arena, const std::string& file) {
    std::ifstream input_file(file, std::ios::binary);
    if (!input_file) throw std::runtime_error("Nie można otworzyć pliku etykiet!");

    uint32_t magic = read32(input_file);
    if (magic != 2049) throw std::runtime_error("Błędny magiczny numer dla etykiet MNIST!");

    uint32_t count = read32(input_file);

    // Tworzymy Tensor [count] (jak vector<float>)
    Tensor labels(&arena, {count});

    // Bufor na etykiety
    std::vector<uint8_t> buffer(count);
    input_file.read(reinterpret_cast<char*>(buffer.data()), count);

    // Kopiujemy do float (bez normalizacji, bo to klasy 0-9)
    float* ptr = labels.data();
    for (size_t i = 0; i < count; ++i) {
        ptr[i] = static_cast<float>(buffer[i]);
    }

    return labels;
}

Tensor MNIST_loader::load_one_hot_labels(Arena& arena, const std::string& file, int num_classes) {
    // Najpierw ładujemy raw etykiety
    Tensor raw_labels = load_labels(arena, file);  // [count]

    size_t count = raw_labels.numel();

    // Tworzymy one-hot [count, num_classes]
    Tensor one_hot(&arena, {count, static_cast<size_t>(num_classes)});
    one_hot.zero();  // wypełniamy zerami

    // Używamy get do ustawiania 1.0 w odpowiednich miejscach (tu get jest OK, bo rzadko)
    for (size_t i = 0; i < count; ++i) {
        size_t label = static_cast<size_t>(raw_labels[i]);
        if (label >= static_cast<size_t>(num_classes)) throw std::runtime_error("Etykieta poza zakresem!");
        one_hot.get(i, label) = 1.0f;
    }

    return one_hot;
}