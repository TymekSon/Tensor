//
// Created by chomi on 13.11.2025.
//
#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <numeric>
#include <initializer_list>

#include "Tensor.h"

Tensor::Tensor() : data_(nullptr), numel_(0), req_grad_(false), grad_(nullptr) {}

Tensor::Tensor(Arena *arena, const std::vector<size_t> &shape, bool req_grad) : shape_(shape), req_grad_(req_grad), grad_(nullptr) {
    numel_ = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    data_ = arena->allocate(numel_);
}

Tensor::~Tensor() {
    delete[] data_;
    delete[] grad_;
}

void Tensor::zero() {
    std::fill(data_, data_ + numel_, 0.0f);
}

void Tensor::fill(float val) {
    std::fill(data_, data_ + numel_, val);
}

void Tensor::random(float mean, float stddev) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < numel_; ++i) {
        data_[i] = dist(gen);
    }
}

float& Tensor::operator[](size_t index) {
    if (index >= numel_) throw std::out_of_range("Tensor index out of range");
    return data_[index];
}

const float& Tensor::operator[](size_t index) const {
    if (index >= numel_) throw std::out_of_range("Tensor index out of range");
    return data_[index];
}

void Tensor::set_grad(Tensor* grad_tensor) {
    grad_ = grad_tensor;
}

Tensor Tensor::add(Arena &arena, Tensor &a, Tensor &b) {
    if (a.numel_ != b.numel_) {
        throw std::out_of_range("Add: Tensor add not equal Tensor size");
    }
    Tensor out(&arena, a.shape_);
    for (size_t i = 0; i < a.numel_; ++i) {
        out.data_[i] = a.data_[i] + b.data_[i];
    }
    return out;
}

Tensor Tensor::sub(Arena &arena, Tensor &a, Tensor &b) {
    if (a.numel_ != b.numel_) {
        throw std::out_of_range("Sub: Tensor add not equal Tensor size");
    }
    Tensor out(&arena, a.shape_);
    for (size_t i = 0; i < a.numel_; ++i) {
        out.data_[i] = a.data_[i] - b.data_[i];
    }
    return out;
}

Tensor Tensor::matmul(Arena &arena, Tensor &a, Tensor &b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::out_of_range("Matmul: both inputs have to be 2D");
    }
    size_t m = a.shape_[0];
    size_t n = a.shape_[1];
    size_t p = b.shape_[1];

    if (b.shape_[0] != n) {
        throw std::invalid_argument("Matmul: inner dimensions mismatch");
    }
    Tensor out(&arena, {m, p});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += a.data_[i * n + k] * b.data_[k * p + j];
            }
            out.data_[i * p + j] = sum;
        }
    }
    return out;
}

void Tensor::print(const std::string& name="") const {
    if (!name.empty()) std::cout << name << " = ";
    std::cout << "[";
    for (size_t i = 0; i < numel_; ++i) {
        std::cout << data_[i];
        if (i + 1 < numel_) std::cout << ", ";
    }
    std::cout << "]\n";
}