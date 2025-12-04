//
// Created by chomi on 13.11.2025.
//
#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <numeric>
#include <initializer_list>
#include<iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

#include "Tensor.h"

Tensor::Tensor() : data_(nullptr), numel_(0), req_grad_(false), grad_(nullptr) {}

Tensor::Tensor(Arena *arena, const std::vector<size_t> &shape, bool req_grad) : shape_(shape), req_grad_(req_grad), grad_(nullptr) {
    numel_ = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    data_ = arena->allocate(numel_);
}

Tensor::~Tensor() {}

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

void Tensor::add(Arena &arena, const Tensor &a, const Tensor &b, Tensor& out) {
    if (a.numel_ != b.numel_)
        throw std::out_of_range("Add: Tensor sizes differ");

    if (a.numel_ != out.numel_)
        throw std::out_of_range("Add: Output size mismatch");

    for (size_t i = 0; i < a.numel_; ++i)
        out[i] = a[i] + b[i];
}


void Tensor::sub(Arena &arena, const Tensor &a, const Tensor &b, Tensor& out) {
    if (a.numel_ != b.numel_)
        throw std::out_of_range("Sub: Tensor sizes differ");

    if (a.numel_ != out.numel_)
        throw std::out_of_range("Sub: Output size mismatch");

    for (size_t i = 0; i < a.numel_; ++i)
        out[i] = a[i] - b[i];
}


void Tensor::matmul(Arena &arena, const Tensor &a, const Tensor &b, Tensor& out) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2)
        throw std::out_of_range("Matmul: both inputs must be 2D");

    size_t m = a.shape_[0];
    size_t n = a.shape_[1];
    size_t p = b.shape_[1];

    if (b.shape_[0] != n)
        throw std::invalid_argument("Matmul: inner dimensions mismatch");

    if (out.shape_.size() != 2 || out.shape_[0] != m || out.shape_[1] != p)
        throw std::invalid_argument("Matmul: output shape mismatch");

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k)
                sum += a.get(i, k) * b.get(k, j);

            out.get(i, j) = sum;
        }
    }
}


void Tensor::element_wise(Arena &arena, const Tensor &a, const Tensor &b, Tensor& out) {
    if (a.numel_ != b.numel_)
        throw std::invalid_argument("Element_wise: Tensor sizes differ");

    if (a.numel_ != out.numel_)
        throw std::invalid_argument("Element_wise: Output size mismatch");

    for (size_t i = 0; i < a.numel_; ++i)
        out[i] = a[i] * b[i];
}


void Tensor::conv2d(Arena &arena, const Tensor &image, const Tensor &kernel, int stride, Tensor& out) {
    size_t H = image.shape_[0];
    size_t W = image.shape_[1];
    size_t KH = kernel.shape_[0];
    size_t KW = kernel.shape_[1];

    if (H < KH || W < KW)
        throw std::invalid_argument("Kernel larger than image");

    size_t OH = (H - KH) / stride + 1;
    size_t OW = (W - KW) / stride + 1;

    if (out.shape_.size() != 2 || out.shape_[0] != OH || out.shape_[1] != OW)
        throw std::invalid_argument("Conv2d: output shape mismatch");

    for (size_t i = 0; i < OH; ++i) {
        for (size_t j = 0; j < OW; ++j) {

            float sum = 0.0f;

            for (size_t ki = 0; ki < KH; ++ki) {
                for (size_t kj = 0; kj < KW; ++kj) {

                    size_t img_row = i * stride + ki;
                    size_t img_col = j * stride + kj;

                    sum += image.get(img_row, img_col) * kernel.get(ki, kj);
                }
            }

            out.get(i, j) = sum;
        }
    }
}


void Tensor::maxpool2d(Arena &arena, const Tensor &image, int kernel_size, PoolingType type, Tensor& out){
    size_t H = image.shape_[0];
    size_t W = image.shape_[1];

    size_t OH = H / kernel_size;
    size_t OW = W / kernel_size;

    if (out.shape_.size() != 2 || out.shape_[0] != OH || out.shape_[1] != OW)
        throw std::invalid_argument("Maxpool2d: output shape mismatch");

    for (size_t i = 0; i < OH; ++i) {
        for (size_t j = 0; j < OW; ++j) {

            float acc = (type == PoolingType::MaxPool ? -std::numeric_limits<float>::infinity() : 0.0f);

            for (size_t ki = 0; ki < kernel_size; ++ki) {
                for (size_t kj = 0; kj < kernel_size; ++kj) {

                    float v = image.get(i * kernel_size + ki,
                                        j * kernel_size + kj);

                    if (type == PoolingType::MaxPool)
                        acc = std::max(acc, v);
                    else
                        acc += v;
                }
            }

            if (type == PoolingType::AvgPool)
                acc /= (kernel_size * kernel_size);

            out.get(i, j) = acc;
        }
    }
}

void Tensor::batch_norm(const Tensor &in, const Tensor &out, float beta, float gamma) {
    float avg = 0.0f;
    for (size_t i = 0; i < in.numel_; ++i) {
        avg += in.data_[i];
    }
    avg = avg / in.numel_;

    float stddev = 0.0f;

    for (size_t i = 0; i < in.numel_; ++i) {
        stddev += (in.data_[i] - avg) * (in.data_[i] - avg);
    }
    stddev = stddev/in.numel_;

    for (size_t i = 0; i < in.numel_; ++i) {
        out.data_[i] = (in.data_[i] - avg)/sqrt(stddev+1e-6);
        out.data_[i] = gamma * out.data_[i] + beta;
    }
}

void Tensor::dropout(const Tensor &in, Tensor &out, float p, bool train) {
    if (!train) return;

    float scale = 1.0f / (1.0f - p);

    for (int i = 0; i < in.numel_; ++i) {
        if ((float)rand() / RAND_MAX < p) {
            out.data_[i] = 0.0f;
        } else {
            out.data_[i] *= scale;
        }
    }
}



void Tensor::activate(Arena &arena, ActivationType type, Tensor& out) const {
    if (out.shape_ != shape_ || out.numel_ != numel_)
        throw std::invalid_argument("Activate: output shape mismatch");

    switch (type) {
        case ActivationType::Identity:
            for (size_t i = 0; i < numel_; i++) out.data_[i] = data_[i];
            break;

        case ActivationType::ReLU:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = data_[i] > 0.0f ? data_[i] : 0.0f;
            break;

        case ActivationType::LReLU:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = data_[i] > 0.0f ? data_[i] : 0.01f * data_[i];
            break;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
            break;

        case ActivationType::Tanh:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = std::tanh(data_[i]);
            break;

        case ActivationType::Softmax: {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < numel_; i++)
                max_val = std::max(max_val, data_[i]);

            float sum = 0.0f;
            for (size_t i = 0; i < numel_; i++)
                sum += std::exp(data_[i] - max_val);

            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = std::exp(data_[i] - max_val) / sum;

            break;
        }
    }
}

void Tensor::activate_derivative(Arena &arena, ActivationType type, Tensor& out) const {
    if (out.shape_ != shape_ || out.numel_ != numel_)
        throw std::invalid_argument("Activate derivative: output shape mismatch");

    switch (type) {
        case ActivationType::Identity:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = 1.0f;
            break;

        case ActivationType::ReLU:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = data_[i] > 0.0f ? 1.0f : 0.0f;
            break;

        case ActivationType::LReLU:
            for (size_t i = 0; i < numel_; i++)
                out.data_[i] = data_[i] > 0.0f ? 1.0f : 0.01f;
            break;

        case ActivationType::Sigmoid:
            for (size_t i = 0; i < numel_; i++) {
                float s = 1.0f / (1.0f + std::exp(-data_[i]));
                out.data_[i] = s * (1.0f - s);
            }
            break;

        case ActivationType::Tanh:
            for (size_t i = 0; i < numel_; i++) {
                float t = std::tanh(data_[i]);
                out.data_[i] = 1.0f - t * t;
            }
            break;

        case ActivationType::Softmax: {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < numel_; i++)
                max_val = std::max(max_val, data_[i]);

            float sum = 0.0f;
            for (size_t i = 0; i < numel_; i++)
                sum += std::exp(data_[i] - max_val);

            for (size_t i = 0; i < numel_; i++) {
                float s = std::exp(data_[i] - max_val) / sum;
                out.data_[i] = s * (1.0f - s);
            }

            break;
        }
    }
}

void Tensor::transpose(Arena &arena, Tensor& out) const {
    if (shape_.size() != 2)
        throw std::invalid_argument("Transpose: input must be 2D");

    size_t rows = shape_[0];
    size_t cols = shape_[1];

    if (out.shape_.size() != 2 || out.shape_[0] != cols || out.shape_[1] != rows)
        throw std::invalid_argument("Transpose: output shape mismatch");

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            out.get(j, i) = get(i, j);
        }
    }
}


void Tensor::print(const std::string &name, bool pretty) const {
    if (!name.empty()) {
        std::cout << name << " = ";
    }

    if (!pretty) {
        // Standardowy print
        std::cout << "[";
        for (size_t i = 0; i < numel_; ++i) {
            std::cout << data_[i];
            if (i + 1 < numel_) std::cout << ", ";
        }
        std::cout << "]\n";
        return;
    }

    // pretty print
    std::cout << "\n";

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(2);

    if (shape_.size() == 1) {
        for (size_t i = 0; i < numel_; ++i) {
            std::cout << data_[i];
            if (i + 1 < numel_) std::cout << "\t";
        }
        std::cout << "\n";
        return;
    }

    if (shape_.size() == 2) {
        size_t H = shape_[0];
        size_t W = shape_[1];

        for (size_t i = 0; i < H; ++i) {
            for (size_t j = 0; j < W; ++j) {
                std::cout << data_[i * W + j] << "\t";
            }
            std::cout << "\n";
        }
        return;
    }

    size_t stride = shape_.back();
    for (size_t i = 0; i < numel_; ++i) {
        std::cout << data_[i] << "\t";
        if ((i + 1) % stride == 0) std::cout << "\n";
    }
}