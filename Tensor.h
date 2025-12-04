//
// Created by chomi on 13.11.2025.
//
#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <numeric>
#include <initializer_list>

#include "Arena.h"

#ifndef TENSOR_H
#define TENSOR_H

enum class ActivationType { Identity, ReLU, LReLU, Sigmoid, Tanh, Softmax };
enum class PoolingType { MaxPool, AvgPool };

class Tensor {
public:
    Tensor();

    Tensor(Arena* arena_ptr, const std::vector<size_t>& shape, bool req_grad = false);

    ~Tensor();

    void zero();
    void fill(float value);
    void random(float mean = 0.0f, float stddev = 1.0f);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    void set_grad(Tensor* grad);
    Tensor* grad() const { return grad_; }
    bool req_grad() const { return req_grad_; }

    static void add(Arena& arena, const Tensor& a, const Tensor& b, Tensor& out);
    static void sub(Arena& arena, const Tensor& a, const Tensor& b, Tensor& out);

    void activate(Arena &arena, ActivationType type, Tensor& out) const;
    void activate_derivative(Arena &arena, ActivationType type, Tensor& out) const;

    static void conv2d(Arena &arena, const Tensor &image, const Tensor &kernel, int stride, Tensor& out);
    static void maxpool2d(Arena &arena, const Tensor &image, int kernel_size, PoolingType type, Tensor& out);

    static void batch_norm(const Tensor &in, const Tensor &out, float mean, float stddev);
    static void dropout(const Tensor &in, Tensor &out, float p, bool train);

    void transpose(Arena &arena, Tensor& out) const;
    static void matmul(Arena& arena, const Tensor& a, const Tensor& b, Tensor& out);
    static void element_wise(Arena &arena, const Tensor& a, const Tensor& b, Tensor& out);

    size_t flatten_index(const std::vector<size_t>& shape, const std::vector<size_t>& indices) const;

    template<typename... Args>
    float& get(Args... args);

    template<typename... Args>
    const float& get(Args... args) const;

    void print(const std::string& name, bool pretty) const;

    size_t numel() const { return numel_; }
    const std::vector<size_t>& shape() const { return shape_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

private:
    float* data_;
    std::vector<size_t> shape_;
    size_t numel_;
    bool req_grad_;
    Tensor* grad_;
};

inline size_t Tensor::flatten_index(const std::vector<size_t>& shape,
                                    const std::vector<size_t>& indices) const {
    if (shape.size() != indices.size())
        throw std::out_of_range("Wrong number of indices");

    size_t idx = 0;
    size_t stride = 1;

    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape[i])
            throw std::out_of_range("Index out of bounds");
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return idx;
}

template<typename... Args>
float& Tensor::get(Args... args) {
    std::vector<size_t> idx{ static_cast<size_t>(args)... };
    size_t flat = flatten_index(shape_, idx);
    return data_[flat];
}

template<typename... Args>
const float& Tensor::get(Args... args) const {
    std::vector<size_t> idx{ static_cast<size_t>(args)... };
    size_t flat = flatten_index(shape_, idx);
    return data_[flat];
}



#endif //TENSOR_H