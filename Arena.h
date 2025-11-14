//
// Created by chomi on 13.11.2025.
//
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <vector>

#ifndef ARENA_H
#define ARENA_H



class Arena {
public:
  explicit Arena(size_t total_floats);
  ~Arena();

  float* allocate(int size);

  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }
  size_t peak() const { return peak_; }

  float* base() const { return data_; }
  void reset();

private:
  float* data_;
  size_t capacity_;
  size_t offset_;
  size_t peak_;
};



#endif //ARENA_H
