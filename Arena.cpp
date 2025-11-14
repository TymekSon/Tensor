//
// Created by chomi on 13.11.2025.
//

#include "Arena.h"

Arena::Arena(size_t total_floats) : capacity_(total_floats), offset_(0), peak_(0) {
  if (capacity_ == 0) {
    data_ = nullptr;
  }else{
    data_ = new float[capacity_];
  }
}

Arena::~Arena() {
  delete[] data_;
}

float* Arena::allocate(int size) {
  if(offset_ + size > capacity_) {
    throw std::runtime_error("Arena overflow: not enough space");
  }
  float* ptr = data_ + offset_;
  offset_ += size;
  peak_ = std::max(peak_, offset_);
  return ptr;
}

void Arena::reset(){
  offset_ = 0;
}