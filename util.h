#pragma once

#include<vector>
#include<iostream>

#include "tensorflow/core/framework/tensor.h"

void PrintTensor(const tensorflow::Tensor& tensor, int id);
void PrintTensorVectors(const std::vector<tensorflow::Tensor>& tensor_vector);
