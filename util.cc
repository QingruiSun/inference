#include <fstream>

#include "util.h"

void PrintTensorVectors(const std::vector<tensorflow::Tensor>& tensor_vector) {
  int vector_size = tensor_vector.size();
  std::cout << "a new tensor vector" << std::endl;
  for (int i = 0; i < vector_size; ++i) {
    std::cout << "outputs a tensor: " << std::endl;
    
    tensorflow::TensorShape tensorShape = tensor_vector[i].shape();
    auto dim_len = tensorShape.dims();
    std::string dim_string = "dims: ";
    for (int i = 0; i < dim_len; ++i) {
      dim_string = dim_string + std::to_string(tensorShape.dim_size(i)) + " ";
    }
    std::cout << dim_string << std::endl;

    auto tensorData = tensor_vector[i].flat<float>().data();
    // Print the tensor data
    for (int i = 0; i < tensorShape.num_elements(); i++) {
      std::cout << tensorData[i] << " ";
    }
    std::cout << "-----------------" << std::endl;
  }
  std::cout << std::endl;
}

void PrintTensor(const tensorflow::Tensor& tensor, int id) {
  tensorflow::TensorShape tensorShape = tensor.shape();
  auto dim_len = tensorShape.dims();
  std::string dim_string = "id: " + std::to_string(id) + " " "dims: ";
  for (int i = 0; i < dim_len; ++i) {
    dim_string = dim_string + std::to_string(tensorShape.dim_size(i)) + " ";
  }
  std::cout << dim_string << std::endl;

  auto tensorData = tensor.flat<float>().data();
  // Print the tensor data
  std::cout << "-----------------" << std::endl;
}
