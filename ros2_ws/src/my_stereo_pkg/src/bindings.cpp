#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <iostream>

// Helper function to convert numpy to torch tensor
torch::Tensor numpy_to_torch(pybind11::array_t<float> input) {
    auto buf = input.request();
    
    // Create torch tensor from numpy data
    std::vector<int64_t> shape;
    for (auto dim : buf.shape) {
        shape.push_back(dim);
    }
    
    // Create tensor options
    auto options = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .device(torch::kCPU);
    
    // Create tensor from data
    return torch::from_blob(buf.ptr, shape, options).clone();
}

void test_connection(pybind11::array_t<float> np_input) {
    std::cout << "Hello from C++!" << std::endl;
    
    // Convert numpy to torch tensor
    auto tensor = numpy_to_torch(np_input);
    
    std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
    std::cout << "Tensor device: " << tensor.device() << std::endl;
    std::cout << "Tensor dtype: " << tensor.dtype() << std::endl;
    
    // Basic tensor operations
    auto tensor_sum = tensor.sum();
    std::cout << "Tensor sum: " << tensor_sum.item<float>() << std::endl;
    
    // Test if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available in C++!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is not available in C++" << std::endl;
    }
}

PYBIND11_MODULE(_core_cpp, m) {
    m.doc() = "My Stereo Package C++ Core Module with PyTorch support";
    m.def("test_connection", &test_connection, "Test connection with numpy->torch conversion");
}