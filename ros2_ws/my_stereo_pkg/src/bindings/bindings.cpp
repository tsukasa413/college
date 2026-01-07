#include <pybind11/pybind11.h>
#include <iostream>

#ifdef USE_TORCH
#include <torch/extension.h>
#endif

namespace py = pybind11;

#ifdef USE_TORCH
// LibTorch使用時: テスト用の接続確認関数
void test_connection(at::Tensor input) {
    std::cout << "Hello from C++! Tensor size: [";
    for (int i = 0; i < input.dim(); ++i) {
        std::cout << input.size(i);
        if (i < input.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // デバイス情報も表示
    std::cout << "Tensor device: " << input.device() << std::endl;
    std::cout << "Tensor dtype: " << input.dtype() << std::endl;
}
#else
// LibTorchなし: ダミー関数
void test_connection_dummy() {
    std::cout << "LibTorch is not available. Please install LibTorch to use tensor operations." << std::endl;
}
#endif

// Pybind11モジュール定義
PYBIND11_MODULE(_core_cpp, m) {
    m.doc() = "my_stereo_pkg C++ bindings with optional LibTorch support";
    
#ifdef USE_TORCH
    m.def("test_connection", &test_connection, 
          "Test connection between Python and C++ with a PyTorch tensor",
          py::arg("input"));
    m.attr("HAS_TORCH") = true;
#else
    m.def("test_connection", &test_connection_dummy, 
          "Dummy function (LibTorch not available)");
    m.attr("HAS_TORCH") = false;
#endif
}
