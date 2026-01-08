#include <pybind11/pybind11.h>
#include <iostream>

void test_connection() {
    std::cout << "Hello from C++! Basic test without tensor" << std::endl;
}

PYBIND11_MODULE(_core_cpp, m) {
    m.def("test_connection", &test_connection, "Test connection without tensor");
}