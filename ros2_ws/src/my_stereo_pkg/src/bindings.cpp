#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "my_stereo_pkg/depth_estimation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sphere_stereo_cuda, m) {
    m.doc() = "CUDA-accelerated Sphere Sweeping Stereo implementation";
    
    // RGBD_Estimator binding
    py::class_<RGBD_Estimator>(m, "RGBD_Estimator")
        .def(py::init<const std::vector<float>&, const std::vector<float>&,
                      const std::vector<float>&, const std::vector<float>&,
                      float, float, int,
                      const std::vector<int>&, const std::vector<float>&,
                      const std::vector<int>&, const std::vector<int>&,
                      int, int, int, int, int, int,
                      float, float, int>(),
             py::arg("calibrations_rt"),
             py::arg("calibrations_intrinsics"),
             py::arg("calibrations_sphere"),
             py::arg("calibrations_resolution"),
             py::arg("min_dist"),
             py::arg("max_dist"),
             py::arg("candidate_count"),
             py::arg("references_indices"),
             py::arg("reprojection_viewpoint"),
             py::arg("image_widths"),
             py::arg("image_heights"),
             py::arg("matching_width"),
             py::arg("matching_height"),
             py::arg("rgb_to_stitch_width"),
             py::arg("rgb_to_stitch_height"),
             py::arg("panorama_width"),
             py::arg("panorama_height"),
             py::arg("sigma_i"),
             py::arg("sigma_s"),
             py::arg("device") = 0,
             "Initialize RGBD_Estimator with camera calibrations and parameters")
        .def("estimate_RGBD_panorama",
             [](RGBD_Estimator& self,
                const std::vector<std::vector<float>>& images_to_match,
                const std::vector<std::vector<float>>& images_to_stitch) {
                 auto result = self.estimate_RGBD_panorama(images_to_match, images_to_stitch);
                 
                 // Convert to numpy arrays
                 py::array_t<uint8_t> rgb_array(result.first.size());
                 py::array_t<float> distance_array(result.second.size());
                 
                 auto rgb_buf = rgb_array.request();
                 auto dist_buf = distance_array.request();
                 
                 std::memcpy(rgb_buf.ptr, result.first.data(), result.first.size() * sizeof(uint8_t));
                 std::memcpy(dist_buf.ptr, result.second.data(), result.second.size() * sizeof(float));
                 
                 return py::make_tuple(rgb_array, distance_array);
             },
             "Estimate RGB-D panorama from fisheye images",
             py::arg("images_to_match"),
             py::arg("images_to_stitch"));
    
    // DoubleSphereCalibration binding (for testing/inspection)
    py::class_<DoubleSphereCalibration>(m, "DoubleSphereCalibration")
        .def(py::init<>())
        .def_readwrite("fx", &DoubleSphereCalibration::fx)
        .def_readwrite("fy", &DoubleSphereCalibration::fy)
        .def_readwrite("cx", &DoubleSphereCalibration::cx)
        .def_readwrite("cy", &DoubleSphereCalibration::cy)
        .def_readwrite("xi", &DoubleSphereCalibration::xi)
        .def_readwrite("alpha", &DoubleSphereCalibration::alpha)
        .def_readwrite("width", &DoubleSphereCalibration::width)
        .def_readwrite("height", &DoubleSphereCalibration::height);
}
