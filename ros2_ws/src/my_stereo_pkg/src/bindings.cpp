#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include "my_stereo_pkg/cuda_kernels.hpp"
#include "my_stereo_pkg/calibration.hpp"
#include "my_stereo_pkg/stitcher.hpp"
#include "my_stereo_pkg/isb_filter.hpp"
#include "my_stereo_pkg/depth_estimation.hpp"

namespace py = pybind11;
using namespace my_stereo;

// Custom type casters for CUDA vector types
namespace pybind11 { namespace detail {
    template <> struct type_caster<float2> {
    public:
        PYBIND11_TYPE_CASTER(float2, _("float2"));
        
        bool load(handle src, bool) {
            if (!py::isinstance<py::tuple>(src) && !py::isinstance<py::list>(src)) {
                return false;
            }
            auto seq = py::reinterpret_borrow<py::sequence>(src);
            if (seq.size() != 2) return false;
            
            value.x = seq[0].cast<float>();
            value.y = seq[1].cast<float>();
            return true;
        }
        
        static handle cast(float2 src, return_value_policy, handle) {
            return py::make_tuple(src.x, src.y).release();
        }
    };
    
    template <> struct type_caster<float3> {
    public:
        PYBIND11_TYPE_CASTER(float3, _("float3"));
        
        bool load(handle src, bool) {
            if (!py::isinstance<py::tuple>(src) && !py::isinstance<py::list>(src)) {
                return false;
            }
            auto seq = py::reinterpret_borrow<py::sequence>(src);
            if (seq.size() != 3) return false;
            
            value.x = seq[0].cast<float>();
            value.y = seq[1].cast<float>();
            value.z = seq[2].cast<float>();
            return true;
        }
        
        static handle cast(float3 src, return_value_policy, handle) {
            return py::make_tuple(src.x, src.y, src.z).release();
        }
    };
}} // namespace pybind11::detail

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
    
    // Calibration struct binding
    py::class_<Calibration>(m, "Calibration")
        .def(py::init<>())
        .def_readwrite("fl", &Calibration::fl, "Focal length (fx, fy)")
        .def_readwrite("principal", &Calibration::principal, "Principal point (cx, cy)")
        .def_readwrite("xi", &Calibration::xi, "First distortion parameter")
        .def_readwrite("alpha", &Calibration::alpha, "Second distortion parameter")
        .def_readwrite("matching_scale", &Calibration::matching_scale, "Scale factor for matching resolution")
        .def_readwrite("rt", &Calibration::rt, "Extrinsic matrix [4, 4]");
    
    // Stitcher class binding
    py::class_<Stitcher>(m, "Stitcher")
        .def(py::init<
            const std::vector<Calibration>&,  // calibrations
            const at::Tensor&,                 // reprojection_viewpoint
            const at::Tensor&,                 // masks
            float,                             // min_dist
            float,                             // max_dist
            int,                               // matching_cols
            int,                               // matching_rows
            int,                               // rgb_to_stitch_cols
            int,                               // rgb_to_stitch_rows
            int,                               // panorama_cols
            int,                               // panorama_rows
            const at::Device&,                 // device
            int,                               // smoothing_radius (default: 15)
            int                                // inpainting_iterations (default: 32)
        >(),
        py::arg("calibrations"),
        py::arg("reprojection_viewpoint"),
        py::arg("masks"),
        py::arg("min_dist"),
        py::arg("max_dist"),
        py::arg("matching_cols"),
        py::arg("matching_rows"),
        py::arg("rgb_to_stitch_cols"),
        py::arg("rgb_to_stitch_rows"),
        py::arg("panorama_cols"),
        py::arg("panorama_rows"),
        py::arg("device"),
        py::arg("smoothing_radius") = 15,
        py::arg("inpainting_iterations") = 32,
        "Create RGB-D panorama stitcher for fisheye cameras")
        .def("stitch", &Stitcher::stitch,
            py::arg("images"),
            py::arg("distance_maps"),
            "Stitch fisheye images and distance maps into RGB-D panorama\n\n"
            "Args:\n"
            "    images: List of [H, W, 3] uint8 color fisheye images\n"
            "    distance_maps: List of [H, W] float32 distance maps\n\n"
            "Returns:\n"
            "    Tuple of (RGB panorama [H, W, 3] uint8, distance panorama [H, W] float32)");
    
    // ISBFilter class binding for edge-preserving cost volume filtering
    py::class_<my_stereo_pkg::ISBFilter>(m, "ISBFilter")
        .def(py::init<int, const std::pair<int, int>&, const at::Device&>(),
            py::arg("candidate_count"),
            py::arg("resolution"),
            py::arg("device"),
            "Create ISB Filter for edge-preserving cost volume aggregation\n\n"
            "Args:\n"
            "    candidate_count: Number of depth candidates (cost volume channels)\n"
            "    resolution: Tuple of (cols, rows) for the image resolution\n"
            "    device: CUDA device for processing (e.g., torch.device('cuda:0'))")
        .def("apply", 
            [](my_stereo_pkg::ISBFilter& self,
               const at::Tensor& guide,
               const at::Tensor& cost,
               float sigma_i,
               float sigma_s) {
                // Python binding wrapper: Call with default stream (nullptr)
                return self.apply(guide, cost, sigma_i, sigma_s, nullptr);
            },
            py::arg("guide"),
            py::arg("cost"),
            py::arg("sigma_i"),
            py::arg("sigma_s"),
            "Apply edge-preserving filter to cost volume\n\n"
            "Args:\n"
            "    guide: Guide image [H, W, 3] (uint8) for edge preservation\n"
            "    cost: Cost volume [candidate_count, H, W] (float32) to be filtered\n"
            "    sigma_i: Edge preservation parameter (lower = preserve edges more)\n"
            "    sigma_s: Smoothing parameter (higher = more smoothing from coarse scales)\n\n"
            "Returns:\n"
            "    Tuple of (filtered cost volume [candidate_count, H, W], filtered guide [H, W, 3])")
        .def("get_scale_count", &my_stereo_pkg::ISBFilter::get_scale_count,
            "Get the number of pyramid scales used");
    
    // RGBDEstimator class binding - complete RGBD estimation pipeline
    // Matches Python RGBD_Estimator class from depth_estimation.py
    py::class_<my_stereo_pkg::RGBDEstimator>(m, "RGBDEstimator")
        .def(py::init<
            const std::vector<Calibration>&,   // calibrations
            float,                              // min_dist
            float,                              // max_dist
            int,                                // candidate_count
            const std::vector<int>&,            // references_indices
            const at::Tensor&,                  // reprojection_viewpoint
            const std::vector<at::Tensor>&,     // masks
            const std::pair<int, int>&,         // matching_resolution
            const std::pair<int, int>&,         // rgb_to_stitch_resolution
            const std::pair<int, int>&,         // panorama_resolution
            float,                              // sigma_i
            float,                              // sigma_s
            const at::Device&                   // device
        >(),
        py::arg("calibrations"),
        py::arg("min_dist"),
        py::arg("max_dist"),
        py::arg("candidate_count"),
        py::arg("references_indices"),
        py::arg("reprojection_viewpoint"),
        py::arg("masks"),
        py::arg("matching_resolution"),
        py::arg("rgb_to_stitch_resolution"),
        py::arg("panorama_resolution"),
        py::arg("sigma_i"),
        py::arg("sigma_s"),
        py::arg("device"),
        "Prepare RGB-D estimation from fisheye images (matches Python RGBD_Estimator)\n\n"
        "Args:\n"
        "    calibrations: List of camera calibration objects\n"
        "    min_dist: Minimum distance for sphere sweep volume (meters)\n"
        "    max_dist: Maximum distance for sphere sweep volume (meters)\n"
        "    candidate_count: Number of distance candidates in sweep volume\n"
        "    references_indices: List of camera indices to use as reference views\n"
        "    reprojection_viewpoint: [3] Reference viewpoint for panorama creation\n"
        "    masks: List of [1, H, W] validity masks for each camera\n"
        "    matching_resolution: (cols, rows) resolution for stereo matching\n"
        "    rgb_to_stitch_resolution: (cols, rows) resolution for RGB stitching\n"
        "    panorama_resolution: (cols, rows) output panorama resolution\n"
        "    sigma_i: Edge preservation parameter for ISB filtering\n"
        "    sigma_s: Smoothing parameter for ISB filtering\n"
        "    device: CUDA device for processing")
        .def("estimate_RGBD_panorama", &my_stereo_pkg::RGBDEstimator::run,
            py::arg("images_to_match"),
            py::arg("images_to_stitch"),
            "Estimate RGB-D panorama from fisheye images (matches Python estimate_RGBD_panorama)\n\n"
            "Args:\n"
            "    images_to_match: List of [H, W, 3] float32 images for stereo matching\n"
            "    images_to_stitch: List of [H, W, 3] float32 images for RGB stitching\n\n"
            "Returns:\n"
            "    Tuple of (RGB panorama [H, W, 3] uint8, distance panorama [H, W] float32)");
    
    // Temporarily comment out CUDA wrapper functions until signatures are fixed
    /*
    // CUDA wrapper functions
    m.def("reproject_distance", &launch_reproject_distance, 
          "Reproject distance map using z-buffering",
          py::arg("distance_in"), py::arg("distance_out"), 
          py::arg("cam_params"), py::arg("translation"),
          py::arg("cols"), py::arg("rows"));
    
    m.def("create_inpainting_weights", &launch_create_inpainting_weights,
          "Create inpainting weights based on occlusion direction",
          py::arg("inpaint_weights"), py::arg("cam_params"), 
          py::arg("translation"), py::arg("cols"), py::arg("rows"),
          py::arg("min_dist"), py::arg("max_dist"));
    
    m.def("inpaint", &launch_inpaint,
          "Fill holes in distance map using inpainting",
          py::arg("distance_map"), py::arg("inpaint_weights"),
          py::arg("cols"), py::arg("rows"), py::arg("max_dist"));
    
    m.def("create_blending_lut", &launch_create_blending_lut,
          "Create sampling and blending lookup tables",
          py::arg("sampling_lut"), py::arg("blending_weights"),
          py::arg("masks"), py::arg("cam_params_list"),
          py::arg("rotations"), py::arg("translations"),
          py::arg("pano_cols"), py::arg("pano_rows"),
          py::arg("cols"), py::arg("rows"),
          py::arg("min_dist"), py::arg("max_dist"));
    
    m.def("merge_rgbd_panorama", &launch_merge_rgbd_panorama,
          "Merge RGBD panorama from multiple cameras",
          py::arg("sampling_lut"), py::arg("blending_weights"),
          py::arg("reprojected_distance_maps"), py::arg("distance_maps"),
          py::arg("stitching_imgs"), py::arg("translations"),
          py::arg("cam_params_list"), py::arg("distance_panorama"),
          py::arg("rgb_panorama"), py::arg("pano_cols"), py::arg("pano_rows"),
          py::arg("cols"), py::arg("rows"),
          py::arg("stitching_imgs_rows"), py::arg("stitching_imgs_cols"));
    */
    
    // CUDA kernel helper structs
    py::class_<Intrinsics>(m, "Intrinsics")
        .def(py::init<>())
        .def_readwrite("fl", &Intrinsics::fl)
        .def_readwrite("principal", &Intrinsics::principal)
        .def_readwrite("xi", &Intrinsics::xi)
        .def_readwrite("alpha", &Intrinsics::alpha);
    
    py::class_<Rotation>(m, "Rotation")
        .def(py::init<>());
        
    py::class_<CamParams>(m, "CamParams")
        .def(py::init<>())
        .def_readwrite("intrinsics", &CamParams::intrinsics);
        
    py::class_<RotationParams>(m, "RotationParams")
        .def(py::init<>())
        .def_readwrite("rotation", &RotationParams::rotation);
}