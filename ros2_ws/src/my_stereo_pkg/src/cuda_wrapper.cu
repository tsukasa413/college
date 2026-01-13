#include "my_stereo_pkg/cuda_kernels.hpp"
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <iostream>

// CUDA kernel function declarations (from stitcher.cu)
extern "C" __global__ void reprojectDistanceKernel(const float* distanceIn, float* distanceOut, 
    const Intrinsics* calib, const float3* translation, int cols, int rows);

extern "C" __global__ void createInpaintingWeightsKernel(uchar2* inpaintDirWeights, 
    const Intrinsics* calib, const float3* translation, int cols, int rows, float min_dist, float max_dist);

extern "C" __global__ void inpaintKernel(float* distanceMap, const uchar2* inpaintDirWeights, 
    int cols, int rows, float max_dist);

extern "C" __global__ void createBlendingLutKernel(float2* samplingLut, float* blendingWeights, 
    float* masks, const Intrinsics* calibs, const Rotation* rotations, const float3* translations,
    int pano_cols, int pano_rows, int cols, int rows, int references_count, float min_dist, float max_dist);

extern "C" __global__ void mergeRGBDPanoramaKernel(
    const float2* samplingLut, const float* blendingWeights, 
    const float* reprojectedDistanceMaps, const float* distanceMaps, 
    const uchar3* stitchingImgs, int stitchingImgsRows, int stitchingImgsCols, 
    const float3* translations, const Intrinsics* calibs, 
    float* DistancePanorama, uchar3* RGBPanorama,
    int pano_cols, int pano_rows, int cols, int rows, int references_count);

// Helper function to check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Helper function to convert torch tensor to device pointer
template<typename T>
T* get_device_ptr(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
    return tensor.data_ptr<T>();
}

// Wrapper function implementations
at::Tensor launch_reproject_distance(
    const at::Tensor& distance_in,
    const at::Tensor& distance_out,
    const Intrinsics& intrinsics,
    const at::Tensor& translation,
    int cols, int rows
) {
    TORCH_CHECK(distance_in.device() == distance_out.device(), "Input tensors must be on same device");
    TORCH_CHECK(distance_in.is_cuda(), "Tensors must be on CUDA device");
    
    // Get device pointers
    const float* d_distance_in = get_device_ptr<float>(distance_in);
    float* d_distance_out = get_device_ptr<float>(distance_out);
    const float3* d_translation = reinterpret_cast<const float3*>(get_device_ptr<float>(translation));
    
    // Copy calibration to device
    Intrinsics* d_calib;
    CUDA_CHECK(cudaMalloc(&d_calib, sizeof(Intrinsics)));
    CUDA_CHECK(cudaMemcpy(d_calib, &intrinsics, sizeof(Intrinsics), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int num_pixels = cols * rows;
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    
    reprojectDistanceKernel<<<gridSize, blockSize>>>(
        d_distance_in, d_distance_out, d_calib, d_translation, cols, rows
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_calib));
    
    return distance_out;
}

at::Tensor launch_create_inpainting_weights(
    const at::Tensor& inpaint_weights,
    const Intrinsics& intrinsics,
    const at::Tensor& translation,
    int cols, int rows,
    float min_dist, float max_dist
) {
    TORCH_CHECK(inpaint_weights.is_cuda(), "Tensor must be on CUDA device");
    
    // Get device pointers
    uchar2* d_inpaint_weights = reinterpret_cast<uchar2*>(get_device_ptr<uint8_t>(inpaint_weights));
    const float3* d_translation = reinterpret_cast<const float3*>(get_device_ptr<float>(translation));
    
    // Copy calibration to device
    Intrinsics* d_calib;
    CUDA_CHECK(cudaMalloc(&d_calib, sizeof(Intrinsics)));
    CUDA_CHECK(cudaMemcpy(d_calib, &intrinsics, sizeof(Intrinsics), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int num_pixels = cols * rows;
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    
    createInpaintingWeightsKernel<<<gridSize, blockSize>>>(
        d_inpaint_weights, d_calib, d_translation, cols, rows, min_dist, max_dist
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_calib));
    
    return inpaint_weights;
}

at::Tensor launch_inpaint(
    const at::Tensor& distance_map,
    const at::Tensor& inpaint_weights,
    int cols, int rows,
    float max_dist
) {
    TORCH_CHECK(distance_map.is_cuda(), "Tensors must be on CUDA device");
    TORCH_CHECK(inpaint_weights.is_cuda(), "Tensors must be on CUDA device");
    
    // Get device pointers
    float* d_distance_map = get_device_ptr<float>(distance_map);
    const uchar2* d_inpaint_weights = reinterpret_cast<const uchar2*>(get_device_ptr<uint8_t>(inpaint_weights));
    
    // Launch kernel
    int num_pixels = cols * rows;
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    
    inpaintKernel<<<gridSize, blockSize>>>(
        d_distance_map, d_inpaint_weights, cols, rows, max_dist
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return distance_map;
}

std::pair<at::Tensor, at::Tensor> launch_create_blending_lut(
    const at::Tensor& sampling_lut,
    const at::Tensor& blending_weights,
    const at::Tensor& masks,
    const std::vector<Intrinsics>& calibrations,
    const std::vector<RotationParams>& rotations,
    const at::Tensor& translations,
    int pano_cols, int pano_rows,
    int cols, int rows,
    float min_dist, float max_dist
) {
    int references_count = calibrations.size();
    TORCH_CHECK(references_count == rotations.size(), "Number of cameras and rotations must match");
    
    // Get device pointers
    float2* d_sampling_lut = reinterpret_cast<float2*>(get_device_ptr<float>(sampling_lut));
    float* d_blending_weights = get_device_ptr<float>(blending_weights);
    float* d_masks = get_device_ptr<float>(masks);
    const float3* d_translations = reinterpret_cast<const float3*>(get_device_ptr<float>(translations));
    
    // Copy calibrations and rotations to device
    Intrinsics* d_calibs;
    Rotation* d_rotations;
    CUDA_CHECK(cudaMalloc(&d_calibs, references_count * sizeof(Intrinsics)));
    CUDA_CHECK(cudaMalloc(&d_rotations, references_count * sizeof(Rotation)));
    
    for (int i = 0; i < references_count; i++) {
        CUDA_CHECK(cudaMemcpy(&d_calibs[i], &calibrations[i], sizeof(Intrinsics), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_rotations[i], &rotations[i].rotation, sizeof(Rotation), cudaMemcpyHostToDevice));
    }
    
    // Launch kernel
    int num_pixels = pano_cols * pano_rows;
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    
    createBlendingLutKernel<<<gridSize, blockSize>>>(
        d_sampling_lut, d_blending_weights, d_masks,
        d_calibs, d_rotations, d_translations,
        pano_cols, pano_rows, cols, rows, references_count, min_dist, max_dist
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_calibs));
    CUDA_CHECK(cudaFree(d_rotations));
    
    return std::make_pair(sampling_lut, blending_weights);
}

std::pair<at::Tensor, at::Tensor> launch_merge_rgbd_panorama(
    const at::Tensor& sampling_lut,
    const at::Tensor& blending_weights,
    const at::Tensor& reprojected_distance_maps,
    const at::Tensor& distance_maps,
    const at::Tensor& stitching_imgs,
    const at::Tensor& translations,
    const std::vector<Intrinsics>& calibrations,
    const at::Tensor& distance_panorama,
    const at::Tensor& rgb_panorama,
    int pano_cols, int pano_rows,
    int cols, int rows,
    int stitching_imgs_rows, int stitching_imgs_cols
) {
    int references_count = calibrations.size();
    
    // Get device pointers
    const float2* d_sampling_lut = reinterpret_cast<const float2*>(get_device_ptr<float>(sampling_lut));
    const float* d_blending_weights = get_device_ptr<float>(blending_weights);
    const float* d_reprojected_distance_maps = get_device_ptr<float>(reprojected_distance_maps);
    const float* d_distance_maps = get_device_ptr<float>(distance_maps);
    const uchar3* d_stitching_imgs = reinterpret_cast<const uchar3*>(get_device_ptr<uint8_t>(stitching_imgs));
    const float3* d_translations = reinterpret_cast<const float3*>(get_device_ptr<float>(translations));
    float* d_distance_panorama = get_device_ptr<float>(distance_panorama);
    uchar3* d_rgb_panorama = reinterpret_cast<uchar3*>(get_device_ptr<uint8_t>(rgb_panorama));
    
    // Copy calibrations to device
    Intrinsics* d_calibs;
    CUDA_CHECK(cudaMalloc(&d_calibs, references_count * sizeof(Intrinsics)));
    
    for (int i = 0; i < references_count; i++) {
        CUDA_CHECK(cudaMemcpy(&d_calibs[i], &calibrations[i], sizeof(Intrinsics), cudaMemcpyHostToDevice));
    }
    
    // Launch kernel
    int num_pixels = pano_cols * pano_rows;
    int blockSize = 256;
    int gridSize = (num_pixels + blockSize - 1) / blockSize;
    
    mergeRGBDPanoramaKernel<<<gridSize, blockSize>>>(
        d_sampling_lut, d_blending_weights,
        d_reprojected_distance_maps, d_distance_maps,
        d_stitching_imgs, stitching_imgs_rows, stitching_imgs_cols,
        d_translations, d_calibs,
        d_distance_panorama, d_rgb_panorama,
        pano_cols, pano_rows, cols, rows, references_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_calibs));
    
    return std::make_pair(distance_panorama, rgb_panorama);
}