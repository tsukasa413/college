/**
 * @file Stitcher.cpp
 * @brief Implementation of C++ wrapper class for CUDA-based panorama stitching
 * 
 * @copyright CC BY-NC-SA 3.0
 */

#include "sphere_stereo_ros/Stitcher.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace sphere_stereo_ros {

// =============================================================================
// Constructor / Destructor
// =============================================================================

Stitcher::Stitcher(const CalibrationSet& calibration, const Config& config)
    : calibration_(calibration)
    , config_(config)
    , num_cameras_(static_cast<int>(calibration.calibrations().size()))
    , initialized_(false)
    , stream_(nullptr)
    , d_intrinsics_(nullptr)
    , d_rotations_(nullptr)
    , d_translations_(nullptr)
    , d_masks_(nullptr)
    , d_sampling_lut_(nullptr)
    , d_blending_weights_(nullptr)
    , d_inpaint_weights_(nullptr)
    , d_stitch_images_(nullptr)
    , d_distance_maps_(nullptr)
    , d_reprojected_distances_(nullptr)
    , d_rgb_panorama_(nullptr)
    , d_distance_panorama_(nullptr)
    , h_rgb_panorama_(nullptr)
    , h_distance_panorama_(nullptr)
{
    cuda_config_.pano_cols = config_.pano_width;
    cuda_config_.pano_rows = config_.pano_height;
    cuda_config_.fisheye_cols = config_.fisheye_width;
    cuda_config_.fisheye_rows = config_.fisheye_height;
    cuda_config_.stitch_cols = config_.stitch_width;
    cuda_config_.stitch_rows = config_.stitch_height;
    cuda_config_.num_references = num_cameras_;
    cuda_config_.min_dist = config_.min_dist;
    cuda_config_.max_dist = config_.max_dist;
}

Stitcher::~Stitcher()
{
    freeGPUMemory();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

Stitcher::Stitcher(Stitcher&& other) noexcept
    : calibration_(std::move(other.calibration_))
    , config_(other.config_)
    , cuda_config_(other.cuda_config_)
    , num_cameras_(other.num_cameras_)
    , initialized_(other.initialized_)
    , stream_(other.stream_)
    , d_intrinsics_(other.d_intrinsics_)
    , d_rotations_(other.d_rotations_)
    , d_translations_(other.d_translations_)
    , d_masks_(other.d_masks_)
    , d_sampling_lut_(other.d_sampling_lut_)
    , d_blending_weights_(other.d_blending_weights_)
    , d_inpaint_weights_(other.d_inpaint_weights_)
    , d_stitch_images_(other.d_stitch_images_)
    , d_distance_maps_(other.d_distance_maps_)
    , d_reprojected_distances_(other.d_reprojected_distances_)
    , d_rgb_panorama_(other.d_rgb_panorama_)
    , d_distance_panorama_(other.d_distance_panorama_)
    , h_rgb_panorama_(other.h_rgb_panorama_)
    , h_distance_panorama_(other.h_distance_panorama_)
{
    other.stream_ = nullptr;
    other.d_intrinsics_ = nullptr;
    other.d_rotations_ = nullptr;
    other.d_translations_ = nullptr;
    other.d_masks_ = nullptr;
    other.d_sampling_lut_ = nullptr;
    other.d_blending_weights_ = nullptr;
    other.d_inpaint_weights_ = nullptr;
    other.d_stitch_images_ = nullptr;
    other.d_distance_maps_ = nullptr;
    other.d_reprojected_distances_ = nullptr;
    other.d_rgb_panorama_ = nullptr;
    other.d_distance_panorama_ = nullptr;
    other.h_rgb_panorama_ = nullptr;
    other.h_distance_panorama_ = nullptr;
    other.initialized_ = false;
}

Stitcher& Stitcher::operator=(Stitcher&& other) noexcept
{
    if (this != &other) {
        freeGPUMemory();
        if (stream_) cudaStreamDestroy(stream_);

        calibration_ = std::move(other.calibration_);
        config_ = other.config_;
        cuda_config_ = other.cuda_config_;
        num_cameras_ = other.num_cameras_;
        initialized_ = other.initialized_;
        stream_ = other.stream_;
        d_intrinsics_ = other.d_intrinsics_;
        d_rotations_ = other.d_rotations_;
        d_translations_ = other.d_translations_;
        d_masks_ = other.d_masks_;
        d_sampling_lut_ = other.d_sampling_lut_;
        d_blending_weights_ = other.d_blending_weights_;
        d_inpaint_weights_ = other.d_inpaint_weights_;
        d_stitch_images_ = other.d_stitch_images_;
        d_distance_maps_ = other.d_distance_maps_;
        d_reprojected_distances_ = other.d_reprojected_distances_;
        d_rgb_panorama_ = other.d_rgb_panorama_;
        d_distance_panorama_ = other.d_distance_panorama_;
        h_rgb_panorama_ = other.h_rgb_panorama_;
        h_distance_panorama_ = other.h_distance_panorama_;

        other.stream_ = nullptr;
        other.d_intrinsics_ = nullptr;
        other.d_rotations_ = nullptr;
        other.d_translations_ = nullptr;
        other.d_masks_ = nullptr;
        other.d_sampling_lut_ = nullptr;
        other.d_blending_weights_ = nullptr;
        other.d_inpaint_weights_ = nullptr;
        other.d_stitch_images_ = nullptr;
        other.d_distance_maps_ = nullptr;
        other.d_reprojected_distances_ = nullptr;
        other.d_rgb_panorama_ = nullptr;
        other.d_distance_panorama_ = nullptr;
        other.h_rgb_panorama_ = nullptr;
        other.h_distance_panorama_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

void Stitcher::initialize()
{
    if (initialized_) return;

    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + 
                                 std::string(cudaGetErrorString(err)));
    }

    allocateGPUMemory();
    uploadCalibration();
    computeMasks();
    computeLookupTables();
    computeInpaintingWeights();
    cudaStreamSynchronize(stream_);

    initialized_ = true;
}

void Stitcher::allocateGPUMemory()
{
    cudaError_t err;
    const int pano_size = config_.pano_width * config_.pano_height;
    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;
    const int stitch_size = config_.stitch_width * config_.stitch_height;

    err = cudaMalloc(&d_intrinsics_, num_cameras_ * sizeof(cuda::Intrinsics));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_intrinsics_ failed");
    
    err = cudaMalloc(&d_rotations_, num_cameras_ * sizeof(cuda::Rotation));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_rotations_ failed");
    
    err = cudaMalloc(&d_translations_, num_cameras_ * sizeof(float3));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_translations_ failed");
    
    err = cudaMalloc(&d_masks_, num_cameras_ * fisheye_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_masks_ failed");

    err = cudaMalloc(&d_sampling_lut_, num_cameras_ * pano_size * sizeof(float2));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_sampling_lut_ failed");
    
    err = cudaMalloc(&d_blending_weights_, num_cameras_ * pano_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_blending_weights_ failed");
    
    err = cudaMalloc(&d_inpaint_weights_, num_cameras_ * fisheye_size * sizeof(uchar2));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_inpaint_weights_ failed");

    err = cudaMalloc(&d_stitch_images_, num_cameras_ * stitch_size * sizeof(uchar3));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_stitch_images_ failed");
    
    err = cudaMalloc(&d_distance_maps_, num_cameras_ * fisheye_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_distance_maps_ failed");
    
    err = cudaMalloc(&d_reprojected_distances_, num_cameras_ * fisheye_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_reprojected_distances_ failed");

    err = cudaMalloc(&d_rgb_panorama_, pano_size * sizeof(uchar3));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_rgb_panorama_ failed");
    
    err = cudaMalloc(&d_distance_panorama_, pano_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_distance_panorama_ failed");

    err = cudaMallocHost(&h_rgb_panorama_, pano_size * sizeof(uchar3));
    if (err != cudaSuccess) throw std::runtime_error("cudaMallocHost h_rgb_panorama_ failed");
    
    err = cudaMallocHost(&h_distance_panorama_, pano_size * sizeof(float));
    if (err != cudaSuccess) throw std::runtime_error("cudaMallocHost h_distance_panorama_ failed");
}

void Stitcher::freeGPUMemory()
{
    if (d_intrinsics_) { cudaFree(d_intrinsics_); d_intrinsics_ = nullptr; }
    if (d_rotations_) { cudaFree(d_rotations_); d_rotations_ = nullptr; }
    if (d_translations_) { cudaFree(d_translations_); d_translations_ = nullptr; }
    if (d_masks_) { cudaFree(d_masks_); d_masks_ = nullptr; }
    if (d_sampling_lut_) { cudaFree(d_sampling_lut_); d_sampling_lut_ = nullptr; }
    if (d_blending_weights_) { cudaFree(d_blending_weights_); d_blending_weights_ = nullptr; }
    if (d_inpaint_weights_) { cudaFree(d_inpaint_weights_); d_inpaint_weights_ = nullptr; }
    if (d_stitch_images_) { cudaFree(d_stitch_images_); d_stitch_images_ = nullptr; }
    if (d_distance_maps_) { cudaFree(d_distance_maps_); d_distance_maps_ = nullptr; }
    if (d_reprojected_distances_) { cudaFree(d_reprojected_distances_); d_reprojected_distances_ = nullptr; }
    if (d_rgb_panorama_) { cudaFree(d_rgb_panorama_); d_rgb_panorama_ = nullptr; }
    if (d_distance_panorama_) { cudaFree(d_distance_panorama_); d_distance_panorama_ = nullptr; }
    if (h_rgb_panorama_) { cudaFreeHost(h_rgb_panorama_); h_rgb_panorama_ = nullptr; }
    if (h_distance_panorama_) { cudaFreeHost(h_distance_panorama_); h_distance_panorama_ = nullptr; }
}

void Stitcher::uploadCalibration()
{
    std::vector<cuda::Intrinsics> intrinsics(num_cameras_);
    std::vector<cuda::Rotation> rotations(num_cameras_);
    std::vector<float3> translations(num_cameras_);

    const auto& calibs = calibration_.calibrations();
    for (int i = 0; i < num_cameras_; ++i) {
        const auto& calib = calibs[i];
        
        Vec2f fl = calib.scaledFocalLength();
        Vec2f pp = calib.scaledPrincipal();
        
        intrinsics[i].fl.x = fl.x();
        intrinsics[i].fl.y = fl.y();
        intrinsics[i].principal.x = pp.x();
        intrinsics[i].principal.y = pp.y();
        intrinsics[i].xi = calib.xi();
        intrinsics[i].alpha = calib.alpha();

        // Python: rotations = [torch.inverse(calibration.rt[:3, :3]) ...]
        // Python: translation = torch.matmul(torch.inverse(calibration.rt), [0,0,0,1])[:3]
        // = inv(RT)[:, 3][:3] = -R^T @ t
        const Mat4f& rt = calib.rt();
        
        // R^T (inverse rotation = transpose)
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                rotations[i].r[row][col] = rt(col, row);
            }
        }

        // -R^T @ t
        float tx = rt(0, 3), ty = rt(1, 3), tz = rt(2, 3);
        translations[i].x = -(rt(0, 0) * tx + rt(1, 0) * ty + rt(2, 0) * tz);
        translations[i].y = -(rt(0, 1) * tx + rt(1, 1) * ty + rt(2, 1) * tz);
        translations[i].z = -(rt(0, 2) * tx + rt(1, 2) * ty + rt(2, 2) * tz);
    }

    cudaMemcpyAsync(d_intrinsics_, intrinsics.data(), 
                    num_cameras_ * sizeof(cuda::Intrinsics), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_rotations_, rotations.data(),
                    num_cameras_ * sizeof(cuda::Rotation), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_translations_, translations.data(),
                    num_cameras_ * sizeof(float3), cudaMemcpyHostToDevice, stream_);
}

void Stitcher::computeMasks()
{
    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;
    std::vector<float> masks(num_cameras_ * fisheye_size);

    const auto& calibs = calibration_.calibrations();
    for (int cam = 0; cam < num_cameras_; ++cam) {
        const auto& calib = calibs[cam];
        float* mask_ptr = masks.data() + cam * fisheye_size;
        Vec2f pp = calib.scaledPrincipal();

        for (int v = 0; v < config_.fisheye_height; ++v) {
            for (int u = 0; u < config_.fisheye_width; ++u) {
                Vec2f pixel(static_cast<float>(u), static_cast<float>(v));
                bool valid_unproj = false;
                Vec3f pt = calib.unproject(pixel, valid_unproj);
                
                bool valid_proj = false;
                Vec2f reproj = calib.project(pt, valid_proj);
                float error = (reproj - pixel).norm();
                float dist_to_center = (pixel - pp).norm();
                float max_radius = std::min(pp.x(), pp.y()) * 0.95f;

                bool valid = valid_unproj && valid_proj && (error < 1.0f) && (dist_to_center < max_radius);
                mask_ptr[v * config_.fisheye_width + u] = valid ? 1.0f : 0.0f;
            }
        }
    }

    // Box filter smoothing (smoothing_radius=15)
    const int sr = 15;
    std::vector<float> smoothed(num_cameras_ * fisheye_size);
    
    for (int cam = 0; cam < num_cameras_; ++cam) {
        const float* src = masks.data() + cam * fisheye_size;
        float* dst = smoothed.data() + cam * fisheye_size;
        
        for (int v = 0; v < config_.fisheye_height; ++v) {
            for (int u = 0; u < config_.fisheye_width; ++u) {
                float sum = 0.0f;
                int count = 0;
                for (int dv = -sr; dv <= sr; ++dv) {
                    for (int du = -sr; du <= sr; ++du) {
                        int vv = v + dv, uu = u + du;
                        float val = 1.0f;  // pad with 1.0
                        if (vv >= 0 && vv < config_.fisheye_height && 
                            uu >= 0 && uu < config_.fisheye_width) {
                            val = src[vv * config_.fisheye_width + uu];
                        }
                        sum += val;
                        count++;
                    }
                }
                dst[v * config_.fisheye_width + u] = sum / static_cast<float>(count);
            }
        }
    }

    cudaMemcpyAsync(d_masks_, smoothed.data(),
                    num_cameras_ * fisheye_size * sizeof(float), cudaMemcpyHostToDevice, stream_);
}

void Stitcher::computeLookupTables()
{
    cuda::launchCreateBlendingLutKernel(
        d_sampling_lut_, d_blending_weights_, d_masks_,
        d_intrinsics_, d_rotations_, d_translations_,
        cuda_config_, stream_);
    
    cudaStreamSynchronize(stream_);
    
    // Smooth blending weights on CPU
    const int pano_size = config_.pano_width * config_.pano_height;
    const int sr = 15;
    
    std::vector<float> h_weights(num_cameras_ * pano_size);
    cudaMemcpy(h_weights.data(), d_blending_weights_, 
               num_cameras_ * pano_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<float> smoothed(num_cameras_ * pano_size);
    
    for (int cam = 0; cam < num_cameras_; ++cam) {
        const float* src = h_weights.data() + cam * pano_size;
        float* dst = smoothed.data() + cam * pano_size;
        
        for (int v = 0; v < config_.pano_height; ++v) {
            for (int u = 0; u < config_.pano_width; ++u) {
                float sum = 0.0f;
                int count = 0;
                for (int dv = -sr; dv <= sr; ++dv) {
                    for (int du = -sr; du <= sr; ++du) {
                        int vv = v + dv;
                        int uu = (u + du + config_.pano_width) % config_.pano_width;
                        if (vv >= 0 && vv < config_.pano_height) {
                            sum += src[vv * config_.pano_width + uu];
                            count++;
                        }
                    }
                }
                dst[v * config_.pano_width + u] = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
            }
        }
    }
    
    // Normalize across cameras
    for (int idx = 0; idx < pano_size; ++idx) {
        float weight_sum = 0.0f;
        for (int cam = 0; cam < num_cameras_; ++cam) {
            weight_sum += smoothed[cam * pano_size + idx];
        }
        if (weight_sum > 1e-8f) {
            for (int cam = 0; cam < num_cameras_; ++cam) {
                smoothed[cam * pano_size + idx] /= weight_sum;
            }
        }
    }
    
    cudaMemcpyAsync(d_blending_weights_, smoothed.data(),
                    num_cameras_ * pano_size * sizeof(float), cudaMemcpyHostToDevice, stream_);
}

void Stitcher::computeInpaintingWeights()
{
    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;
    for (int cam = 0; cam < num_cameras_; ++cam) {
        cuda::launchCreateInpaintingWeightsKernel(
            d_inpaint_weights_ + cam * fisheye_size,
            d_intrinsics_ + cam, d_translations_ + cam,
            config_.fisheye_width, config_.fisheye_height,
            config_.min_dist, config_.max_dist, stream_);
    }
}

void Stitcher::uploadImages(const std::vector<cv::Mat>& images)
{
    if (!initialized_) throw std::runtime_error("Stitcher not initialized");
    if (static_cast<int>(images.size()) != num_cameras_) {
        throw std::runtime_error("Expected " + std::to_string(num_cameras_) + 
                                 " images, got " + std::to_string(images.size()));
    }

    const int stitch_size = config_.stitch_width * config_.stitch_height;

    for (int i = 0; i < num_cameras_; ++i) {
        const cv::Mat& img = images[i];
        if (img.cols != config_.stitch_width || img.rows != config_.stitch_height) {
            throw std::runtime_error("Image size mismatch for camera " + std::to_string(i));
        }

        cv::Mat rgb;
        if (img.channels() == 3) {
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        } else {
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
        }
        if (!rgb.isContinuous()) rgb = rgb.clone();

        cudaMemcpyAsync(d_stitch_images_ + i * stitch_size, rgb.data,
                        stitch_size * sizeof(uchar3), cudaMemcpyHostToDevice, stream_);
    }
}

void Stitcher::uploadDistances(const std::vector<cv::Mat>& distances)
{
    if (!initialized_) throw std::runtime_error("Stitcher not initialized");
    if (static_cast<int>(distances.size()) != num_cameras_) {
        throw std::runtime_error("Expected " + std::to_string(num_cameras_) + 
                                 " distance maps, got " + std::to_string(distances.size()));
    }

    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;

    for (int i = 0; i < num_cameras_; ++i) {
        const cv::Mat& dist = distances[i];
        if (dist.cols != config_.fisheye_width || dist.rows != config_.fisheye_height) {
            throw std::runtime_error("Distance map size mismatch for camera " + std::to_string(i));
        }
        if (dist.type() != CV_32FC1) {
            throw std::runtime_error("Distance map must be CV_32FC1");
        }

        const cv::Mat* src = &dist;
        cv::Mat contiguous;
        if (!dist.isContinuous()) { contiguous = dist.clone(); src = &contiguous; }

        cudaMemcpyAsync(d_distance_maps_ + i * fisheye_size, src->data,
                        fisheye_size * sizeof(float), cudaMemcpyHostToDevice, stream_);
    }
}

void Stitcher::stitch()
{
    if (!initialized_) throw std::runtime_error("Stitcher not initialized");
    reprojectDistances();
    applyInpainting();
    cuda::launchMergeRGBDPanoramaKernel(
        d_sampling_lut_, d_blending_weights_, d_reprojected_distances_,
        d_distance_maps_, d_stitch_images_, d_translations_, d_intrinsics_,
        d_distance_panorama_, d_rgb_panorama_, cuda_config_, stream_);
}

void Stitcher::reprojectDistances()
{
    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;
    const float invalid_dist = 1e8f;

    for (int cam = 0; cam < num_cameras_; ++cam) {
        float* reproj_ptr = d_reprojected_distances_ + cam * fisheye_size;
        cuda::launchFillKernel(reproj_ptr, invalid_dist, fisheye_size, stream_);
        for (int pass = 0; pass < 2; ++pass) {
            cuda::launchReprojectDistanceKernel(
                d_distance_maps_ + cam * fisheye_size, reproj_ptr,
                d_intrinsics_ + cam, d_translations_ + cam,
                config_.fisheye_width, config_.fisheye_height, stream_);
        }
    }
}

void Stitcher::applyInpainting()
{
    const int fisheye_size = config_.fisheye_width * config_.fisheye_height;
    for (int iter = 0; iter < config_.inpaint_iterations; ++iter) {
        for (int cam = 0; cam < num_cameras_; ++cam) {
            cuda::launchInpaintKernel(
                d_reprojected_distances_ + cam * fisheye_size,
                d_inpaint_weights_ + cam * fisheye_size,
                config_.fisheye_width, config_.fisheye_height,
                config_.max_dist, stream_);
        }
    }
}

cv::Mat Stitcher::downloadRGBPanorama()
{
    if (!initialized_) throw std::runtime_error("Stitcher not initialized");
    const int pano_size = config_.pano_width * config_.pano_height;
    cudaMemcpyAsync(h_rgb_panorama_, d_rgb_panorama_,
                    pano_size * sizeof(uchar3), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    cv::Mat rgb(config_.pano_height, config_.pano_width, CV_8UC3, h_rgb_panorama_);
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

cv::Mat Stitcher::downloadDistancePanorama()
{
    if (!initialized_) throw std::runtime_error("Stitcher not initialized");
    const int pano_size = config_.pano_width * config_.pano_height;
    cudaMemcpyAsync(h_distance_panorama_, d_distance_panorama_,
                    pano_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    cv::Mat result(config_.pano_height, config_.pano_width, CV_32FC1, h_distance_panorama_);
    return result.clone();
}

void Stitcher::synchronize()
{
    if (stream_) cudaStreamSynchronize(stream_);
}

}  // namespace sphere_stereo_ros
