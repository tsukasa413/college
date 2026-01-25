// ============================================================================
// Cost Volume Based Depth Estimation (Python-equivalent pipeline)
// ============================================================================

/**
 * Generate full cost volume for all distance candidates
 * Output: [candidate_count, height, width] cost values
 */
__global__ void compute_cost_volume_kernel_impl(
    float* d_cost_volume,           // [candidate_count * height * width]
    const uchar4* d_reference_image,
    const uchar4* const* d_images,
    const int* d_selected_cameras,
    const DoubleSphereCalibration* d_calibrations,
    const DoubleSphereCalibration reference_calib,
    int num_cameras,
    cudaTextureObject_t* d_texobjs
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int dist_idx = blockIdx.z;  // Depth candidate index
    
    if (pixel_x >= d_config_constant.matching_width || pixel_y >= d_config_constant.matching_height)
        return;
    
    int pixel_idx = pixel_y * d_config_constant.matching_width + pixel_x;
    int cost_idx = dist_idx * (d_config_constant.matching_width * d_config_constant.matching_height) + pixel_idx;
    
    // Unproject reference pixel
    float2 uv_ref = make_float2((float)pixel_x, (float)pixel_y);
    float3 pt_unit = unproject_double_sphere(uv_ref, reference_calib);
    
    if (pt_unit.z <= 0.0f) {
        d_cost_volume[cost_idx] = 500.0f;  // Max cost for invalid
        return;
    }
    
    // Get selected camera
    int selected_cam = d_selected_cameras[pixel_idx];
    if (selected_cam < 0) {
        d_cost_volume[cost_idx] = 500.0f;
        return;
    }
    
    const DoubleSphereCalibration& selected_calib = d_calib_constant[selected_cam];
    
    // Compute distance for this candidate
    float inv_dist_min = 1.0f / d_config_constant.min_dist;
    float inv_dist_max = 1.0f / d_config_constant.max_dist;
    float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                     ((float)dist_idx / (float)(d_config_constant.candidate_count - 1));
    float distance = 1.0f / inv_dist;
    
    // 3D point
    float3 pt_3d = distance * pt_unit;
    
    // Transform to selected camera
    float3 pt_cam = transform_point(pt_3d, selected_calib);
    
    // Project
    float2 uv_proj = project_double_sphere(pt_cam, selected_calib);
    
    if (uv_proj.x < 0 || uv_proj.x >= selected_calib.width || 
        uv_proj.y < 0 || uv_proj.y >= selected_calib.height) {
        d_cost_volume[cost_idx] = 500.0f;
        return;
    }
    
    // Sample via texture
    float2 uv_normalized = make_float2(
        (uv_proj.x + 0.5f) / selected_calib.width,
        (uv_proj.y + 0.5f) / selected_calib.height
    );
    float4 sampled_color = tex2D<float4>(d_texobjs[selected_cam], uv_normalized.x, uv_normalized.y);
    
    // Compute SAD cost
    uchar4 ref_color = d_reference_image[pixel_idx];
    float3 ref_rgb = make_float3((float)ref_color.x, (float)ref_color.y, (float)ref_color.z);
    float3 tgt_rgb = make_float3(sampled_color.x, sampled_color.y, sampled_color.z);
    float cost = absSum(ref_rgb - tgt_rgb);
    
    // Python: cost_volume = torch.clamp(cost_volume, max=500)
    cost = fminf(cost, 500.0f);
    
    d_cost_volume[cost_idx] = cost;
}

/**
 * Select distance from filtered cost volume with quadratic fitting
 */
__global__ void select_distance_from_cost_volume_kernel_impl(
    float* d_distance_map,
    const float* d_cost_volume,
    int width,
    int height
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= width || pixel_y >= height)
        return;
    
    int pixel_idx = pixel_y * width + pixel_x;
    
    // Find minimum cost
    float min_cost = FLT_MAX;
    int min_idx = d_config_constant.candidate_count - 1;
    
    for (int d = 0; d < d_config_constant.candidate_count; d++) {
        int cost_idx = d * (width * height) + pixel_idx;
        float cost = d_cost_volume[cost_idx];
        if (cost < min_cost) {
            min_cost = cost;
            min_idx = d;
        }
    }
    
    // Quadratic fitting
    float variation = 0.0f;
    if (min_idx > 0 && min_idx < d_config_constant.candidate_count - 1) {
        int left_idx = (min_idx - 1) * (width * height) + pixel_idx;
        int center_idx = min_idx * (width * height) + pixel_idx;
        int right_idx = (min_idx + 1) * (width * height) + pixel_idx;
        
        float left_cost = d_cost_volume[left_idx];
        float center_cost = d_cost_volume[center_idx];
        float right_cost = d_cost_volume[right_idx];
        
        float denominator = left_cost + right_cost - 2.0f * center_cost + 1e-8f;
        if (fabsf(denominator) > 1e-6f) {
            variation = 0.5f * (left_cost - right_cost) / denominator;
            variation = fmaxf(-0.5f, fminf(0.5f, variation));
        }
    }
    
    float refined_idx = (float)min_idx + variation;
    
    // Convert to distance
    float inv_dist_min = 1.0f / d_config_constant.min_dist;
    float inv_dist_max = 1.0f / d_config_constant.max_dist;
    float inv_dist = inv_dist_min - (inv_dist_min - inv_dist_max) * 
                     (refined_idx / (float)(d_config_constant.candidate_count - 1));
    float distance = 1.0f / inv_dist;
    
    d_distance_map[pixel_idx] = distance;
}

