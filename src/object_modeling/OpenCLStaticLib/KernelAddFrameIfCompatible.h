#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelAddFrameIfCompatible
{
public:
	static const std::string kernel_name;

	KernelAddFrameIfCompatible(OpenCLAllKernels & opencl_kernels);

    void runKernel(VolumeBuffer & buffer_mean,
        VolumeBuffer & buffer_count,
        ImageBuffer const& buffer_depth_image,
        ImageBuffer const& buffer_normals_image,
        ImageBuffer const& buffer_segments,
        const int which_segment,
        const float voxel_size,
        const Eigen::Affine3f& model_pose,
        const Eigen::Array2f& camera_focal,
        const Eigen::Array2f& camera_center,
        const Eigen::Vector3f this_volume_normal,
        const float min_dot_product,
        const bool empty_always_included,
        const bool cos_weight,
                   const float min_truncation_distance);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
