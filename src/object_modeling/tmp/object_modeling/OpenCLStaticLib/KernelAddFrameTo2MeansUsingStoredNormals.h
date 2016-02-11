#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelAddFrameTo2MeansUsingStoredNormals
{
public:
	static const std::string kernel_name;

    KernelAddFrameTo2MeansUsingStoredNormals(OpenCLAllKernels & opencl_kernels);

    void runKernel(VolumeBuffer & buffer_mean_1,
            VolumeBuffer & buffer_count_1,
            VolumeBuffer & buffer_normals_1,
            VolumeBuffer & buffer_mean_2,
            VolumeBuffer & buffer_count_2,
            VolumeBuffer & buffer_normals_2,
            ImageBuffer const& buffer_depth_image,
            ImageBuffer const& buffer_normals_image,
            ImageBuffer const& buffer_segments,
            int which_segment,
            float voxel_size,
            const Eigen::Affine3f& model_pose,
            const Eigen::Array2f& camera_focal,
            const Eigen::Array2f& camera_center
            , const float min_truncation_distance);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
