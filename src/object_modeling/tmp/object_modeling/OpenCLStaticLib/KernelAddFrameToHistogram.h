#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelAddFrameToHistogram
{
public:
	static const std::string kernel_name;

	KernelAddFrameToHistogram(OpenCLAllKernels & opencl_kernels);

    void runKernel(VolumeBuffer & buffer_histogram_bin,
        ImageBuffer const& buffer_depth_image,
        ImageBuffer const& buffer_segments,
        int which_segment,
        float bin_min,
        float bin_max,
        float voxel_size,
        const Eigen::Affine3f& pose,
        const Eigen::Array2f& focal_lengths,
        const Eigen::Array2f& camera_centers
        , const float min_truncation_distance);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
