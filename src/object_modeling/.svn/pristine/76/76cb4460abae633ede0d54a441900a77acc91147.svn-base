#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelMarkPointsViolateEmpty
{
public:
	static const std::string kernel_name;

	KernelMarkPointsViolateEmpty(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume,
            VolumeBuffer & weights,
            cl::Buffer & points,
            cl::Buffer & result_violates_empty,
            size_t points_size,
            const float voxel_size,
            const Eigen::Affine3f& volume_pose,
                float min_value_invalid
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
