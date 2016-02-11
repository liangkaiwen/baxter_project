#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelBetterNormal
{
public:
	static const std::string kernel_name;

    KernelBetterNormal(OpenCLAllKernels & opencl_kernels);

    void runKernel(
            VolumeBuffer & volume_1,
            VolumeBuffer & volume_counts_1,
            VolumeBuffer & volume_normals_1,
            VolumeBuffer & volume_2,
            VolumeBuffer & volume_counts_2,
            VolumeBuffer & volume_normals_2,
            VolumeBuffer & result_volume,
            VolumeBuffer & result_counts,
            Eigen::Affine3f const& pose,
			float minimum_relative_count
            );

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
