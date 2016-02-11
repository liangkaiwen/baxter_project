#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelExtractVolumeFloat
{
public:
	static const std::string kernel_name;

	KernelExtractVolumeFloat(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume,
		Eigen::Array3i const& voxel,
		float & result_float
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};