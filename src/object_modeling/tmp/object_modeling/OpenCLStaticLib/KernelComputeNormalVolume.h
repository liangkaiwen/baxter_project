#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelComputeNormalVolume
{
public:
	static const std::string kernel_name;

    KernelComputeNormalVolume(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume,
		VolumeBuffer & result
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
