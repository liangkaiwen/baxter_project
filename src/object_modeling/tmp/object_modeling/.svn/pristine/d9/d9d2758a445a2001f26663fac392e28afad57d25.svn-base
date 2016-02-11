#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelExtractVolumeSlice
{
public:
	static const std::string kernel_name;

	KernelExtractVolumeSlice(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume,
		int axis, int position,
		ImageBuffer & result
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};