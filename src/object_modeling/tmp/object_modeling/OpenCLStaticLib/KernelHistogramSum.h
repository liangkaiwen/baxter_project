#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelHistogramSum
{
public:
	static const std::string kernel_name;

	KernelHistogramSum(OpenCLAllKernels & opencl_kernels);

    void runKernel(
		VolumeBuffer & buffer_histogram_bin,
		VolumeBuffer & buffer_sum,
		VolumeBuffer & buffer_count,
		float bin_min,
		float bin_max
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
