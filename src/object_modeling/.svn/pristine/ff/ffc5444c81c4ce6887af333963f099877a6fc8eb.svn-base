#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelHistogramVariance
{
public:
	static const std::string kernel_name;

    KernelHistogramVariance(OpenCLAllKernels & opencl_kernels);

    void runKernel(
		VolumeBuffer & buffer_histogram_bin,
        VolumeBuffer & buffer_mean,
		VolumeBuffer & buffer_count,
        VolumeBuffer & buffer_variance,
		float bin_min,
		float bin_max
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
