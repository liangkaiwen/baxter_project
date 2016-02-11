#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelHistogramSumCheckIndex
{
public:
	static const std::string kernel_name;

    KernelHistogramSumCheckIndex(OpenCLAllKernels & opencl_kernels);

    void runKernel(
		VolumeBuffer & buffer_histogram_bin,
        VolumeBuffer & buffer_index,
		VolumeBuffer & buffer_sum,
		VolumeBuffer & buffer_count,
        int this_index,
        int index_range,
		float bin_min,
		float bin_max
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
