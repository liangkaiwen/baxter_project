#pragma once

#include "OpenCLAllKernels.h"

class KernelMinAbsFloatsWithWeightsRecordIndex
{
public:
    static const std::string kernel_name;

    KernelMinAbsFloatsWithWeightsRecordIndex(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		cl::Buffer & running_values,
		cl::Buffer & running_weights,
		cl::Buffer & runnin_index,
		cl::Buffer & to_min_values,
		cl::Buffer & to_min_weights,
		int this_index,
		size_t size);

protected:
    CL & cl_;
    cl::Kernel kernel_;
};
