#pragma once

#include "OpenCLAllKernels.h"

class KernelMinAbsFloatsWithWeights
{
public:
    static const std::string kernel_name;

    KernelMinAbsFloatsWithWeights(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		cl::Buffer & running_values,
		cl::Buffer & running_weights,
		cl::Buffer & to_min_values,
		cl::Buffer & to_min_weights,
		size_t size);

protected:
    CL & cl_;
    cl::Kernel kernel_;
};
