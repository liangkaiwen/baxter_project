#pragma once

#include "OpenCLAllKernels.h"

class KernelAddFloatsWithWeights
{
public:
	static const std::string kernel_name;

    KernelAddFloatsWithWeights(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & running_values, cl::Buffer & running_weights, cl::Buffer & to_add_values, cl::Buffer & to_add_weights, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
