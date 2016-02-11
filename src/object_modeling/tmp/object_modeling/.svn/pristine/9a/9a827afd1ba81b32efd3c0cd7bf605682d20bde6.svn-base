#pragma once

#include "OpenCLAllKernels.h"

class KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction
{
public:
    static const std::string kernel_name;

    KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction(OpenCLAllKernels & opencl_kernels);

	void runKernel(
	cl::Buffer & running_values,
	cl::Buffer & running_weights,
	cl::Buffer & to_min_values,
	cl::Buffer & to_min_weights,
	cl::Buffer & max_weights,
	float min_weight_fraction,
	size_t size);

protected:
    CL & cl_;
    cl::Kernel kernel_;
};
