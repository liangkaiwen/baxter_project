#pragma once

#include "OpenCLAllKernels.h"

class KernelAddFloats
{
public:
	static const std::string kernel_name;

    KernelAddFloats(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & buffer, cl::Buffer & to_be_added, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
