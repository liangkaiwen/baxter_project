#pragma once

#include "OpenCLAllKernels.h"

class KernelSetInt
{
public:
	static const std::string kernel_name;

    KernelSetInt(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & buffer, size_t size, int value);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
