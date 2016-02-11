#pragma once

#include "OpenCLAllKernels.h"

class KernelSetFloat4
{
public:
	static const std::string kernel_name;

    KernelSetFloat4(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & buffer, size_t size, Eigen::Array4f const& value);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
