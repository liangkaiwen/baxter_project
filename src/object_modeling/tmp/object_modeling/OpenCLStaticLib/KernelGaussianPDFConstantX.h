#pragma once

#include "OpenCLAllKernels.h"

class KernelGaussianPDFConstantX
{
public:
	static const std::string kernel_name;

    KernelGaussianPDFConstantX(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & means, cl::Buffer & variances, float x_value, cl::Buffer & pdf_values, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
