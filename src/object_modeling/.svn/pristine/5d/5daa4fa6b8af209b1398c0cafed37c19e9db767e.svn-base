#pragma once

#include "OpenCLAllKernels.h"

class KernelGaussianPDF
{
public:
	static const std::string kernel_name;

    KernelGaussianPDF(OpenCLAllKernels & opencl_kernels);

    void runKernel(cl::Buffer & means, cl::Buffer & variances, cl::Buffer & x_values, cl::Buffer & pdf_values, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
