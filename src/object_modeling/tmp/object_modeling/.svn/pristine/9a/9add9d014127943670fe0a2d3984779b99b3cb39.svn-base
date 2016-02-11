#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelOptimizeNormalEquationTerms
{
public:
	static const std::string kernel_name;

	KernelOptimizeNormalEquationTerms(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		const ImageBuffer & weighted_error,
		const ImageBuffer & weighted_jacobian,
		BufferWrapper & LHS_terms,
		BufferWrapper & RHS_terms,
		const int total_count,
		const int offset,
		const float weight
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
