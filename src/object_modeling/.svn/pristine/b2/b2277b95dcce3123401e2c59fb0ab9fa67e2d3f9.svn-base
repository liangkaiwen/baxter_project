#include "KernelOptimizeNormalEquationTerms.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelOptimizeNormalEquationTerms::kernel_name = "KernelOptimizeNormalEquationTerms";

KernelOptimizeNormalEquationTerms::KernelOptimizeNormalEquationTerms(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelOptimizeNormalEquationTerms::runKernel(
	const ImageBuffer & weighted_error,
	const ImageBuffer & weighted_jacobian,
	BufferWrapper & LHS_terms,
	BufferWrapper & RHS_terms,
	const int total_count,
	const int offset,
	const float weight
	)
{
	try {
		const int cols = weighted_error.getCols();
		const int rows = weighted_error.getRows();
		const int point_count = rows * cols;

		// no resize here

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, weighted_error.getBuffer());
		kernel_.setArg(kernel_arg++, weighted_jacobian.getBuffer());
		kernel_.setArg(kernel_arg++, LHS_terms.getBuffer());
		kernel_.setArg(kernel_arg++, RHS_terms.getBuffer());
		kernel_.setArg(kernel_arg++, total_count);
		kernel_.setArg(kernel_arg++, offset);
		kernel_.setArg(kernel_arg++, weight);

		// run kernel
		cl::NDRange global(point_count);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
