#include "KernelDivideFloats.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelDivideFloats::kernel_name = "KernelDivideFloats";

KernelDivideFloats::KernelDivideFloats(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelDivideFloats::runKernel(cl::Buffer & buffer, cl::Buffer & divisors, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer);
		kernel_.setArg(kernel_arg++, divisors);

		// run kernel
		cl::NDRange global(size);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
