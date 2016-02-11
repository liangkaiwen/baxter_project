#include "KernelSetFloat.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelSetFloat::kernel_name = "KernelSetFloat";

KernelSetFloat::KernelSetFloat(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelSetFloat::runKernel(cl::Buffer & buffer, size_t size, float value)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer);
		kernel_.setArg(kernel_arg++, value);

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
