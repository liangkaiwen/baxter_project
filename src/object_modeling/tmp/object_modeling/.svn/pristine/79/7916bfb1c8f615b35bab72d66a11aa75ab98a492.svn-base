#include "KernelAddFloats.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelAddFloats::kernel_name = "KernelAddFloats";

KernelAddFloats::KernelAddFloats(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelAddFloats::runKernel(cl::Buffer & buffer, cl::Buffer & to_be_added, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer);
        kernel_.setArg(kernel_arg++, to_be_added);

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
