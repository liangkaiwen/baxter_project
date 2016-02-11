#include "KernelMinFloats.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelMinFloats::kernel_name = "KernelMinFloats";

KernelMinFloats::KernelMinFloats(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelMinFloats::runKernel(cl::Buffer & running_values, cl::Buffer & running_index, cl::Buffer & new_values, int new_index, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, running_values);
		kernel_.setArg(kernel_arg++, running_index);
		kernel_.setArg(kernel_arg++, new_values);
		kernel_.setArg(kernel_arg++, (cl_int)new_index);

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
