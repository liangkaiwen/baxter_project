#include "KernelPickIfIndexFloats.h"

#include <iostream>
using std::cout;
using std::endl;


const std::string KernelPickIfIndexFloats::kernel_name = "KernelPickIfIndexFloats";

KernelPickIfIndexFloats::KernelPickIfIndexFloats(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelPickIfIndexFloats::runKernel(cl::Buffer & running_buffer, cl::Buffer & possible_index, cl::Buffer & possible_values, int index, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, running_buffer);
		kernel_.setArg(kernel_arg++, possible_index);
		kernel_.setArg(kernel_arg++, possible_values);
		kernel_.setArg(kernel_arg++, (cl_int)index);

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
