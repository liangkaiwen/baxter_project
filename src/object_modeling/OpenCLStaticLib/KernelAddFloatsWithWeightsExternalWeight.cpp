#include "KernelAddFloatsWithWeightsExternalWeight.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelAddFloatsWithWeightsExternalWeight::kernel_name = "KernelAddFloatsWithWeightsExternalWeight";

KernelAddFloatsWithWeightsExternalWeight::KernelAddFloatsWithWeightsExternalWeight(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelAddFloatsWithWeightsExternalWeight::runKernel(cl::Buffer & running_values, cl::Buffer & running_weights, cl::Buffer & to_add_values, cl::Buffer & to_add_weights, float external_weight, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, running_values);
        kernel_.setArg(kernel_arg++, running_weights);
        kernel_.setArg(kernel_arg++, to_add_values);
        kernel_.setArg(kernel_arg++, to_add_weights);
        kernel_.setArg(kernel_arg++, external_weight);

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
