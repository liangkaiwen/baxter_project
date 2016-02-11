#include "KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction::kernel_name = "KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction";

KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction::KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction::runKernel(
	cl::Buffer & running_values,
	cl::Buffer & running_weights,
	cl::Buffer & to_min_values,
	cl::Buffer & to_min_weights,
	cl::Buffer & max_weights,
	float min_weight_fraction,
	size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, running_values);
        kernel_.setArg(kernel_arg++, running_weights);
        kernel_.setArg(kernel_arg++, to_min_values);
        kernel_.setArg(kernel_arg++, to_min_weights);
        kernel_.setArg(kernel_arg++, max_weights);
        kernel_.setArg(kernel_arg++, min_weight_fraction);

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
