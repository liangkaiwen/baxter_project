#include "KernelMinAbsFloatsWithWeightsRecordIndex.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelMinAbsFloatsWithWeightsRecordIndex::kernel_name = "KernelMinAbsFloatsWithWeightsRecordIndex";

KernelMinAbsFloatsWithWeightsRecordIndex::KernelMinAbsFloatsWithWeightsRecordIndex(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelMinAbsFloatsWithWeightsRecordIndex::runKernel(
		cl::Buffer & running_values,
		cl::Buffer & running_weights,
		cl::Buffer & runnin_index,
		cl::Buffer & to_min_values,
		cl::Buffer & to_min_weights,
		int this_index,
		size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, running_values);
        kernel_.setArg(kernel_arg++, running_weights);
        kernel_.setArg(kernel_arg++, runnin_index);
        kernel_.setArg(kernel_arg++, to_min_values);
        kernel_.setArg(kernel_arg++, to_min_weights);
        kernel_.setArg(kernel_arg++, (cl_int)this_index);

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
