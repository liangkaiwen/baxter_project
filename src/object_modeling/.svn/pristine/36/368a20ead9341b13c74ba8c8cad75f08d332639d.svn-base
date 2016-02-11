#include "KernelSetFloat4.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelSetFloat4::kernel_name = "KernelSetFloat4";

KernelSetFloat4::KernelSetFloat4(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelSetFloat4::runKernel(cl::Buffer & buffer, size_t size, Eigen::Array4f const& value)
{
	try {
		// convert args
        cl_float4 cl_value = {value[0], value[1], value[2], value[3]};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer);
        kernel_.setArg(kernel_arg++, cl_value);

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
