#include "KernelGaussianPDFConstantX.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelGaussianPDFConstantX::kernel_name = "KernelGaussianPDFConstantX";

KernelGaussianPDFConstantX::KernelGaussianPDFConstantX(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelGaussianPDFConstantX::runKernel(cl::Buffer & means, cl::Buffer & variances, float x_value, cl::Buffer & pdf_values, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, means);
        kernel_.setArg(kernel_arg++, variances);
        kernel_.setArg(kernel_arg++, x_value);
        kernel_.setArg(kernel_arg++, pdf_values);

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
