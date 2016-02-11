#include "KernelGaussianPDF.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelGaussianPDF::kernel_name = "KernelGaussianPDF";

KernelGaussianPDF::KernelGaussianPDF(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelGaussianPDF::runKernel(cl::Buffer & means, cl::Buffer & variances, cl::Buffer & x_values, cl::Buffer & pdf_values, size_t size)
{
	try {
		// convert args

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, means);
        kernel_.setArg(kernel_arg++, variances);
        kernel_.setArg(kernel_arg++, x_values);
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
