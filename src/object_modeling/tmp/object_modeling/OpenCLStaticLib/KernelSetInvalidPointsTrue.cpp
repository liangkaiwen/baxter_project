#include "KernelSetInvalidPointsTrue.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelSetInvalidPointsTrue::kernel_name = "KernelSetInvalidPointsTrue";

KernelSetInvalidPointsTrue::KernelSetInvalidPointsTrue(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelSetInvalidPointsTrue::runKernel(
	const ImageBuffer & depth_image,
	ImageBuffer & bool_image,
	bool resize_bool_image
	)
{
	try {
		// currently not done because we want to size and zero the image before...
		if (resize_bool_image) {
			bool_image.resize(depth_image.getRows(), depth_image.getCols(), 1, CV_8U);
		}

		cl_int2 cl_camera_size = {depth_image.getCols(), depth_image.getRows()};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, depth_image.getBuffer());
		kernel_.setArg(kernel_arg++, bool_image.getBuffer());
		kernel_.setArg(kernel_arg++, cl_camera_size);

		// run kernel
		cl::NDRange global(cl_camera_size.s[0], cl_camera_size.s[1]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
