#include "KernelDepthImageToPoints.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelDepthImageToPoints::kernel_name = "KernelDepthImageToPoints";

KernelDepthImageToPoints::KernelDepthImageToPoints(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}



void KernelDepthImageToPoints::runKernel(
	const ImageBuffer & depth_image,
	ImageBuffer & points_image,
	Eigen::Array2f const& camera_focal,
	Eigen::Array2f const& camera_center
	)
{
	try {
		points_image.resize(depth_image.getRows(), depth_image.getCols(), 4, CV_32F);

		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
		cl_int2 cl_camera_size = {depth_image.getCols(), depth_image.getRows()};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, depth_image.getBuffer());
		kernel_.setArg(kernel_arg++, points_image.getBuffer());
		kernel_.setArg(kernel_arg++, cl_camera_size);
		kernel_.setArg(kernel_arg++, cl_camera_focal);
		kernel_.setArg(kernel_arg++, cl_camera_center);

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
