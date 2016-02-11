#include "KernelTransformPoints.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelTransformPoints::kernel_name = "KernelTransformPoints";

KernelTransformPoints::KernelTransformPoints(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelTransformPoints::runKernel(
	const ImageBuffer & input_points,
	ImageBuffer & output_points,
	const Eigen::Affine3f & pose
	)
{
	try {
		output_points.resize(input_points.getRows(), input_points.getCols(), 4, CV_32F);

		cl_float16 cl_pose = getCLPose(pose);
		cl_int2 cl_camera_size = {input_points.getCols(), input_points.getRows()};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, input_points.getBuffer());
		kernel_.setArg(kernel_arg++, output_points.getBuffer());
		kernel_.setArg(kernel_arg++, cl_camera_size);
		kernel_.setArg(kernel_arg++, cl_pose);

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
