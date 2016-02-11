#include "KernelExtractIntForPointImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelExtractIntForPointImage::kernel_name = "KernelExtractIntForPointImage";

KernelExtractIntForPointImage::KernelExtractIntForPointImage(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelExtractIntForPointImage::runKernel(
	VolumeBuffer & volume,
	ImageBuffer & points_image,
	ImageBuffer & result_image,
	Eigen::Affine3f const& model_pose,
	float voxel_size
	)
{
	try {
		cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};
		cl_int2 cl_image_dims = {points_image.getCols(), points_image.getRows()};
		cl_float16 cl_model_pose = getCLPose(model_pose);
		cl_float16 cl_model_pose_inverse = getCLPose(model_pose.inverse());

		result_image.resize(points_image.getRows(), points_image.getCols(), 1, CV_32S);
		// could fill with something?

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, volume.getBuffer());
		kernel_.setArg(kernel_arg++, points_image.getBuffer());
		kernel_.setArg(kernel_arg++, result_image.getBuffer());
		kernel_.setArg(kernel_arg++, cl_model_pose);
		kernel_.setArg(kernel_arg++, cl_model_pose_inverse);
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, voxel_size);
		kernel_.setArg(kernel_arg++, cl_image_dims);


		// run kernel
		cl::NDRange global(cl_image_dims.s[0], cl_image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
