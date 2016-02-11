#include "KernelExtractFloat4ForPointImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelExtractFloat4ForPointImage::kernel_name = "KernelExtractFloat4ForPointImage";

KernelExtractFloat4ForPointImage::KernelExtractFloat4ForPointImage(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

#if 0
__kernel void KernelExtractFloat4ForPointImage(
	__global float4 *volume,
	__global float4 *points_image,
	__global float4 *result_image,
	const float16 model_pose,
	const float16 model_pose_inverse,
	const int4 volume_dims,
	const float voxel_size,
	const int2 image_dims
	)
#endif

void KernelExtractFloat4ForPointImage::runKernel(
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

		result_image.resize(points_image.getRows(), points_image.getCols(), 4, CV_32F);
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
