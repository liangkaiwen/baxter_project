#include "KernelMarkPointsViolateEmpty.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelMarkPointsViolateEmpty::kernel_name = "KernelMarkPointsViolateEmpty";

KernelMarkPointsViolateEmpty::KernelMarkPointsViolateEmpty(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}



void KernelMarkPointsViolateEmpty::runKernel(
	VolumeBuffer & volume,
	VolumeBuffer & weights,
	cl::Buffer & points,
	cl::Buffer & result_violates_empty,
	size_t points_size,
	const float voxel_size,
    const Eigen::Affine3f& volume_pose,
        float min_value_invalid
	)
{
	try {
		// convert args
		cl_float16 cl_volume_pose = getCLPose(volume_pose);
		cl_float16 cl_volume_pose_inverse = getCLPose(volume_pose.inverse());
		cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, volume.getBuffer());
		kernel_.setArg(kernel_arg++, weights.getBuffer());
		kernel_.setArg(kernel_arg++, points);
		kernel_.setArg(kernel_arg++, result_violates_empty);
		kernel_.setArg(kernel_arg++, voxel_size);
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, cl_volume_pose);
		kernel_.setArg(kernel_arg++, cl_volume_pose_inverse);
        kernel_.setArg(kernel_arg++, min_value_invalid);


		// run kernel
		cl::NDRange global(points_size);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
