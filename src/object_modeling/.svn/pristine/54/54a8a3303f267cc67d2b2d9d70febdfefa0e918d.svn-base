#include "KernelSetVolumeSDFBox.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelSetVolumeSDFBox::kernel_name = "KernelSetVolumeSDFBox";

KernelSetVolumeSDFBox::KernelSetVolumeSDFBox(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}




void KernelSetVolumeSDFBox::runKernel(
	VolumeBuffer & volume,
	const float voxel_size,
	const Eigen::Affine3f& volume_pose,
	const Eigen::Array3f& box_corner_from_origin,
	const Eigen::Affine3f& box_pose,
	const Eigen::Array2i& camera_size,
	const Eigen::Array2f& camera_focal,
	const Eigen::Array2f& camera_center
	)
{
	try {
		// convert args
		cl_float16 cl_volume_pose = getCLPose(volume_pose);

		cl_float4 cl_box_corner_from_origin = {box_corner_from_origin[0], box_corner_from_origin[1], box_corner_from_origin[2], 1}; // note 1!!  matches internal points...
		cl_float16 cl_box_pose = getCLPose(box_pose);
		cl_float16 cl_box_pose_inverse = getCLPose(box_pose.inverse());

		cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};

		cl_int2 cl_camera_size = {camera_size[0], camera_size[1]};
		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};


		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, volume.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, voxel_size);
		kernel_.setArg(kernel_arg++, cl_volume_pose);
		kernel_.setArg(kernel_arg++, cl_camera_size);
		kernel_.setArg(kernel_arg++, cl_camera_focal);
		kernel_.setArg(kernel_arg++, cl_camera_center);
		kernel_.setArg(kernel_arg++, cl_box_corner_from_origin);
		kernel_.setArg(kernel_arg++, cl_box_pose);
		kernel_.setArg(kernel_arg++, cl_box_pose_inverse);


		// run kernel
		cl::NDRange global(cl_volume_dims.s[0], cl_volume_dims.s[1], cl_volume_dims.s[2]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
