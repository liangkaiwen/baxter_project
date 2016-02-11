#include "KernelRaytraceBox.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelRaytraceBox::kernel_name = "KernelRaytraceBox";

KernelRaytraceBox::KernelRaytraceBox(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelRaytraceBox::runKernel(
	ImageBuffer & rendered_mask,
	ImageBuffer & rendered_points,
	ImageBuffer & rendered_normals,
	const Eigen::Affine3f & model_pose,
	const Eigen::Array3f& box_corner_from_origin,
	const Eigen::Affine3f& box_pose,
	const Eigen::Array2f & camera_focal,
	const Eigen::Array2f & camera_center,
	float min_render_depth,
	float max_render_depth,
	int mask_value
	)
{
	try {
		cl_float16 cl_model_pose = getCLPose(model_pose);
		cl_float16 cl_model_pose_inverse = getCLPose(model_pose.inverse());

		cl_float16 cl_box_pose = getCLPose(box_pose);
		cl_float16 cl_box_pose_inverse = getCLPose(box_pose.inverse());

		cl_float4 cl_box_corner_from_origin = {box_corner_from_origin[0], box_corner_from_origin[1], box_corner_from_origin[2], 1}; // note 1!!  matches internal points...

		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
		cl_int2 cl_camera_size = {rendered_mask.getCols(), rendered_mask.getRows()};

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, rendered_mask.getBuffer());
		kernel_.setArg(kernel_arg++, rendered_points.getBuffer());
		kernel_.setArg(kernel_arg++, rendered_normals.getBuffer());
		kernel_.setArg(kernel_arg++, cl_model_pose);
		kernel_.setArg(kernel_arg++, cl_model_pose_inverse);
		kernel_.setArg(kernel_arg++, cl_box_pose);
		kernel_.setArg(kernel_arg++, cl_box_pose_inverse);
		kernel_.setArg(kernel_arg++, cl_box_corner_from_origin);
		kernel_.setArg(kernel_arg++, cl_camera_focal);
		kernel_.setArg(kernel_arg++, cl_camera_center);
		kernel_.setArg(kernel_arg++, cl_camera_size);
		kernel_.setArg(kernel_arg++, min_render_depth);
		kernel_.setArg(kernel_arg++, max_render_depth);
		kernel_.setArg(kernel_arg++, (cl_int)mask_value);


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
