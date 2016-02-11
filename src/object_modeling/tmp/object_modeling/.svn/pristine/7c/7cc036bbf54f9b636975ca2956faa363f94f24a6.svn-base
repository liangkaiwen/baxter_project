#include "KernelAddFrameToHistogram.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelAddFrameToHistogram::kernel_name = "KernelAddFrameToHistogram";

KernelAddFrameToHistogram::KernelAddFrameToHistogram(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelAddFrameToHistogram::runKernel(
	VolumeBuffer & buffer_histogram_bin,
	ImageBuffer const& buffer_depth_image,
	ImageBuffer const& buffer_segments,
	int which_segment,
	float bin_min,
	float bin_max,
	float voxel_size,
	const Eigen::Affine3f& pose,
	const Eigen::Array2f& camera_focal,
    const Eigen::Array2f& camera_center,
        const float min_truncation_distance
	)
{
	try {
		// convert args
		cl_float16 cl_pose = getCLPose(pose);

		cl_int4 cl_volume_dims = {buffer_histogram_bin.getVolumeCellCounts()[0], buffer_histogram_bin.getVolumeCellCounts()[1], buffer_histogram_bin.getVolumeCellCounts()[2], 0};
		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
		cl_int2 cl_camera_size = {buffer_depth_image.getCols(), buffer_depth_image.getRows()};

		// not a problem on nvidia, but CPU complains
		cl::Buffer local_segment_buffer;
		if (which_segment == 0) {
			local_segment_buffer = cl::Buffer(cl_.context, 0, 1);
		}
		else {
			local_segment_buffer = buffer_segments.getBuffer();
		}


		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer_histogram_bin.getBuffer());
		kernel_.setArg(kernel_arg++, buffer_depth_image.getBuffer());
		kernel_.setArg(kernel_arg++, local_segment_buffer);
		kernel_.setArg(kernel_arg++, which_segment);
		kernel_.setArg(kernel_arg++, bin_min);
		kernel_.setArg(kernel_arg++, bin_max);
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, voxel_size);
		kernel_.setArg(kernel_arg++, cl_pose);
		kernel_.setArg(kernel_arg++, cl_camera_focal);
		kernel_.setArg(kernel_arg++, cl_camera_center);
		kernel_.setArg(kernel_arg++, cl_camera_size);
        kernel_.setArg(kernel_arg++, min_truncation_distance);


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
