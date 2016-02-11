#include "stdafx.h"
#include "OpenCLNormals.h"

#include <boost/assign.hpp>

using std::cout;
using std::endl;

OpenCLNormals::OpenCLNormals(boost::shared_ptr<OpenCLAllKernels> all_kernels)
	: all_kernels_(all_kernels),
	compute_normals_kernel(all_kernels->getKernel("computeNormals")),
	smooth_normals_kernel(all_kernels->getKernel("smoothNormals")),
	image_buffer_normals_temp_1(all_kernels->getCL()),
	image_buffer_normals_temp_2(all_kernels->getCL())
{
}


void OpenCLNormals::computeNormalsWithBuffers(
	const ImageBuffer & image_buffer_points,
	float max_sigmas,
	int smooth_iterations,
	ImageBuffer & image_buffer_normals)
{
	try {
		image_buffer_normals_temp_1.resize(image_buffer_points.getRows(), image_buffer_points.getCols(), 4, CV_32F);
		image_buffer_normals_temp_2.resize(image_buffer_points.getRows(), image_buffer_points.getCols(), 4, CV_32F);
		image_buffer_normals.resize(image_buffer_points.getRows(), image_buffer_points.getCols(), 4, CV_32F);

		cl_int2 points_dims = {image_buffer_points.getCols(), image_buffer_points.getRows()};

		size_t local_mem_size = 4 * sizeof(float) * (local_work_size_side + 2) * (local_work_size_side + 2);

		//////////////////////////
		int kernel_arg = 0;
		// buffers first
		compute_normals_kernel.setArg(kernel_arg++, image_buffer_points.getBuffer());
		compute_normals_kernel.setArg(kernel_arg++, image_buffer_normals.getBuffer());
		compute_normals_kernel.setArg(kernel_arg++, local_mem_size, NULL);
		compute_normals_kernel.setArg(kernel_arg++, points_dims);
		compute_normals_kernel.setArg(kernel_arg++, max_sigmas);
		////////////////////////

		// set sizes and run
		//cl::NDRange global(points_dims.s[0], points_dims.s[1]);
		cl::NDRange global(local_work_size_side * getNumBlocks(points_dims.s[0], local_work_size_side), local_work_size_side * getNumBlocks(points_dims.s[1], local_work_size_side));
		cl::NDRange local(local_work_size_side, local_work_size_side);
		all_kernels_->getCL().queue.enqueueNDRangeKernel(compute_normals_kernel, cl::NullRange, global, local);

		//////////////////////
		// smoothing
		all_kernels_->getCL().queue.enqueueCopyBuffer(image_buffer_normals.getBuffer(), image_buffer_normals_temp_1.getBuffer(), 0, 0, image_buffer_normals.getSizeBytes());

		cl::Buffer buffer_temp_1 = image_buffer_normals_temp_1.getBuffer();
		cl::Buffer buffer_temp_2 = image_buffer_normals_temp_2.getBuffer();

		cl::Buffer* smooth_input_buffer = &buffer_temp_1;
		cl::Buffer* smooth_output_buffer = &buffer_temp_2;
		for (int smooth = 0; smooth < smooth_iterations; ++smooth) {
			//////////////////////////
			int kernel_arg = 0;
			// buffers first
			smooth_normals_kernel.setArg(kernel_arg++, image_buffer_points.getBuffer());
			smooth_normals_kernel.setArg(kernel_arg++, *smooth_input_buffer);
			smooth_normals_kernel.setArg(kernel_arg++, *smooth_output_buffer);
			smooth_normals_kernel.setArg(kernel_arg++, local_mem_size, NULL);
			smooth_normals_kernel.setArg(kernel_arg++, local_mem_size, NULL);
			smooth_normals_kernel.setArg(kernel_arg++, points_dims);
			smooth_normals_kernel.setArg(kernel_arg++, max_sigmas);
			////////////////////////

			all_kernels_->getCL().queue.enqueueNDRangeKernel(smooth_normals_kernel, cl::NullRange, global, local);

			// swap
			cl::Buffer* swap_ptr = smooth_output_buffer;
			smooth_output_buffer = smooth_input_buffer;
			smooth_input_buffer = swap_ptr;
		}

		// regardless of any smoothing, smooth input buffer contains the result we want
		// assumes image_buffer_normals is already sized correctly
		all_kernels_->getCL().queue.enqueueCopyBuffer(*smooth_input_buffer, image_buffer_normals.getBuffer(), 0, 0, image_buffer_normals.getSizeBytes());
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}


