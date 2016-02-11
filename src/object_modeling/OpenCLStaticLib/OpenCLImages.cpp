#include "stdafx.h"
#include "OpenCLImages.h"

#include <boost/assign.hpp>

using std::cout;
using std::endl;

OpenCLImages::OpenCLImages(OpenCLAllKernels& opencl_kernels)
	: cl(opencl_kernels.getCL()),
	extract_y_float_kernel(opencl_kernels.getKernel("extractYFloat")),
	extract_cb_float_kernel(opencl_kernels.getKernel("extractCbFloat")),
	extract_cr_float_kernel(opencl_kernels.getKernel("extractCrFloat")),
	extract_ycrcb_float_kernel(opencl_kernels.getKernel("extractYCrCbFloat")),
	split_float_3_kernel(opencl_kernels.getKernel("splitFloat3")),
	merge_float_3_kernel(opencl_kernels.getKernel("mergeFloat3")),
	convolution_float_horizontal_kernel(opencl_kernels.getKernel("convolutionFloatHorizontal")),
	convolution_float_vertical_kernel(opencl_kernels.getKernel("convolutionFloatVertical")),
	half_size_image_kernel(opencl_kernels.getKernel("halfSizeImage")),
	half_size_float4_kernel(opencl_kernels.getKernel("halfSizeFloat4")),
	half_size_float4_mean_kernel(opencl_kernels.getKernel("halfSizeFloat4Mean"))
{
}

ImageBuffer OpenCLImages::extractYCrCbFloat(ImageBuffer const& input, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 3, CV_32F);
		
		//////////////////////////
		int kernel_arg = 0;
		extract_ycrcb_float_kernel.setArg(kernel_arg++, input.getBuffer());
		extract_ycrcb_float_kernel.setArg(kernel_arg++, result.getBuffer());
		extract_ycrcb_float_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(extract_ycrcb_float_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

ImageBuffer OpenCLImages::extractYFloat(ImageBuffer const& input, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 1, CV_32F);

		//////////////////////////
		int kernel_arg = 0;
		extract_y_float_kernel.setArg(kernel_arg++, input.getBuffer());
		extract_y_float_kernel.setArg(kernel_arg++, result.getBuffer());
		extract_y_float_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(extract_y_float_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

ImageBuffer OpenCLImages::extractCrFloat(ImageBuffer const& input, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 1, CV_32F);
		
		//////////////////////////
		int kernel_arg = 0;
		extract_cr_float_kernel.setArg(kernel_arg++, input.getBuffer());
		extract_cr_float_kernel.setArg(kernel_arg++, result.getBuffer());
		extract_cr_float_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(extract_cr_float_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

ImageBuffer OpenCLImages::extractCbFloat(ImageBuffer const& input, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 1, CV_32F);
		
		//////////////////////////
		int kernel_arg = 0;
		extract_cb_float_kernel.setArg(kernel_arg++, input.getBuffer());
		extract_cb_float_kernel.setArg(kernel_arg++, result.getBuffer());
		extract_cb_float_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(extract_cb_float_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

void OpenCLImages::splitFloat3(ImageBuffer const& input, ImageBuffer & output1, ImageBuffer & output2, ImageBuffer & output3, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		output1.resize(height, width, 1, CV_32F);
		output2.resize(height, width, 1, CV_32F);
		output3.resize(height, width, 1, CV_32F);
		
		//////////////////////////
		int kernel_arg = 0;
		split_float_3_kernel.setArg(kernel_arg++, input.getBuffer());
		split_float_3_kernel.setArg(kernel_arg++, output1.getBuffer());
		split_float_3_kernel.setArg(kernel_arg++, output2.getBuffer());
		split_float_3_kernel.setArg(kernel_arg++, output3.getBuffer());
		split_float_3_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(split_float_3_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

void OpenCLImages::mergeFloat3(ImageBuffer const& input1, ImageBuffer const& input2, ImageBuffer const& input3, ImageBuffer & output, int width, int height)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		output.resize(height, width, 3, CV_32F);
		
		//////////////////////////
		int kernel_arg = 0;
		merge_float_3_kernel.setArg(kernel_arg++, input1.getBuffer());
		merge_float_3_kernel.setArg(kernel_arg++, input2.getBuffer());
		merge_float_3_kernel.setArg(kernel_arg++, input3.getBuffer());
		merge_float_3_kernel.setArg(kernel_arg++, output.getBuffer());
		merge_float_3_kernel.setArg(kernel_arg++, image_dims);
		////////////////////////

		cl::NDRange global(image_dims.s[0], image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(merge_float_3_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

ImageBuffer OpenCLImages::convolutionFilterHorizontal(ImageBuffer const& input, int width, int height, int kernel_size, float* kernel_coeffs)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 1, CV_32F);

		cl::Buffer filter_constant (cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel_size * sizeof(float), kernel_coeffs);
		size_t local_mem_side_length = local_work_size_side + kernel_size - 1;
		size_t local_mem_size = local_mem_side_length * local_mem_side_length * sizeof(float);

		//////////////////////////
		int kernel_arg = 0;
		convolution_float_horizontal_kernel.setArg(kernel_arg++, input.getBuffer());
		convolution_float_horizontal_kernel.setArg(kernel_arg++, result.getBuffer());
		convolution_float_horizontal_kernel.setArg(kernel_arg++, image_dims);
		convolution_float_horizontal_kernel.setArg(kernel_arg++, filter_constant);
		convolution_float_horizontal_kernel.setArg(kernel_arg++, (cl_int)kernel_size);
		convolution_float_horizontal_kernel.setArg(kernel_arg++, local_mem_size, NULL);
		////////////////////////

		cl::NDRange global(local_work_size_side * getNumBlocks(image_dims.s[0], local_work_size_side), local_work_size_side * getNumBlocks(image_dims.s[1], local_work_size_side));
		cl::NDRange local(local_work_size_side, local_work_size_side);
		cl.queue.enqueueNDRangeKernel(convolution_float_horizontal_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}


ImageBuffer OpenCLImages::convolutionFilterVertical(ImageBuffer const& input, int width, int height, int kernel_size, float* kernel_coeffs)
{
	try {
		cl_int2 image_dims; // x, y
		image_dims.s[0] = width;
		image_dims.s[1] = height;
		ImageBuffer result(cl);
		result.resize(height, width, 1, CV_32F);

		cl::Buffer filter_constant (cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, kernel_size * sizeof(float), kernel_coeffs);
		size_t local_mem_side_length = local_work_size_side + kernel_size - 1;
		size_t local_mem_size = local_mem_side_length * local_mem_side_length * sizeof(float);

		//////////////////////////
		int kernel_arg = 0;
		convolution_float_vertical_kernel.setArg(kernel_arg++, input.getBuffer());
		convolution_float_vertical_kernel.setArg(kernel_arg++, result.getBuffer());
		convolution_float_vertical_kernel.setArg(kernel_arg++, image_dims);
		convolution_float_vertical_kernel.setArg(kernel_arg++, filter_constant);
		convolution_float_vertical_kernel.setArg(kernel_arg++, (cl_int)kernel_size);
		convolution_float_vertical_kernel.setArg(kernel_arg++, local_mem_size, NULL);
		////////////////////////

		cl::NDRange global(local_work_size_side * getNumBlocks(image_dims.s[0], local_work_size_side), local_work_size_side * getNumBlocks(image_dims.s[1], local_work_size_side));
		cl::NDRange local(local_work_size_side, local_work_size_side);
		cl.queue.enqueueNDRangeKernel(convolution_float_vertical_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

// todo: don't need args
ImageBuffer OpenCLImages::halfSizeImage(ImageBuffer const& input)
{
	try {
		cl_int2 input_dims; // x, y
		input_dims.s[0] = input.getCols();
		input_dims.s[1] = input.getRows();
		ImageBuffer result(cl);
		result.resize(input.getRows()/2, input.getCols()/2, input.getChannels(), input.getElementCvType());

		//////////////////////////
		int kernel_arg = 0;
		half_size_image_kernel.setArg(kernel_arg++, input.getBuffer());
		half_size_image_kernel.setArg(kernel_arg++, result.getBuffer());
		half_size_image_kernel.setArg(kernel_arg++, input_dims);
		////////////////////////

		cl::NDRange global(input.getCols() / 2, input.getRows() / 2);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(half_size_image_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		std::string error = oclErrorString(er.err());
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

// todo: don't need args
ImageBuffer OpenCLImages::halfSizeFloat4(ImageBuffer const& input)
{
	if (input.getChannels() != 4) {
		cout << "what are you doing?" << endl;
		throw std::runtime_error("stupid");
	}

	try {
		cl_int2 input_dims; // x, y
		input_dims.s[0] = input.getCols();
		input_dims.s[1] = input.getRows();
		ImageBuffer result(cl);
		result.resize(input.getRows()/2, input.getCols()/2, input.getChannels(), input.getElementCvType());

		//////////////////////////
		int kernel_arg = 0;
		half_size_float4_kernel.setArg(kernel_arg++, input.getBuffer());
		half_size_float4_kernel.setArg(kernel_arg++, result.getBuffer());
		half_size_float4_kernel.setArg(kernel_arg++, input_dims);
		////////////////////////

		cl::NDRange global(input.getCols() / 2, input.getRows() / 2);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(half_size_float4_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		std::string error = oclErrorString(er.err());
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}

// todo: don't need args
ImageBuffer OpenCLImages::halfSizeFloat4Mean(ImageBuffer const& input)
{
	if (input.getChannels() != 4) {
		cout << "what are you doing?" << endl;
		throw std::runtime_error("stupid");
	}

	try {
		cl_int2 input_dims; // x, y
		input_dims.s[0] = input.getCols();
		input_dims.s[1] = input.getRows();
		ImageBuffer result(cl);
		result.resize(input.getRows()/2, input.getCols()/2, input.getChannels(), input.getElementCvType());

		//////////////////////////
		int kernel_arg = 0;
		half_size_float4_mean_kernel.setArg(kernel_arg++, input.getBuffer());
		half_size_float4_mean_kernel.setArg(kernel_arg++, result.getBuffer());
		half_size_float4_mean_kernel.setArg(kernel_arg++, input_dims);
		////////////////////////

		cl::NDRange global(input.getCols() / 2, input.getRows() / 2);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(half_size_float4_mean_kernel, cl::NullRange, global, local);

		return result;
	}
	catch (cl::Error er) {
		std::string error = oclErrorString(er.err());
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		throw er;
	}
}