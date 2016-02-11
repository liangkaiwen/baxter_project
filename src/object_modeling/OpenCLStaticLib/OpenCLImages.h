#pragma once

#include "cll.h"
#include "OpenCLAllKernels.h"
#include "ImageBuffer.h"

class OpenCLImages
{
private:
	OpenCLImages(const OpenCLImages& other);
	void operator=(const OpenCLImages& other);

public:
	static const int local_work_size_side = 16;// public for debugging?

	OpenCLImages(OpenCLAllKernels& opencl_kernels);

	CL& getCL() {return cl;} // this is hackish

	ImageBuffer extractYCrCbFloat(ImageBuffer const& input, int width, int height);
	ImageBuffer extractYFloat(ImageBuffer const& input, int width, int height);
	ImageBuffer extractCrFloat(ImageBuffer const& input, int width, int height);
	ImageBuffer extractCbFloat(ImageBuffer const& input, int width, int height);

	void splitFloat3(ImageBuffer const& input, ImageBuffer & output1, ImageBuffer & output2, ImageBuffer & output3, int width, int height);
	void mergeFloat3(ImageBuffer const& input1, ImageBuffer const& input2, ImageBuffer const& input3, ImageBuffer & output, int width, int height);

	ImageBuffer convolutionFilterHorizontal(ImageBuffer const& input, int width, int height, int kernel_size, float* kernel_coeffs);
	ImageBuffer convolutionFilterVertical(ImageBuffer const& input, int width, int height, int kernel_size, float* kernel_coeffs);

	ImageBuffer halfSizeImage(ImageBuffer const& input);
	ImageBuffer halfSizeFloat4(ImageBuffer const& input);
	ImageBuffer halfSizeFloat4Mean(ImageBuffer const& input);

protected:
	//////////
	// members
	CL& cl;

	cl::Kernel extract_y_float_kernel;
	cl::Kernel extract_cr_float_kernel;
	cl::Kernel extract_cb_float_kernel;
	cl::Kernel extract_ycrcb_float_kernel; // old interlaced style

	cl::Kernel split_float_3_kernel;
	cl::Kernel merge_float_3_kernel;

	cl::Kernel convolution_float_horizontal_kernel;
	cl::Kernel convolution_float_vertical_kernel;

	cl::Kernel half_size_image_kernel;
	cl::Kernel half_size_float4_kernel;
	cl::Kernel half_size_float4_mean_kernel;
};

