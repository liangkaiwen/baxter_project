#include "KernelVignetteApplyModelPolynomial3Uchar4.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelVignetteApplyModelPolynomial3Uchar4::kernel_name = "KernelVignetteApplyModelPolynomial3Uchar4";

KernelVignetteApplyModelPolynomial3Uchar4::KernelVignetteApplyModelPolynomial3Uchar4(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

#if 0
KernelVignetteApplyModelPolynomial3Uchar4(
    __global uchar4 * input_image,
    __global uchar4 * output_image,
    const int2 image_dims,
    const float2 camera_center,
    const float4 vignette_model)
#endif


void KernelVignetteApplyModelPolynomial3Uchar4::runKernel(
    const ImageBuffer & input_image,
    ImageBuffer & output_image,
    const Eigen::Array2f & camera_center,
    const Eigen::Array3f & vignette_model
	)
{
	try {
        cl_int2 cl_image_dims = {input_image.getCols(), input_image.getRows()};
        cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
        cl_float4 cl_vignette_model = {vignette_model[0], vignette_model[1], vignette_model[2], 0};

        output_image.resize(input_image.getRows(), input_image.getCols(), 4, CV_8U);

		// assign args
		int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, input_image.getBuffer());
        kernel_.setArg(kernel_arg++, output_image.getBuffer());
        kernel_.setArg(kernel_arg++, cl_image_dims);
        kernel_.setArg(kernel_arg++, cl_camera_center);
        kernel_.setArg(kernel_arg++, cl_vignette_model);


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
