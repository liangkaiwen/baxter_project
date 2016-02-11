#include "KernelNormalsToShadedImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelNormalsToShadedImage::kernel_name = "KernelNormalsToShadedImage";

KernelNormalsToShadedImage::KernelNormalsToShadedImage(OpenCLAllKernels & opencl_kernels)
    : cl_(opencl_kernels.getCL()),
      kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelNormalsToShadedImage::runKernel(
        ImageBuffer & normals,
        ImageBuffer & result_image,
        Eigen::Vector3f const& vector_to_light
        )
{

    try {
        // converg args
        cl_float4 cl_vector_to_light = {vector_to_light[0], vector_to_light[1], vector_to_light[2], 0};
        cl_int2 cl_image_dims = {normals.getCols(), normals.getRows()};

        // allocate result
        result_image.resize(normals.getRows(), normals.getCols(), 4, CV_8U);

        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, normals.getBuffer());
        kernel_.setArg(kernel_arg++, result_image.getBuffer());
        kernel_.setArg(kernel_arg++, cl_image_dims);
        kernel_.setArg(kernel_arg++, cl_vector_to_light);


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
