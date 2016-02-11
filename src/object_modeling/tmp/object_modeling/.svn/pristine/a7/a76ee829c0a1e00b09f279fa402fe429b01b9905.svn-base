#include "KernelPointsToDepthImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelPointsToDepthImage::kernel_name = "KernelPointsToDepthImage";

KernelPointsToDepthImage::KernelPointsToDepthImage(OpenCLAllKernels & opencl_kernels)
    : cl_(opencl_kernels.getCL()),
      kernel_(opencl_kernels.getKernel(kernel_name))
{
}



void KernelPointsToDepthImage::runKernel(
        ImageBuffer & points,
        ImageBuffer & result_depth_image
        )
{

    try {
        // converg args
        cl_int2 cl_image_dims = {points.getCols(), points.getRows()};

        // allocate result
        result_depth_image.resize(points.getRows(), points.getCols(), 1, CV_32F);

        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, points.getBuffer());
        kernel_.setArg(kernel_arg++, result_depth_image.getBuffer());
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
