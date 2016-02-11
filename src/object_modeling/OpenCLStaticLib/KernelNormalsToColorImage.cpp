#include "KernelNormalsToColorImage.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelNormalsToColorImage::kernel_name = "KernelNormalsToColorImage";

KernelNormalsToColorImage::KernelNormalsToColorImage(OpenCLAllKernels & opencl_kernels)
    : cl_(opencl_kernels.getCL()),
      kernel_(opencl_kernels.getKernel(kernel_name))
{
}


void KernelNormalsToColorImage::runKernel(
        ImageBuffer & normals,
        ImageBuffer & result_image
        )
{

    try {
        // converg args
        cl_int2 cl_image_dims = {normals.getCols(), normals.getRows()};

        // allocate result
        result_image.resize(normals.getRows(), normals.getCols(), 4, CV_8U);

        // assign args
        int kernel_arg = 0;
        kernel_.setArg(kernel_arg++, normals.getBuffer());
        kernel_.setArg(kernel_arg++, result_image.getBuffer());
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
