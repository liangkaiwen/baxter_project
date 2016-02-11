#include "KernelApplyPoseToNormals.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelApplyPoseToNormals::kernel_name = "KernelApplyPoseToNormals";

KernelApplyPoseToNormals::KernelApplyPoseToNormals(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelApplyPoseToNormals::runKernel(cl::Buffer & buffer, Eigen::Affine3f const& pose, size_t size)
{
	try {
		// convert args
		cl_float16 cl_pose = getCLPose(pose);

		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, buffer);
        kernel_.setArg(kernel_arg++, cl_pose);

		// run kernel
		cl::NDRange global(size);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
