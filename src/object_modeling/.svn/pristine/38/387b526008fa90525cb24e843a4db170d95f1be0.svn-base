#include "KernelExtractVolumeSlice.h"

#include <iostream>
using std::cout;
using std::endl;

const std::string KernelExtractVolumeSlice::kernel_name = "KernelExtractVolumeSlice";

KernelExtractVolumeSlice::KernelExtractVolumeSlice(OpenCLAllKernels & opencl_kernels)
	: cl_(opencl_kernels.getCL()),
	kernel_(opencl_kernels.getKernel(kernel_name))
{
}

void KernelExtractVolumeSlice::runKernel(
	VolumeBuffer & volume,
	int axis, int position,
	ImageBuffer & result
	)
{
	try {
		cl_int4 cl_volume_dims = {volume.getVolumeCellCounts()[0], volume.getVolumeCellCounts()[1], volume.getVolumeCellCounts()[2], 0};

		// ensure correct size of results:
		Eigen::Array2i image_size;
		if (axis == 0) {
			image_size[0] = cl_volume_dims.s[1];
			image_size[1] = cl_volume_dims.s[2];
		}
		else if (axis == 1) {
			image_size[0] = cl_volume_dims.s[0];
			image_size[1] = cl_volume_dims.s[2];
		}
		else if (axis == 2) {
			image_size[0] = cl_volume_dims.s[0];
			image_size[1] = cl_volume_dims.s[1];
		}
		else {
			cout << "idiot" << endl;
			exit(1);
		}
		result.resize(image_size[1], image_size[0], 1, CV_32F);


		// assign args
		int kernel_arg = 0;
		kernel_.setArg(kernel_arg++, volume.getBuffer());
		kernel_.setArg(kernel_arg++, result.getBuffer());
		kernel_.setArg(kernel_arg++, cl_volume_dims);
		kernel_.setArg(kernel_arg++, axis);
		kernel_.setArg(kernel_arg++, position);


		// run kernel
		cl::NDRange global(image_size[0], image_size[1]);
		cl::NDRange local = cl::NullRange;
		cl_.queue.enqueueNDRangeKernel(kernel_, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		cout << kernel_name << endl;
		cout << "cl::Error: " << oclErrorString(er.err()) << endl;
		throw er;
	}
}
