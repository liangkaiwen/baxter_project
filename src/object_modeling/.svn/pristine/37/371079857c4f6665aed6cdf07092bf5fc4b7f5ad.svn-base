#include "stdafx.h"
#include "VolumeBuffer.h"

#include "util.h"

#include "KernelSetFloat.h"
#include "KernelSetUChar.h"
#include "KernelSetInt.h"
#include "KernelSetFloat4.h"

using std::cout;
using std::endl;


VolumeBuffer::VolumeBuffer(boost::shared_ptr<OpenCLAllKernels> all_kernels)
	: all_kernels_(all_kernels),
	buffer_wrapper_(all_kernels->getCL()),
	volume_cell_counts_(Eigen::Array3i::Zero()),
	element_byte_size_(0)
{
}

VolumeBuffer::VolumeBuffer(boost::shared_ptr<OpenCLAllKernels> all_kernels, const Eigen::Array3i & cell_counts, size_t element_byte_size)
	: all_kernels_(all_kernels),
	buffer_wrapper_(all_kernels->getCL()),
	volume_cell_counts_(cell_counts),
	element_byte_size_(element_byte_size)
{
	buffer_wrapper_.reallocate(getSizeInBytes());
}

VolumeBuffer::VolumeBuffer(VolumeBuffer const& other)
	: all_kernels_(other.all_kernels_),
	buffer_wrapper_(other.buffer_wrapper_),
	volume_cell_counts_(other.volume_cell_counts_),
	element_byte_size_(other.element_byte_size_)
{
}

VolumeBuffer& VolumeBuffer::operator=(const VolumeBuffer& other)
{
	// all kernels stays

	buffer_wrapper_ = other.buffer_wrapper_;
	volume_cell_counts_ = other.volume_cell_counts_;
	element_byte_size_ = other.element_byte_size_;

	return *this;
}

void VolumeBuffer::deallocate()
{
	buffer_wrapper_.reallocate(0);
	volume_cell_counts_ = Eigen::Array3i::Zero();
	element_byte_size_ = 0;
}

size_t VolumeBuffer::getSizeInBytes() const
{ 
    return getSizeInCells() * element_byte_size_;
}

size_t VolumeBuffer::getSizeInCells() const
{
	return volume_cell_counts_[0] * volume_cell_counts_[1] * volume_cell_counts_[2];
}

void VolumeBuffer::resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size)
{
	volume_cell_counts_ = volume_cell_counts;
	element_byte_size_ = element_byte_size;
    buffer_wrapper_.reallocate(getSizeInBytes());
}

void VolumeBuffer::setFloat(float value)
{
	KernelSetFloat _KernelSetFloat(*all_kernels_);
	_KernelSetFloat.runKernel(buffer_wrapper_.getBuffer(), getSizeInBytes() / sizeof(float), value);
}

void VolumeBuffer::setUChar(unsigned char value)
{
	KernelSetUChar _KernelSetUChar(*all_kernels_);
	_KernelSetUChar.runKernel(buffer_wrapper_.getBuffer(), getSizeInBytes() / sizeof(unsigned char), value);
}

void VolumeBuffer::setInt(int value)
{
	KernelSetInt _KernelSetInt(*all_kernels_);
	_KernelSetInt.runKernel(buffer_wrapper_.getBuffer(), getSizeInBytes() / sizeof(int), value);
}

void VolumeBuffer::setFloat4(const Eigen::Array4f &value)
{
	KernelSetFloat4 _KernelSetFloat4(*all_kernels_);
	_KernelSetFloat4.runKernel(buffer_wrapper_.getBuffer(), getSizeInBytes() / (4 * sizeof(float)), value);
}

void VolumeBuffer::getNonzeroPointsAndFloatValues(const Eigen::Affine3f& pose, float cell_size, float epsilon, std::vector<std::pair<Eigen::Vector3f, float> >& result)
{
    std::vector<float> values(getSizeInCells());
	buffer_wrapper_.readToFloatVector(values);

	result.clear();
	for (int v_x = 0; v_x < volume_cell_counts_[0]; ++v_x) {
		for (int v_y = 0; v_y < volume_cell_counts_[1]; ++v_y) {
			for (int v_z = 0; v_z < volume_cell_counts_[2]; ++v_z) {
				Eigen::Array3i v(v_x, v_y, v_z);
				size_t buffer_index = getVolumeIndex(volume_cell_counts_, v);
				if (fabs(values[buffer_index]) > epsilon) {
					float value = values[buffer_index];
					result.push_back(std::make_pair(pose * (cell_size * v.matrix().cast<float>()), value));
				}
			}
		}
	}
}


// todo: run length encoding to CPU memory?  Lets us compress non-tsdf volumes out of GPU memory
// also possible we want a TSDF equivalent which does this instead...

// todo: add volume here instead?  NO, that's weighted...

