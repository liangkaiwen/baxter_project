#pragma once

#include <stdint.h>

#include "cll.h"

#include "BufferWrapper.h"
#include "OpenCLAllKernels.h"


class VolumeBuffer
{
public:
	VolumeBuffer(boost::shared_ptr<OpenCLAllKernels> all_kernels);
	VolumeBuffer(boost::shared_ptr<OpenCLAllKernels> all_kernels, const Eigen::Array3i & cell_counts, size_t element_byte_size);
	VolumeBuffer(VolumeBuffer const& other);
	VolumeBuffer& operator=(const VolumeBuffer& other);

    const BufferWrapper & getBufferWrapper() const { return buffer_wrapper_; }
	BufferWrapper & getBufferWrapper() { return buffer_wrapper_; }
    
	const cl::Buffer & getBuffer() const { return buffer_wrapper_.getBuffer(); }
    cl::Buffer & getBuffer() { return buffer_wrapper_.getBuffer(); }

    Eigen::Array3i getVolumeCellCounts() const { return volume_cell_counts_;}
	size_t getElementByteSize() const { return element_byte_size_;}

    size_t getSizeInCells() const;
    size_t getSizeInBytes() const;

	void resize(Eigen::Array3i const& volume_cell_counts, size_t element_byte_size);

	void setFloat(float value);
	void setUChar(unsigned char value);
    void setInt(int value);
    void setFloat4(Eigen::Array4f const& value);

	void getNonzeroPointsAndFloatValues(const Eigen::Affine3f& pose, float cell_size, float epsilon, std::vector<std::pair<Eigen::Vector3f, float> >& result);

	void deallocate();

protected:
	boost::shared_ptr<OpenCLAllKernels> all_kernels_;
	BufferWrapper buffer_wrapper_;
	Eigen::Array3i volume_cell_counts_;
	size_t element_byte_size_;
};

