#pragma once

#include <stdint.h>

#include "cll.h"

#include <boost/shared_ptr.hpp>

class BufferWrapper
{
public:
	BufferWrapper(CL& cl);
	BufferWrapper(BufferWrapper const& other); // newly added
	BufferWrapper& operator=(const BufferWrapper& other);

	void reallocate(size_t size);
	void reallocateIfNeeded(size_t size);
	
	size_t getBufferSize() const { return buffer_size; }
	
	const cl::Buffer & getBuffer() const { return buffer; }
	cl::Buffer & getBuffer() { return buffer; }

	void readToIntVector(std::vector<int>& result) const;
	void writeFromIntVector(std::vector<int> const& input);
	void readToFloatVector(std::vector<float>& result) const;
	void readToFloatVectorAll(std::vector<float>& result) const;
	void writeFromFloatVector(std::vector<float> const& input);
	void readToByteVector(std::vector<uint8_t>& result) const;
	void writeFromByteVector(std::vector<uint8_t> const& input);
	void readToBytePointer(uint8_t* result, size_t size) const;
	void writeFromBytePointer(uint8_t* input, size_t size);

	// copy values from a cl buffer
	void writeFromByteBuffer(cl::Buffer const& input_buffer, size_t size);

protected:

    void enqueueReadBufferCheckError(size_t size, void* ptr) const;
    void enqueueWriteBufferCheckError(size_t size, const void *ptr);

	CL& cl;
	size_t buffer_size;
	cl::Buffer buffer;
	// also last write size??
};

