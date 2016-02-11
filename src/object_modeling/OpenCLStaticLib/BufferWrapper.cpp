#include "stdafx.h"
#include "BufferWrapper.h"

#include <cl_util.h>

#include <iostream>
using std::cout;
using std::endl;

BufferWrapper::BufferWrapper(CL& cl)
	: cl(cl),
	buffer_size(0)
{
}

BufferWrapper::BufferWrapper(BufferWrapper const& other)
	: cl(other.cl),
	buffer_size(other.buffer_size),
	buffer(other.buffer)
{

}

// required because cl is a reference
BufferWrapper& BufferWrapper::operator=(const BufferWrapper& other)
{
	cl = other.cl;
	buffer_size = other.buffer_size;
	buffer = other.buffer;
	return *this;
}

void BufferWrapper::reallocate(size_t size)
{
    if (size == buffer_size) return;

	buffer_size = size;
	if (buffer_size > 0) {
		buffer = cl::Buffer (cl.context, 0, getBufferSize());
	}
	else {
		buffer = cl::Buffer();
	}
}

void BufferWrapper::reallocateIfNeeded(size_t size)
{
	if (size > buffer_size) {
		reallocate(size);
	}
}

void BufferWrapper::readToIntVector(std::vector<int>& result) const
{
	size_t read_size = result.size() * sizeof(int);
	if (read_size > buffer_size) throw std::runtime_error ("OOB");
    enqueueReadBufferCheckError(read_size, result.data());

}

void BufferWrapper::writeFromIntVector(std::vector<int> const& input)
{
	if (input.empty()) return;
	size_t write_size = input.size() * sizeof(int);
	reallocateIfNeeded(write_size);
    enqueueWriteBufferCheckError(write_size, input.data());
}

void BufferWrapper::readToFloatVector(std::vector<float>& result) const
{
	size_t read_size = result.size() * sizeof(float);
	if (read_size > buffer_size) throw std::runtime_error ("OOB");
    enqueueReadBufferCheckError(read_size, result.data());
}

void BufferWrapper::readToFloatVectorAll(std::vector<float>& result) const
{
	result.resize(buffer_size / sizeof(float));
	size_t read_size = result.size() * sizeof(float);
	if (read_size > buffer_size) throw std::runtime_error ("OOB");
    enqueueReadBufferCheckError(read_size, result.data());
}

void BufferWrapper::writeFromFloatVector(std::vector<float> const& input)
{
	if (input.empty()) return;
	size_t write_size = input.size() * sizeof(float);
	reallocateIfNeeded(write_size);
    enqueueWriteBufferCheckError(write_size, input.data());
}

void BufferWrapper::readToByteVector(std::vector<uint8_t>& result) const
{
	size_t read_size = result.size() * sizeof(uint8_t);
	if (read_size > buffer_size) throw std::runtime_error ("OOB");
    enqueueReadBufferCheckError(read_size, result.data());
}

void BufferWrapper::writeFromByteVector(std::vector<uint8_t> const& input)
{
	if (input.empty()) return;
	size_t write_size = input.size() * sizeof(uint8_t);
	reallocateIfNeeded(write_size);
    enqueueWriteBufferCheckError(write_size, input.data());
}

void BufferWrapper::readToBytePointer(uint8_t* result, size_t size) const
{
	size_t read_size = size * sizeof(uint8_t);
	if (read_size > buffer_size) throw std::runtime_error ("OOB");
    enqueueReadBufferCheckError(read_size, result);
}

void BufferWrapper::writeFromBytePointer(uint8_t* input, size_t size)
{
	size_t write_size = size * sizeof(uint8_t);
	reallocateIfNeeded(write_size);
    enqueueWriteBufferCheckError(write_size, input);
}


void BufferWrapper::writeFromByteBuffer(cl::Buffer const& input_buffer, size_t size)
{
	reallocateIfNeeded(size);
	cl.queue.enqueueCopyBuffer(input_buffer, buffer, 0, 0, size);
}


// these are stupid...should probably just wrap all execution up in a single try block ;)
void BufferWrapper::enqueueReadBufferCheckError(size_t size, void* ptr) const
{
    try {
        cl.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, ptr);
    }
    catch (cl::Error er) {
        cout << "enqueueReadBufferCheckError" << endl;
        cout << "cl::Error: " << oclErrorString(er.err()) << endl;
        throw er;
    }
}
void BufferWrapper::enqueueWriteBufferCheckError(size_t size, const void* ptr)
{
    try {
        cl.queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, ptr);
    }
    catch (cl::Error er) {
        cout << "enqueueWriteBufferCheckError" << endl;
        cout << "cl::Error: " << oclErrorString(er.err()) << endl;
        throw er;
    }
}
