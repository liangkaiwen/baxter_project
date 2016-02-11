#pragma once

#include <stdint.h>

#include "cll.h"

#include "EigenUtilities.h"

#include "BufferWrapper.h"

#include <opencv2/opencv.hpp> // also in stdafx

class ImageBuffer
{
public:
	ImageBuffer(CL& cl);
	ImageBuffer(ImageBuffer const& other); // newly added
	ImageBuffer& operator=(const ImageBuffer& other);

	const BufferWrapper & getBufferWrapper() const { return buffer_wrapper_; }
	const cl::Buffer & getBuffer() const { return buffer_wrapper_.getBuffer(); }
	BufferWrapper & getBufferWrapper() { return buffer_wrapper_; }
	cl::Buffer & getBuffer() { return buffer_wrapper_.getBuffer(); }

	size_t getSizeBytes() const { return rows_ * cols_ * channels_ * CV_ELEM_SIZE1(element_cv_type_); }
	size_t getSizeElements() const { return rows_ * cols_ * channels_; }

	int getRows() const { return rows_; }
	int getCols() const { return cols_; }
	int getChannels() const { return channels_; }
	int getElementCvType() const { return element_cv_type_; }

	void resize(int rows, int cols, int channels, int element_cv_type);
	cv::Mat getMat() const;
	void setMat(cv::Mat const& mat);

	Eigen::Matrix4Xf getMatrix4Xf() const;

protected:
	CL& cl_;
	BufferWrapper buffer_wrapper_;
	int rows_;
	int cols_;
	int channels_;
	int element_cv_type_;
};

