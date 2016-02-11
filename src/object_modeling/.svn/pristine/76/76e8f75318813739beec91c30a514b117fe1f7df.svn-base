#include "stdafx.h"
#include "ImageBuffer.h"

using std::cout;
using std::endl;

ImageBuffer::ImageBuffer(CL& cl)
	: cl_(cl),
	buffer_wrapper_(cl),
	rows_(0),
	cols_(0),
	channels_(0),
	element_cv_type_(0)
{
}

ImageBuffer::ImageBuffer(ImageBuffer const& other)
	: cl_(other.cl_),
	buffer_wrapper_(other.buffer_wrapper_),
	rows_(other.rows_),
	cols_(other.cols_),
	channels_(other.channels_),
	element_cv_type_(other.element_cv_type_)
{

}

ImageBuffer& ImageBuffer::operator=(const ImageBuffer& other)
{
	cl_ = other.cl_;
	buffer_wrapper_ = other.buffer_wrapper_;
	rows_ = other.rows_;
	cols_ = other.cols_;
	channels_ = other.channels_;
	element_cv_type_ = other.element_cv_type_;
	return *this;
}

void ImageBuffer::resize(int rows, int cols, int channels, int element_cv_type)
{
	rows_ = rows;
	cols_ = cols;
	channels_ = channels;
	element_cv_type_ = element_cv_type;
	buffer_wrapper_.reallocateIfNeeded(getSizeBytes());
}

cv::Mat ImageBuffer::getMat() const
{
	// ugly as fuck
	cv::Mat result;
	if (element_cv_type_ == CV_8U) {
		result = cv::Mat(rows_, cols_, CV_8UC(channels_));
	}
	else if (element_cv_type_ == CV_32F) {
		result = cv::Mat(rows_, cols_, CV_32FC(channels_));
	}
	else if (element_cv_type_ == CV_32S) {
		result = cv::Mat(rows_, cols_, CV_32SC(channels_));
	}
	else {
		cout << "unsupported element_cv_type_: " << element_cv_type_ << endl;
		throw std::runtime_error("unsupported element_cv_type_");
	}
	buffer_wrapper_.readToBytePointer(result.data, getSizeBytes());
	return result;
}

void ImageBuffer::setMat(cv::Mat const& mat)
{
	resize(mat.rows, mat.cols, mat.channels(), mat.depth());
	buffer_wrapper_.writeFromBytePointer(mat.data, getSizeBytes());
}

Eigen::Matrix4Xf ImageBuffer::getMatrix4Xf() const
{
	cv::Mat mat = getMat();
	if (mat.type() != CV_32FC4) {
		cout << "wrong type for getMatrix4Xf" << endl;
		throw std::runtime_error("getMatrix4Xf");
	}
	return Eigen::Map<Eigen::Matrix4Xf>((float*)mat.data, 4, mat.total());
}
