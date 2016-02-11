#include "stdafx.h"
#include "RenderBuffers.h"

#include "KernelSetFloat.h"
#include "KernelSetInt.h"
#include "KernelSetUChar.h"
#include "KernelNormalsToShadedImage.h"

using std::cout;
using std::endl;

RenderBuffers::RenderBuffers(boost::shared_ptr<OpenCLAllKernels> all_kernels)
    : all_kernels_(all_kernels),
      width_(0),
      height_(0),
      image_buffer_mask_(all_kernels->getCL()),
      image_buffer_points_(all_kernels->getCL()),
      image_buffer_normals_(all_kernels->getCL()),
      image_buffer_color_image_(all_kernels->getCL())
{
}


void RenderBuffers::setSize(size_t new_width, size_t new_height)
{
    width_ = new_width;
    height_ = new_height;
    resizeImageBuffers();
}

void RenderBuffers::resizeImageBuffers()
{
    image_buffer_mask_.resize(height_, width_, 1, CV_32S);
    image_buffer_points_.resize(height_, width_, 4, CV_32F);
    image_buffer_normals_.resize(height_, width_, 4, CV_32F);
    image_buffer_color_image_.resize(height_, width_, 4, CV_8U);
}

void RenderBuffers::resetRenderMask()
{
    KernelSetInt _KernelSetInt(*all_kernels_);
    _KernelSetInt.runKernel(image_buffer_mask_.getBuffer(), image_buffer_mask_.getSizeBytes() / sizeof(int), 0);
}

void RenderBuffers::resetPointsBuffer()
{
    KernelSetFloat _KernelSetFloat(*all_kernels_);
    _KernelSetFloat.runKernel(image_buffer_points_.getBuffer(), image_buffer_points_.getSizeBytes() / sizeof(float), std::numeric_limits<float>::quiet_NaN());
}

void RenderBuffers::resetNormalsBuffer()
{
    KernelSetFloat _KernelSetFloat(*all_kernels_);
    _KernelSetFloat.runKernel(image_buffer_normals_.getBuffer(), image_buffer_normals_.getSizeBytes() / sizeof(float), std::numeric_limits<float>::quiet_NaN());
}

void RenderBuffers::resetColorsBuffer()
{
    KernelSetUChar _KernelSetUChar(*all_kernels_);
    _KernelSetUChar.runKernel(image_buffer_color_image_.getBuffer(), image_buffer_color_image_.getSizeBytes() / sizeof(uint8_t), 0);
}

void RenderBuffers::resetAllBuffers()
{
    resetRenderMask();
    resetPointsBuffer();
    resetNormalsBuffer();
    resetColorsBuffer();
}

// this is probably alright...
void RenderBuffers::getRenderPretty(cv::Mat & render_color, cv::Mat & render_normals)
{
    render_color = image_buffer_color_image_.getMat();

    Eigen::Vector3f vector_to_light = Eigen::Vector3f(1,1,-1).normalized();
    ImageBuffer normals_image(all_kernels_->getCL());

    KernelNormalsToShadedImage _KernelNormalsToShadedImage(*all_kernels_);
    _KernelNormalsToShadedImage.runKernel(image_buffer_normals_, normals_image, vector_to_light);
    render_normals = normals_image.getMat();
}


