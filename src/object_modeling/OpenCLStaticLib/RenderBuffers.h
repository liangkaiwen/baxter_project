#pragma once

#include "cll.h"
#include "ImageBuffer.h"
#include "OpenCLAllKernels.h"


class RenderBuffers
{
public:
    RenderBuffers(boost::shared_ptr<OpenCLAllKernels> all_kernels);

	void setSize(size_t new_width, size_t new_height);
    size_t getWidth() const {return width_;}
    size_t getHeight() const {return height_;}

	void resetRenderMask();
	void resetPointsBuffer();
	void resetNormalsBuffer();
	void resetColorsBuffer();
	void resetAllBuffers();

	// These are smart though
    const ImageBuffer & getImageBufferMask() const { return image_buffer_mask_; }
    const ImageBuffer & getImageBufferPoints() const { return image_buffer_points_; }
    const ImageBuffer & getImageBufferNormals() const { return image_buffer_normals_; }
    const ImageBuffer & getImageBufferColorImage() const { return image_buffer_color_image_; }
	ImageBuffer & getImageBufferMask() { return image_buffer_mask_; }
    ImageBuffer & getImageBufferPoints() { return image_buffer_points_; }
    ImageBuffer & getImageBufferNormals() { return image_buffer_normals_; }
    ImageBuffer & getImageBufferColorImage() { return image_buffer_color_image_; }


    // get a pretty version of the render (was in volume modeler)
    void getRenderPretty(cv::Mat & render_color, cv::Mat & render_normals);


protected:
	// methods
    size_t getImageSize() const {return width_ * height_;}
	void resizeImageBuffers(); // from current width and height

	// members
    boost::shared_ptr<OpenCLAllKernels> all_kernels_;
    size_t width_;
    size_t height_;

	// buffers:
    ImageBuffer image_buffer_mask_;
    ImageBuffer image_buffer_points_;
    ImageBuffer image_buffer_normals_;
    ImageBuffer image_buffer_color_image_;


};

