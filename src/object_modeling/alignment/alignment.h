#pragma once

#include "RenderBuffers.h"
#include "OpenCLAllKernels.h"

#include "OpenCLOptimize.h"
#include "OpenCLOptimizeNew.h"
#include "OpenCLImages.h"

#include "params_alignment.h"
#include "params_camera.h"

class Alignment
{
public:
    Alignment(boost::shared_ptr<OpenCLAllKernels> all_kernels, ParamsAlignment const& params_alignment);
    virtual ~Alignment() {};

    // you used to split out prepareFrame so loop closure wouldn't need to mess with the frame again
    // New note: probably not that important... call alignMultiscaleNew directly
    // Also note you weren't taking advantage of "prepare" for multiscale ever!
    bool align(
        ImageBuffer const& frame_buffer_color,
        ImageBuffer const& frame_buffer_points,
        ImageBuffer const& frame_buffer_normals,
        ImageBuffer const& frame_buffer_align_weights,
        ImageBuffer const& render_buffer_color,
        ImageBuffer const& render_buffer_points,
        ImageBuffer const& render_buffer_normals,
        ImageBuffer const& render_buffer_mask,
        ParamsCamera const& params_camera,
        Eigen::Affine3f const& render_pose,
        Eigen::Affine3f const& initial_relative_pose, 
        Eigen::Affine3f & result_pose,
        std::vector<int> & result_iterations);

    bool alignMultiscaleNew(
        ImageBuffer const& frame_buffer_color,
        ImageBuffer const& frame_buffer_points,
        ImageBuffer const& frame_buffer_normals,
        ImageBuffer const& frame_buffer_align_weights,
        ImageBuffer const& render_buffer_color,
        ImageBuffer const& render_buffer_points,
        ImageBuffer const& render_buffer_normals,
        ParamsCamera const& params_camera,
        Eigen::Affine3f const& render_pose,
        Eigen::Affine3f const& initial_relative_pose, 
        Eigen::Affine3f & result_pose,
        std::vector<int> & result_iterations);

    void setAlignDebugImages(bool value);

    void getAlignDebugImages(std::vector<cv::Mat> & image_list);

    void getPyramidDebugImages(std::vector<cv::Mat> & image_list);

protected:
    void prepareFrame(
        ImageBuffer const& image_buffer_color,
        ImageBuffer const& image_buffer_points,
        ImageBuffer const& image_buffer_normals,
        ImageBuffer const& image_buffer_align_weights,
        ParamsCamera const& params_camera);

    bool alignWithCombinedOptimization(ImageBuffer const& image_buffer_color,
        ImageBuffer const& image_buffer_points,
        ImageBuffer const& image_buffer_normals,
        ImageBuffer const& image_buffer_mask, // only for debug images!
        ParamsCamera const& params_camera,
        Eigen::Affine3f const& render_pose,
        Eigen::Affine3f const& initial_relative_pose,
        Eigen::Affine3f & result_pose,
        int & iterations);

    bool alignMultiscale(ImageBuffer const& frame_buffer_color,
        ImageBuffer const& frame_buffer_points,
        ImageBuffer const& frame_buffer_normals,
        ImageBuffer const& frame_buffer_align_weights,
        ImageBuffer const& render_buffer_color,
        ImageBuffer const& render_buffer_points,
        ImageBuffer const& render_buffer_normals,
        ImageBuffer const& render_buffer_mask, // only for debug images!
        ParamsCamera const& params_camera,
        Eigen::Affine3f const& render_pose,
        Eigen::Affine3f const& initial_relative_pose,
        Eigen::Affine3f & result_pose,
        std::vector<int> & result_iterations);


    std::vector<ImageBuffer> getImageChannelsList(ImageBuffer const& color_bgra_uchar, int width, int height);

    ImageBuffer packImageChannelsList(std::vector<ImageBuffer> const& image_buffer_list, int width, int height);
    void unpackImageChannelsList(ImageBuffer const& packed_image, int channels, int width, int height, std::vector<ImageBuffer> & result_image_buffer_list);

    Eigen::Matrix4f getTransformMatrixFor6DOFQuaternion(Eigen::VectorXf const& p);

    // new style:
    cv::Mat getSingleDebugImage(const OpenCLOptimizeNew::OptimizeDebugImages &optimize_debug_images, float scale);

    cv::Mat getDebugImage(cv::Mat const& render_mask, std::vector<float> const& error_vector, std::vector<float> const& weight_vector, int error_channel_count, float scale);

    Eigen::Array2i getCameraSizeForLevel(ParamsCamera const& params_camera, int level);
    ParamsCamera getCameraForLevel(ParamsCamera const& params_camera, int level);
    cv::Rect getRectForLevel(cv::Rect const& rect, int level);

    void createPyramidImage(ImageBuffer const& image_buffer_color, ParamsCamera const& params_camera, int levels, std::vector<std::vector<ImageBuffer> > & result_pyramid_image_lists);
    void createPyramidFloat4(ImageBuffer const& image_buffer_float4, ParamsCamera const& params_camera, int levels, std::vector<ImageBuffer> & result_pyramid);
    void createPyramidFloat(ImageBuffer const& image_buffer_float, ParamsCamera const& params_camera, int levels, std::vector<ImageBuffer> & result_pyramid);
    void createPyramidGradients(std::vector<std::vector<ImageBuffer> > const& pyramid_image_lists, ParamsCamera const& params_camera, 
        std::vector<std::vector<ImageBuffer> > & result_pyramid_gradient_x, std::vector<std::vector<ImageBuffer> > & result_pyramid_gradient_y);

    void iterate(Eigen::Affine3f const& initial_relative_pose, int max_iterations, cv::Mat const& render_mask, int image_channels, int level, int & iterations, Eigen::Affine3f & pose_correction);

    // new alternate iterate
    void iterateNew(
        const ImageBuffer & frame_buffer_points,
        const ImageBuffer & frame_buffer_normals,
        const ImageBuffer & frame_buffer_image, // just 1 for now
        const ImageBuffer & frame_buffer_image_gradient_x,  // just 1 for now
        const ImageBuffer & frame_buffer_image_gradient_y,  // just 1 for now
        const ImageBuffer & render_buffer_points,
        const ImageBuffer & render_buffer_normals,
        const ImageBuffer & render_buffer_image,  // just 1 for now
        const ParamsCamera & params_camera_frame,
        const ParamsCamera & params_camera_render,
        const Eigen::Affine3f & initial_relative_pose,
        const int max_iterations,
        int level,
        int & iterations,
        Eigen::Affine3f & pose_correction);


    std::vector<cv::Mat> getPyramidImages(std::vector<std::vector<ImageBuffer> > const& pyr_image_list);
    std::vector<cv::Mat> getPyramidSingle(std::vector<ImageBuffer> const& pyr_images);

    // members
    boost::shared_ptr<OpenCLAllKernels> all_kernels_;

    ParamsAlignment params_alignment_;

    boost::shared_ptr<OpenCLImages> opencl_images_ptr_;
    boost::shared_ptr<OpenCLOptimize> opencl_optimize_ptr_;
    boost::shared_ptr<OpenCLOptimizeNew> opencl_optimize_new_ptr_;

    std::vector<cv::Mat> alignment_debug_images_;
    std::vector<cv::Mat> pyramid_debug_images_;

};
