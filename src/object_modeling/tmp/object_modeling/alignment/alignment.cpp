#include "alignment.h"

#include "opencv_utilities.h"

#include <boost/foreach.hpp>

using std::cout;
using std::endl;

Alignment::Alignment(boost::shared_ptr<OpenCLAllKernels> all_kernels, ParamsAlignment const& params_alignment)
    : all_kernels_(all_kernels),
    params_alignment_(params_alignment)
{
    // holy hell...how long has this been here?
    const int image_channel_count = 1;

    // try to make sure we're using the right code always
    if (params_alignment_.use_new_alignment) {
        opencl_optimize_new_ptr_.reset(new OpenCLOptimizeNew(all_kernels_));
    }
    else {
        opencl_optimize_ptr_.reset(new OpenCLOptimize(all_kernels_, image_channel_count));
    }

    opencl_images_ptr_.reset(new OpenCLImages(*all_kernels_));
}

// NOTE: YOU CAN JUST CALL alignMultiscaleNew directly!!
// benefit: no mask needed
bool Alignment::align(ImageBuffer const& frame_buffer_color,
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
    std::vector<int> & result_iterations)
{
    // new alignment is only multiscale
    if (params_alignment_.use_new_alignment) {
        return alignMultiscaleNew(
            frame_buffer_color,
            frame_buffer_points,
            frame_buffer_normals,
            frame_buffer_align_weights,
            render_buffer_color,
            render_buffer_points,
            render_buffer_normals,
            params_camera,
            render_pose,
            initial_relative_pose,
            result_pose,
            result_iterations);
    }
    else if (params_alignment_.use_multiscale) {
        return alignMultiscale(
                frame_buffer_color,
                frame_buffer_points,
                frame_buffer_normals,
                frame_buffer_align_weights,
                render_buffer_color,
                render_buffer_points,
                render_buffer_normals,
                render_buffer_mask,
                params_camera,
                render_pose,
                initial_relative_pose,
                result_pose,
                result_iterations);
    }
    else {
        prepareFrame(
            frame_buffer_color,
            frame_buffer_points,
            frame_buffer_normals,
            frame_buffer_align_weights,
            params_camera);
        result_iterations.clear();
        result_iterations.push_back(0);
        return alignWithCombinedOptimization(
            render_buffer_color,
            render_buffer_points,
            render_buffer_normals,
            render_buffer_mask,
            params_camera,
            render_pose,
            initial_relative_pose,
            result_pose,
            result_iterations[0]);
    }

    cout << "no way you're here!!" << endl;
}

void Alignment::prepareFrame(
    ImageBuffer const& image_buffer_color,
    ImageBuffer const& image_buffer_points,
    ImageBuffer const& image_buffer_normals,
    ImageBuffer const& image_buffer_align_weights,
    ParamsCamera const& params_camera)
{
    const int rows = params_camera.size.y();
    const int cols = params_camera.size.x();

    ///////////
    // initialize frame
    std::vector<ImageBuffer> frame_image_channels = getImageChannelsList(image_buffer_color, cols, rows);

    // gaussian blur
    std::vector<ImageBuffer> split_blurred;
    if (params_alignment_.color_blur_size > 0) {
        cv::Mat gaussianCoeffs1D = cv::getGaussianKernel(params_alignment_.color_blur_size, -1, CV_32F);
        for (size_t i = 0; i < frame_image_channels.size(); ++i) {
            ImageBuffer temp = opencl_images_ptr_->convolutionFilterHorizontal(frame_image_channels[i], cols, rows, params_alignment_.color_blur_size, (float*)gaussianCoeffs1D.data);
            split_blurred.push_back(opencl_images_ptr_->convolutionFilterVertical(temp, cols, rows, params_alignment_.color_blur_size, (float*)gaussianCoeffs1D.data));
        }
    }
    else {
        for (size_t i = 0; i < frame_image_channels.size(); ++i) {
            split_blurred.push_back(frame_image_channels[i]);
        }
    }

    // sobel filter
    float sobel_smooth[]= {.25f, .5f, .25f};
    float sobel_diff[]= {-.5f, 0.f, .5f};
    std::vector<ImageBuffer> split_gradient_x;
    std::vector<ImageBuffer> split_gradient_y;
    for (size_t i = 0; i < split_blurred.size(); ++i) {
        ImageBuffer temp = opencl_images_ptr_->convolutionFilterVertical(split_blurred[i], cols, rows, 3, sobel_smooth);
        split_gradient_x.push_back(opencl_images_ptr_->convolutionFilterHorizontal(temp, cols, rows, 3, sobel_diff));
        temp = opencl_images_ptr_->convolutionFilterHorizontal(split_blurred[i], cols, rows, 3, sobel_smooth);
        split_gradient_y.push_back(opencl_images_ptr_->convolutionFilterVertical(temp, cols, rows, 3, sobel_diff));
    }

    ////// put this in function for multiscale
    // these are passed to optimizer
    ImageBuffer frame_image_buffer(all_kernels_->getCL());
    ImageBuffer frame_gradient_x_buffer(all_kernels_->getCL());
    ImageBuffer frame_gradient_y_buffer(all_kernels_->getCL());

    frame_image_buffer = packImageChannelsList(split_blurred, cols, rows);
    frame_gradient_x_buffer = packImageChannelsList(split_gradient_x, cols, rows);
    frame_gradient_y_buffer = packImageChannelsList(split_gradient_y, cols, rows);

    opencl_optimize_ptr_->prepareFrameBuffersWithBuffers(
        params_camera.focal[0], params_camera.focal[1], params_camera.center[0], params_camera.center[1], params_camera.size[0], params_camera.size[1],
        image_buffer_points.getBuffer(), image_buffer_normals.getBuffer(), 
        frame_image_buffer.getBuffer(), frame_gradient_x_buffer.getBuffer(), frame_gradient_y_buffer.getBuffer(),
        image_buffer_align_weights.getBuffer());
}

Eigen::Array2i Alignment::getCameraSizeForLevel(ParamsCamera const& params_camera, int level)
{
    return params_camera.size / (1 << level);
}

ParamsCamera Alignment::getCameraForLevel(ParamsCamera const& params_camera, int level)
{
    ParamsCamera result;
    result.center = params_camera.center / (1 << level);
    result.focal = params_camera.focal / (1 << level);
    result.size = params_camera.size / (1 << level);
    return result;
}

cv::Rect Alignment::getRectForLevel(cv::Rect const& rect, int level)
{
    cv::Rect result;
    result.x = rect.x / (1 << level);
    result.y = rect.y / (1 << level);
    result.width = rect.width / (1 << level);
    result.height = rect.height / (1 << level);
    return result;
}

void Alignment::createPyramidImage(ImageBuffer const& image_buffer_color, ParamsCamera const& params_camera, int levels, std::vector<std::vector<ImageBuffer> > & result_pyramid_image_lists)
{
    result_pyramid_image_lists.clear();

    std::vector<ImageBuffer> images_split = getImageChannelsList(image_buffer_color, params_camera.size.x(), params_camera.size.y());
    const int channels = images_split.size();

    const int downsample_gaussian_width = 5;
    cv::Mat gaussianCoeffs1D = cv::getGaussianKernel(downsample_gaussian_width, -1, CV_32F);

    for (int level = 0; level < levels; ++level) {
        std::vector<ImageBuffer> images_this_level;

        if (level == 0) {
            images_this_level = images_split;
        }
        else {
            // need to extract and downsample the previous level images
            Eigen::Array2i size_previous_level = getCameraSizeForLevel(params_camera, level-1);

            std::vector<ImageBuffer> const& previous_image_list = result_pyramid_image_lists[level-1];
            images_this_level.clear();

            // could avoid all this allocation by alternating and allocating outside loop
            for (int c = 0; c < previous_image_list.size(); ++c) {
                ImageBuffer temp = opencl_images_ptr_->convolutionFilterHorizontal(previous_image_list[c], size_previous_level.x(), size_previous_level.y(), downsample_gaussian_width, (float*)gaussianCoeffs1D.data);
                ImageBuffer temp2 = opencl_images_ptr_->convolutionFilterVertical(temp, size_previous_level.x(), size_previous_level.y(), downsample_gaussian_width, (float*)gaussianCoeffs1D.data);
                ImageBuffer temp3 = opencl_images_ptr_->halfSizeImage(temp2);
                images_this_level.push_back(temp3);
            }
        }

        result_pyramid_image_lists.push_back(images_this_level);
    }
}

void Alignment::createPyramidFloat4(ImageBuffer const& image_buffer_float4, ParamsCamera const& params_camera, int levels, std::vector<ImageBuffer> & result_pyramid)
{
    result_pyramid.clear();

    for (int level = 0; level < levels; ++level) {
        if (level == 0) {
            result_pyramid.push_back(image_buffer_float4);
        }
        else {
            bool no_mean = false;
            if (no_mean) {
                result_pyramid.push_back(opencl_images_ptr_->halfSizeFloat4(result_pyramid[level-1]));
            }
            else {
                result_pyramid.push_back(opencl_images_ptr_->halfSizeFloat4Mean(result_pyramid[level-1]));
            }
        }
    }
}

void Alignment::createPyramidFloat(ImageBuffer const& image_buffer_float, ParamsCamera const& params_camera, int levels, std::vector<ImageBuffer> & result_pyramid)
{
    result_pyramid.clear();

    for (int level = 0; level < levels; ++level) {
        if (level == 0) {
            result_pyramid.push_back(image_buffer_float);
        }
        else {
            Eigen::Array2i size_previous_level = getCameraSizeForLevel(params_camera, level-1);
            result_pyramid.push_back(opencl_images_ptr_->halfSizeImage(result_pyramid[level-1]));
        }
    }
}

void Alignment::createPyramidGradients(std::vector<std::vector<ImageBuffer> > const& pyramid_image_lists, ParamsCamera const& params_camera, 
    std::vector<std::vector<ImageBuffer> > & result_pyramid_gradient_x, std::vector<std::vector<ImageBuffer> > & result_pyramid_gradient_y)
{
    result_pyramid_gradient_x.clear();
    result_pyramid_gradient_y.clear();

    for (int level = 0; level < pyramid_image_lists.size(); ++level) {
        Eigen::Array2i size_this_level = getCameraSizeForLevel(params_camera, level);

        std::vector<ImageBuffer> const& images_this_level = pyramid_image_lists[level];

        // need to compute gradients even on first
        float sobel_smooth[]= {.25f, .5f, .25f};
        float sobel_diff[]= {-.5f, 0.f, .5f};
        std::vector<ImageBuffer> split_gradient_x;
        std::vector<ImageBuffer> split_gradient_y;
        for (int c = 0; c < images_this_level.size(); ++c) {
            ImageBuffer temp = opencl_images_ptr_->convolutionFilterVertical(images_this_level[c], size_this_level.x(), size_this_level.y(), 3, sobel_smooth);
            split_gradient_x.push_back(opencl_images_ptr_->convolutionFilterHorizontal(temp, size_this_level.x(), size_this_level.y(), 3, sobel_diff));
            temp = opencl_images_ptr_->convolutionFilterHorizontal(images_this_level[c], size_this_level.x(), size_this_level.y(), 3, sobel_smooth);
            split_gradient_y.push_back(opencl_images_ptr_->convolutionFilterVertical(temp, size_this_level.x(), size_this_level.y(), 3, sobel_diff));
        }

        result_pyramid_gradient_x.push_back(split_gradient_x);
        result_pyramid_gradient_y.push_back(split_gradient_y);
    }
}

// TODO: clean this up back to old code maybe?
// probably not...though you can get your old alignment code back
bool Alignment::alignMultiscale(
    ImageBuffer const& frame_buffer_color,
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
    std::vector<int> & result_iterations)
{
    if (params_alignment_.use_new_alignment) {
        cout << "alignMultiscale not designed for  use_new_alignment" << endl;
        throw std::runtime_error("alignMultiscale");
    }

    cout << "DEBUG REMOVE" << endl;
    all_kernels_->getCL().queue.finish();

    // create pyramids
    std::vector<std::vector<ImageBuffer> > pyr_frame_image_list;
    std::vector<std::vector<ImageBuffer> > pyr_gradient_x;
    std::vector<std::vector<ImageBuffer> > pyr_gradient_y;
    std::vector<ImageBuffer> pyr_frame_points;
    std::vector<ImageBuffer> pyr_frame_normals;
    std::vector<ImageBuffer> pyr_frame_align_weights;
    createPyramidImage(frame_buffer_color, params_camera, params_alignment_.pyramid_levels, pyr_frame_image_list);

    cout << "DEBUG REMOVE" << endl;
    all_kernels_->getCL().queue.finish();

    createPyramidFloat4(frame_buffer_points, params_camera, params_alignment_.pyramid_levels, pyr_frame_points);
    createPyramidFloat4(frame_buffer_normals, params_camera, params_alignment_.pyramid_levels, pyr_frame_normals);
    createPyramidFloat(frame_buffer_align_weights, params_camera, params_alignment_.pyramid_levels, pyr_frame_align_weights);
    createPyramidGradients(pyr_frame_image_list, params_camera, pyr_gradient_x, pyr_gradient_y);

    std::vector<std::vector<ImageBuffer> > pyr_render_image_list;
    std::vector<ImageBuffer> pyr_render_points;
    std::vector<ImageBuffer> pyr_render_normals;
    std::vector<ImageBuffer> pyr_render_mask; // only for debug images
    createPyramidImage(render_buffer_color, params_camera, params_alignment_.pyramid_levels, pyr_render_image_list);
    createPyramidFloat4(render_buffer_points, params_camera, params_alignment_.pyramid_levels, pyr_render_points);
    createPyramidFloat4(render_buffer_normals, params_camera, params_alignment_.pyramid_levels, pyr_render_normals);
    createPyramidFloat(render_buffer_mask, params_camera, params_alignment_.pyramid_levels, pyr_render_mask); // abuse int == float

    // THIS IS THE LINE WHERE IT FUCKING CL_INVALID_COMMAND_QUEUE FOR SOME REASON ON LINUX
    // on like fucking frame 50
#if 0
    // same switch?
    if (params_alignment_.generate_debug_images) {
        cout << "Debug remove: about to getPyramidImages" << endl;

        // THIS IS THE LINE WHERE IT FUCKING CL_INVALID_COMMAND_QUEUE FOR SOME REASON ON LINUX
        // only during loop closure mind you...
        std::vector<cv::Mat> frame_images = getPyramidImages(pyr_frame_image_list);
        std::vector<cv::Mat> render_images = getPyramidImages(pyr_render_image_list);
        //std::vector<cv::Mat> grad_x = getPyramidImages(pyr_gradient_x);
        //std::vector<cv::Mat> grad_y = getPyramidImages(pyr_gradient_y);

        cout << "Debug remove: did getPyramidImages" << endl;

        pyramid_debug_images_.clear();
        for (int i = 0; i < frame_images.size(); ++i) {
            std::vector<cv::Mat> image_v;
            image_v.push_back(frame_images[i]);
            image_v.push_back(render_images[i]);
            cv::Mat combined_image = createMxN(1, 2, image_v);
            pyramid_debug_images_.push_back(combined_image);
        }
    }
#endif

    alignment_debug_images_.clear();
    result_iterations = std::vector<int>(params_alignment_.pyramid_levels, 0);

    // over levels
    Eigen::Affine3f initial_relative_pose_for_level = initial_relative_pose;
    for (int level = params_alignment_.pyramid_levels - 1; level >= 0; --level)
    {
        const ParamsCamera camera_level = getCameraForLevel(params_camera, level);
        const Eigen::Array2i & dims = camera_level.size;
        int image_channels = pyr_frame_image_list[level].size();

        // prepare optimizer
        ImageBuffer frame_image = pyr_frame_image_list[level][0];
        ImageBuffer frame_gradient_x = pyr_gradient_x[level][0];
        ImageBuffer frame_gradient_y = pyr_gradient_y[level][0];

        ImageBuffer & frame_points = pyr_frame_points[level];
        ImageBuffer & frame_normals = pyr_frame_normals[level];
        ImageBuffer & frame_align_weights = pyr_frame_align_weights[level];

        opencl_optimize_ptr_->prepareFrameBuffersWithBuffers(
            camera_level.focal[0], camera_level.focal[1], camera_level.center[0], camera_level.center[1], camera_level.size[0], camera_level.size[1],
            frame_points.getBuffer(), frame_normals.getBuffer(), 
            frame_image.getBuffer(), frame_gradient_x.getBuffer(), frame_gradient_y.getBuffer(),
            frame_align_weights.getBuffer());

        cv::Rect render_rect (0,0,params_camera.size.x(), params_camera.size.y());
        cv::Rect render_rect_level = getRectForLevel(render_rect, level);
        ImageBuffer render_image = packImageChannelsList(pyr_render_image_list[level], render_rect_level.width, render_rect_level.height);
        ImageBuffer & render_points = pyr_render_points[level];
        ImageBuffer & render_normals = pyr_render_normals[level];

        opencl_optimize_ptr_->prepareRenderedAndErrorBuffersWithBuffers(
            camera_level.focal[0], camera_level.focal[1], camera_level.center[0], camera_level.center[1],
            render_rect_level.x, render_rect_level.y, render_rect_level.width, render_rect_level.height,
            render_points.getBuffer(), render_normals.getBuffer(), render_image.getBuffer());

        // run the optimization

        // for debug images
        cv::Mat render_mask = cv::Mat();
        if (params_alignment_.generate_debug_images) {
            cv::Mat render_mask_int = pyr_render_mask[level].getMat();
            render_mask = render_mask_int > 0;
        }

        Eigen::Affine3f pose_correction;

        // old iterate
        iterate(initial_relative_pose_for_level, params_alignment_.gn_max_iterations, render_mask, image_channels, level, result_iterations[level], pose_correction);

        initial_relative_pose_for_level = pose_correction * initial_relative_pose_for_level;
    }

    {
        cout << "debug remove" << endl;
        float distance, angle;
        EigenUtilities::getAngleAndDistance(initial_relative_pose_for_level, distance, angle);
        cout << "initial_relative_pose_for_level distance: " << distance << endl;
        cout << "initial_relative_pose_for_level angle: " << angle << endl;
    }

    result_pose = initial_relative_pose_for_level * render_pose;

    return true;
}

bool Alignment::alignMultiscaleNew(
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
    std::vector<int> & result_iterations)
{
    // create pyramids
    std::vector<std::vector<ImageBuffer> > pyr_frame_image_list;
    std::vector<std::vector<ImageBuffer> > pyr_gradient_x;
    std::vector<std::vector<ImageBuffer> > pyr_gradient_y;
    std::vector<ImageBuffer> pyr_frame_points;
    std::vector<ImageBuffer> pyr_frame_normals;
    std::vector<ImageBuffer> pyr_frame_align_weights;
    createPyramidImage(frame_buffer_color, params_camera, params_alignment_.pyramid_levels, pyr_frame_image_list);
    createPyramidFloat4(frame_buffer_points, params_camera, params_alignment_.pyramid_levels, pyr_frame_points);
    createPyramidFloat4(frame_buffer_normals, params_camera, params_alignment_.pyramid_levels, pyr_frame_normals);
    createPyramidFloat(frame_buffer_align_weights, params_camera, params_alignment_.pyramid_levels, pyr_frame_align_weights);
    createPyramidGradients(pyr_frame_image_list, params_camera, pyr_gradient_x, pyr_gradient_y);

    std::vector<std::vector<ImageBuffer> > pyr_render_image_list;
    std::vector<ImageBuffer> pyr_render_points;
    std::vector<ImageBuffer> pyr_render_normals;
    std::vector<ImageBuffer> pyr_render_mask; // only for debug images
    createPyramidImage(render_buffer_color, params_camera, params_alignment_.pyramid_levels, pyr_render_image_list);
    createPyramidFloat4(render_buffer_points, params_camera, params_alignment_.pyramid_levels, pyr_render_points);
    createPyramidFloat4(render_buffer_normals, params_camera, params_alignment_.pyramid_levels, pyr_render_normals);
    //createPyramidFloat(render_buffer_mask, params_camera, params_alignment_.pyramid_levels, pyr_render_mask); // abuse int == float

    // THIS IS THE LINE WHERE IT FUCKING CL_INVALID_COMMAND_QUEUE FOR SOME REASON ON LINUX
    // on like fucking frame 50
    // note it still was failing with this removed
#if 0
    // same switch?
    if (params_alignment_.generate_debug_images) {
        cout << "Debug remove: about to getPyramidImages" << endl;

        // THIS IS THE LINE WHERE IT FUCKING CL_INVALID_COMMAND_QUEUE FOR SOME REASON ON LINUX
        // only during loop closure mind you...
        std::vector<cv::Mat> frame_images = getPyramidImages(pyr_frame_image_list);
        std::vector<cv::Mat> render_images = getPyramidImages(pyr_render_image_list);
        //std::vector<cv::Mat> grad_x = getPyramidImages(pyr_gradient_x);
        //std::vector<cv::Mat> grad_y = getPyramidImages(pyr_gradient_y);

        cout << "Debug remove: did getPyramidImages" << endl;

        pyramid_debug_images_.clear();
        for (int i = 0; i < frame_images.size(); ++i) {
            std::vector<cv::Mat> image_v;
            image_v.push_back(frame_images[i]);
            image_v.push_back(render_images[i]);
            cv::Mat combined_image = createMxN(1, 2, image_v);
            pyramid_debug_images_.push_back(combined_image);
        }
    }
#endif

    alignment_debug_images_.clear();
    result_iterations = std::vector<int>(params_alignment_.pyramid_levels, 0);

    // over levels
    Eigen::Affine3f initial_relative_pose_for_level = initial_relative_pose;
    for (int level = params_alignment_.pyramid_levels - 1; level >= 0; --level)
    {
        const ParamsCamera camera_level = getCameraForLevel(params_camera, level);
        const Eigen::Array2i & dims = camera_level.size;
        int image_channels = pyr_frame_image_list[level].size();
        if (image_channels != 1) {
            cout << "alignMultiscaleNew only supports a single image channel" << endl;
            throw std::runtime_error("alignMultiscaleNew");
        }

        // should get rid of render rects
        cv::Rect render_rect (0,0,params_camera.size.x(), params_camera.size.y());
        cv::Rect render_rect_level = getRectForLevel(render_rect, level);

        // reference or not shouldn't matter much
        ImageBuffer & frame_image = pyr_frame_image_list[level][0];
        ImageBuffer & frame_gradient_x = pyr_gradient_x[level][0];
        ImageBuffer & frame_gradient_y = pyr_gradient_y[level][0];

        ImageBuffer & frame_points = pyr_frame_points[level];
        ImageBuffer & frame_normals = pyr_frame_normals[level];
        ImageBuffer & frame_align_weights = pyr_frame_align_weights[level];

        ImageBuffer & render_image = pyr_render_image_list[level][0];
        ImageBuffer & render_points = pyr_render_points[level];
        ImageBuffer & render_normals = pyr_render_normals[level];

        // run the optimization

        Eigen::Affine3f pose_correction;

        iterateNew(frame_points, frame_normals, frame_image, frame_gradient_x, frame_gradient_y,
                render_points, render_normals, render_image,
                camera_level, camera_level,
                initial_relative_pose_for_level, params_alignment_.gn_max_iterations, level, result_iterations[level], pose_correction);

        initial_relative_pose_for_level = pose_correction * initial_relative_pose_for_level;
    }

#if 0
    {
        cout << "debug remove" << endl;
        float distance, angle;
        EigenUtilities::getAngleAndDistance(initial_relative_pose_for_level, distance, angle);
        cout << "initial_relative_pose_for_level distance: " << distance << endl;
        cout << "initial_relative_pose_for_level angle: " << angle << endl;
    }
#endif

    result_pose = initial_relative_pose_for_level * render_pose;

    return true;
}


bool Alignment::alignWithCombinedOptimization(
    ImageBuffer const& image_buffer_color,
    ImageBuffer const& image_buffer_points,
    ImageBuffer const& image_buffer_normals,
    ImageBuffer const& image_buffer_mask, // only for debug images!
    ParamsCamera const& params_camera,
    Eigen::Affine3f const& render_pose,
    Eigen::Affine3f const& initial_relative_pose, 
    Eigen::Affine3f & result_pose,
    int & iterations)
{
    const int rows = params_camera.size.y();
    const int cols = params_camera.size.x();

    /////////////
    // initialize render
    std::vector<ImageBuffer> rendered_image_channels = getImageChannelsList(image_buffer_color, cols, rows);
    ImageBuffer rendered_image_channels_list = packImageChannelsList(rendered_image_channels, cols, rows);
    const int image_channels = rendered_image_channels.size();

    // todo: stop using this
    const cv::Rect render_rect(0,0,cols,rows);

    opencl_optimize_ptr_->prepareRenderedAndErrorBuffersWithBuffers(
        params_camera.focal[0], params_camera.focal[1], params_camera.center[0], params_camera.center[1],
        render_rect.x, render_rect.y, render_rect.width, render_rect.height,
        image_buffer_points.getBuffer(), image_buffer_normals.getBuffer(), rendered_image_channels_list.getBuffer());

    ////////////////
    // run the optimization
    alignment_debug_images_.clear();

    // for debug images
    cv::Mat render_mask = cv::Mat();
    if (params_alignment_.generate_debug_images) {
        cv::Mat render_mask_int = image_buffer_mask.getMat();
        render_mask = render_mask_int > 0;
    }

    Eigen::Affine3f pose_correction;
    iterate(initial_relative_pose, params_alignment_.gn_max_iterations, render_mask, image_channels, 0, iterations, pose_correction);

    result_pose = pose_correction * initial_relative_pose * render_pose;

    return true;
}

void Alignment::iterate(Eigen::Affine3f const& initial_relative_pose, int max_iterations, cv::Mat const& render_mask, int image_channels, int level, int & iterations, Eigen::Affine3f & pose_correction)
{
    Eigen::VectorXf x_result(6);
    x_result.setZero();
    for (iterations = 0 ; iterations < max_iterations; iterations++) {
        Eigen::Matrix<float,6,6> LHS;
        Eigen::Matrix<float,6,1> RHS;

        Eigen::Affine3f x_transform(getTransformMatrixFor6DOFQuaternion(x_result));

        // newish (for loop closure keyframes): allow an initialization of the relative pose
        x_transform = x_transform * initial_relative_pose;

        std::vector<float> error_vector;
        float* error_vector_ptr = NULL;
        std::vector<float> weight_vector;
        float* weight_vector_ptr = NULL;
        if (params_alignment_.generate_debug_images) {
            error_vector.resize(opencl_optimize_ptr_->getErrorVectorSize());
            error_vector_ptr = error_vector.data();
            weight_vector.resize(opencl_optimize_ptr_->getErrorVectorSize());
            weight_vector_ptr = weight_vector.data();
        }

        // params for optimization
        const float optimize_max_distance = params_alignment_.icp_max_distance;
        const float optimize_min_normal_dot = (float) cos(params_alignment_.icp_max_normal * M_PI / 180.0f);
        const float optimize_weight_icp = std::max(params_alignment_.weight_icp, 0.f);
        const float optimize_weight_color = std::max(params_alignment_.weight_color, 0.f);

        // actually optimize
        opencl_optimize_ptr_->computeErrorAndGradient(
            optimize_max_distance, optimize_min_normal_dot,
            optimize_weight_icp, optimize_weight_color,
            params_alignment_.huber_icp, params_alignment_.huber_color,
            x_transform,
            LHS, RHS, error_vector_ptr, NULL, weight_vector_ptr);

        if (params_alignment_.regularize_lambda > 0) {
            LHS = LHS + Eigen::Matrix<float, 6, 6>::Identity() * params_alignment_.regularize_lambda;
        }

        Eigen::VectorXf x_delta = LHS.ldlt().solve(-RHS);
        x_result += x_delta;

        if (params_alignment_.generate_debug_images) {
            float scale_for_level = 1.0 * (1 << level);
            alignment_debug_images_.push_back(getDebugImage(render_mask, error_vector, weight_vector, image_channels + 1, scale_for_level));
        }

        /////////// continue?
        if (params_alignment_.gn_min_change_to_continue > 0) {
            // get the max(abs(x_delta))
            float max_component = x_delta.array().abs().maxCoeff();
            if (max_component < params_alignment_.gn_min_change_to_continue) break;
        }
    }

    pose_correction = Eigen::Affine3f(getTransformMatrixFor6DOFQuaternion(x_result));
}


void Alignment::iterateNew(
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
    Eigen::Affine3f & pose_correction)
{
    Eigen::VectorXf x_result(6);
    x_result.setZero();
    for (iterations = 0 ; iterations < max_iterations; iterations++) {
        Eigen::Matrix<float,6,6> LHS;
        Eigen::Matrix<float,6,1> RHS;

        Eigen::Affine3f x_transform(getTransformMatrixFor6DOFQuaternion(x_result));

        // newish (for loop closure keyframes): allow an initialization of the relative pose
        x_transform = x_transform * initial_relative_pose;

        // params for optimization
        const float optimize_max_distance = params_alignment_.icp_max_distance;
        const float optimize_min_normal_dot = (float) cos(params_alignment_.icp_max_normal * M_PI / 180.0f);

        // todo: use these again:
        const float optimize_weight_icp = std::max(params_alignment_.weight_icp, 0.f);
        const float optimize_weight_color = std::max(params_alignment_.weight_color, 0.f);

        bool copy_debug_images_from_gpu = params_alignment_.generate_debug_images; // always true or params_alignment_.generate_debug_images
        OpenCLOptimizeNew::OptimizeDebugImages optimize_debug_images_icp;
        OpenCLOptimizeNew::OptimizeDebugImages optimize_debug_images_image;

        opencl_optimize_new_ptr_->computeErrorAndGradientNew(
            render_buffer_points,
            render_buffer_normals,
            render_buffer_image,
            frame_buffer_points,
            frame_buffer_normals,
            frame_buffer_image,
            frame_buffer_image_gradient_x,
            frame_buffer_image_gradient_y,
            params_camera_render.focal,
            params_camera_render.center,
            params_camera_frame.focal,
            params_camera_frame.center,
            x_transform,
            optimize_max_distance,
            optimize_min_normal_dot,
            params_alignment_.huber_icp,
            params_alignment_.huber_color,
            params_alignment_.weight_icp,
            params_alignment_.weight_color,
            LHS,
            RHS,
            copy_debug_images_from_gpu,
            optimize_debug_images_icp,
            optimize_debug_images_image
            );



        if (params_alignment_.regularize_lambda > 0) {
            LHS = LHS + Eigen::Matrix<float, 6, 6>::Identity() * params_alignment_.regularize_lambda;
        }

        Eigen::VectorXf x_delta = LHS.ldlt().solve(-RHS);
        x_result += x_delta;

        if (params_alignment_.generate_debug_images) {
            float scale_for_level = 1.0 * (1 << level) * params_alignment_.debug_images_scale;
            cv::Mat debug_icp_image = getSingleDebugImage(optimize_debug_images_icp, scale_for_level);
            cv::Mat debug_image_image = getSingleDebugImage(optimize_debug_images_image, scale_for_level);
            std::vector<cv::Mat> image_v;
            image_v.push_back(debug_icp_image);
            image_v.push_back(debug_image_image);
            cv::Mat final_debug_image = createMxN(2,1,image_v);
            alignment_debug_images_.push_back(final_debug_image);
        }

        /////////// continue?
        if (params_alignment_.gn_min_change_to_continue > 0) {
            // get the max(abs(x_delta))
            float max_component = x_delta.array().abs().maxCoeff();
            if (max_component < params_alignment_.gn_min_change_to_continue) break;
        }
    }

    pose_correction = Eigen::Affine3f(getTransformMatrixFor6DOFQuaternion(x_result));
}


std::vector<ImageBuffer> Alignment::getImageChannelsList(ImageBuffer const& color_bgra_uchar, int width, int height)
{
    std::vector<ImageBuffer> result;

    if (params_alignment_.image_channels_t == ParamsAlignment::IMAGE_ERROR_YCBCR) {
        result.push_back(opencl_images_ptr_->extractYFloat(color_bgra_uchar, width, height));
        result.push_back(opencl_images_ptr_->extractCrFloat(color_bgra_uchar, width, height));
        result.push_back(opencl_images_ptr_->extractCbFloat(color_bgra_uchar, width, height));
    }
    else if (params_alignment_.image_channels_t == ParamsAlignment::IMAGE_ERROR_CBCR) {
        result.push_back(opencl_images_ptr_->extractCrFloat(color_bgra_uchar, width, height));
        result.push_back(opencl_images_ptr_->extractCbFloat(color_bgra_uchar, width, height));
    }
    else if (params_alignment_.image_channels_t == ParamsAlignment::IMAGE_ERROR_Y) {
        result.push_back(opencl_images_ptr_->extractYFloat(color_bgra_uchar, width, height));
    }
    else if (params_alignment_.image_channels_t == ParamsAlignment::IMAGE_ERROR_LAB) {
        // do on cpu like a loser
        // note this puts on GPU, then brings off, then does this stuff...really bad
        cv::Mat mat_color_bgra_uchar = color_bgra_uchar.getMat();
        cv::Mat mat_color_bgr_uchar;
        cv::cvtColor(mat_color_bgra_uchar, mat_color_bgr_uchar, CV_BGRA2BGR);
        cv::Mat mat_lab_uchar;
        cv::cvtColor(mat_color_bgr_uchar, mat_lab_uchar, CV_BGR2Lab);
        cv::Mat mat_lab_float;
        mat_lab_uchar.convertTo(mat_lab_float, CV_32F, 1./255.);
        std::vector<cv::Mat> lab_split;
        cv::split(mat_lab_float, lab_split);
        for (int i = 0; i < 3; ++i) {
            result.push_back(ImageBuffer(all_kernels_->getCL()));
            result.back().resize(height, width, 1, CV_32F);
            result.back().setMat(lab_split[i]);
        }
    }
    else if (params_alignment_.image_channels_t == ParamsAlignment::IMAGE_ERROR_NONE) {
        // nothing!	
    }
    else {
        throw std::runtime_error ("NOT IMPLEMENTED");
    }

    return result;
}

// maybe put this in OpenCLOptimize?
ImageBuffer Alignment::packImageChannelsList(std::vector<ImageBuffer> const& image_buffer_list, int width, int height)
{
    ImageBuffer result(all_kernels_->getCL());
    // stupid
    if (image_buffer_list.empty()) return result;
    const size_t image_size_single_float = image_buffer_list[0].getSizeBytes();
    result.resize(height, width, image_buffer_list.size(), CV_32F); 
    for (size_t i = 0; i < image_buffer_list.size(); ++i) {
        all_kernels_->getCL().queue.enqueueCopyBuffer(image_buffer_list[i].getBuffer(), result.getBuffer(), 0, image_size_single_float * i, image_size_single_float);
    }
    return result;
}

// maybe put this in OpenCLOptimize?
void Alignment::unpackImageChannelsList(ImageBuffer const& packed_image, int channels, int width, int height, std::vector<ImageBuffer> & result_image_buffer_list)
{
    result_image_buffer_list.clear();
    for (int i = 0; i < channels; ++i) {
        result_image_buffer_list.push_back(ImageBuffer(all_kernels_->getCL()));
        result_image_buffer_list.back().resize(height, width, 1, CV_32F);
        const size_t image_size_single_float = result_image_buffer_list.back().getSizeBytes();
        all_kernels_->getCL().queue.enqueueCopyBuffer(packed_image.getBuffer(), result_image_buffer_list.back().getBuffer(), image_size_single_float * i, 0, image_size_single_float);
    }
}

Eigen::Matrix4f Alignment::getTransformMatrixFor6DOFQuaternion(Eigen::VectorXf const& p)
{
    Eigen::Matrix4f trans = Eigen::Matrix4f::Zero ();
    trans (3,3) = 1;

    // Copy the rotation and translation components
    trans.block <4, 1> (0, 3) = Eigen::Vector4f(p[0], p[1], p[2], 1.0);

    // Compute w from the unit quaternion
    Eigen::Quaternionf q (0, p[3], p[4], p[5]);
    q.w () = sqrt (1 - q.dot (q));
    trans.topLeftCorner<3, 3> () = q.toRotationMatrix();

    return trans;
}

// perhaps should be in optimize instead?
cv::Mat Alignment::getSingleDebugImage(const OpenCLOptimizeNew::OptimizeDebugImages & optimize_debug_images, float scale)
{
    // just look at weighted error for starters
    const float error_factor = 50;
    cv::Mat weighted_error_scaled = optimize_debug_images.weighted_error * error_factor + 0.5;
    cv::Mat weighted_error_final;
    cv::resize(floatC1toCharC4(weighted_error_scaled), weighted_error_final, cv::Size(), scale, scale, cv::INTER_NEAREST);


    // convert the debug code to colors
#if 0
#define DEBUG_CODE_RENDER_INVALID 1;
#define DEBUG_CODE_PROJECT_OOB 2;
#define DEBUG_CODE_FRAME_INVALID 3;
#define DEBUG_CODE_OUTLIER_DISTANCE 4;
#define DEBUG_CODE_OUTLIER_ANGLE 5;
#define DEBUG_CODE_SUCCESS 6;
#endif
    cv::Mat debug_code = optimize_debug_images.debug_code;
    cv::Mat debug_code_colored(debug_code.size(), CV_8UC4, cv::Scalar::all(0));
    cv::MatConstIterator_<int> iter_in = debug_code.begin<int>();
    cv::MatIterator_<cv::Vec4b> iter_out = debug_code_colored.begin<cv::Vec4b>();
    for ( ; iter_in != debug_code.end<int>(); ++iter_in, ++iter_out) {
        const int & code = *iter_in;
        if (code == 1) *iter_out = cv::Vec4b(0,0,255,255);
        else if (code == 2) *iter_out = cv::Vec4b(0,0,255,255);
        else if (code == 3) *iter_out = cv::Vec4b(0,0,255,255);
        else if (code == 4) *iter_out = cv::Vec4b(255,0,0,255);
        else if (code == 5) *iter_out = cv::Vec4b(255,0,255,255);
        else if (code == 6) *iter_out = cv::Vec4b(255,255,255,255);
    }
    cv::Mat debug_code_scaled;
    cv::resize(debug_code_colored, debug_code_scaled, cv::Size(), scale, scale, cv::INTER_NEAREST);

    // grab the distance weights and huber weights meaningfully
    cv::Mat distance_weight = optimize_debug_images.weights_distance;
    cv::Mat distance_weight_final;
    cv::resize(floatC1toCharC4(distance_weight), distance_weight_final, cv::Size(), scale, scale, cv::INTER_NEAREST);

    cv::Mat huber_weight = optimize_debug_images.weights_huber;
    cv::Mat huber_weight_final;
    cv::resize(floatC1toCharC4(huber_weight), huber_weight_final, cv::Size(), scale, scale, cv::INTER_NEAREST);


    // combined into single image
    std::vector<cv::Mat> image_v;
    image_v.push_back(debug_code_scaled);
    image_v.push_back(distance_weight_final);
    image_v.push_back(huber_weight_final);
    image_v.push_back(weighted_error_final);


    cv::Mat result = createMxN(1,4,image_v); // ALWAYS CHECK TYPES (until you fix the fucking function)

    return result;
}

cv::Mat Alignment::getDebugImage(cv::Mat const& render_mask, std::vector<float> const& error_vector, std::vector<float> const& weight_vector, int error_channel_count, float scale)
{
    const int rows = render_mask.rows;
    const int cols = render_mask.cols;
    int error_points = rows * cols;

    // init "top row" error images to 0.5
    std::vector<cv::Mat> image_error_vec;
    for (int i = 0; i < error_channel_count; i++) {
        image_error_vec.push_back(cv::Mat(rows, cols, CV_32FC1, cv::Scalar::all(0.5)));
    }

    // fiddle with the "top row"
    std::vector<float> sse_vec(error_channel_count, 0);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // new coallesced indexing (TODO: JUST USE OPENCV)
            // this lets me grab the actual error, though!
            // but what you should do is start with actual error, then later add 0.5
            float sse = 0;
            for (int i = 0; i < error_channel_count; i++) {
                int error_vector_index_image = i * error_points + row * cols + col;
                float error_value = error_vector[error_vector_index_image];
                sse_vec[i] += error_value * error_value;
                image_error_vec[i].at<float>(row, col) += error_value;
            }
        }
    }

    // bottom row is weights errors now....
    std::vector<cv::Mat> image_weights_vec;
    for (int i = 0; i < error_channel_count; i++) {
        image_weights_vec.push_back(cv::Mat(rows, cols, CV_32FC1, cv::Scalar::all(0)));
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // new coallesced indexing (TODO: JUST USE OPENCV)
            for (int i = 0; i < error_channel_count; i++) {
                int index = i * error_points + row * cols + col;
                float weight = weight_vector[index];
                image_weights_vec[i].at<float>(row, col) = weight;
            }
        }
    }

    // we can now print the total error per image
    // TODO: COMPUTE VARIANCE/STD OF ERROR!!!
    // todo: not print here?
#if 0
    for (size_t i = 0; i < image_error_vec.size(); ++i) {
        //float total_squared_error = cv::sum(image_error_vec[i].mul(image_error_vec[i]))[0];
        cout << "Sum Squared Error " << i << " : " << sse_vec[i] << endl;
        // also "variance"
        float variance = sse_vec[i] / render_rect.area();
        cout << "Variance " << i << " : " << variance << endl;
        cout << "STD " << i << " : " << sqrt(variance) << endl;
    }
#endif

    // row 1
    std::vector<cv::Mat> v_images;
    for (int i = 0; i < error_channel_count; i++) {
        cv::Mat error_image_bgr = floatC1toCharC3(image_error_vec[i]);
        cv::Mat error_image_bgr_masked (rows, cols, CV_8UC3, cv::Scalar(0,0,255)); // red outliers?
        error_image_bgr.copyTo(error_image_bgr_masked, render_mask);

        // also outliers (those errors which are exactly 0.5)
        cv::Mat outlier_mask = render_mask & image_error_vec[i] == 0.5;
        error_image_bgr_masked.setTo(cv::Scalar(255,0,0), outlier_mask);

        v_images.push_back(error_image_bgr_masked);
    }
    // row 2
    for (int i = 0; i < error_channel_count; i++) {
        cv::Mat weight_image_bgr = floatC1toCharC3(image_weights_vec[i]);
        v_images.push_back(weight_image_bgr);
    }
    cv::Mat combined_error_images = createMxN(2, error_channel_count, v_images);
    float scale_resize = params_alignment_.debug_images_scale * scale;
    cv::Mat combined_error_images_scaled;
    cv::resize(combined_error_images, combined_error_images_scaled, cv::Size(), scale_resize, scale_resize, cv::INTER_NEAREST); 

    return combined_error_images_scaled;
}

std::vector<cv::Mat> Alignment::getPyramidImages(std::vector<std::vector<ImageBuffer> > const& pyr_image_list)
{
    std::vector<cv::Mat> result;

    BOOST_FOREACH(std::vector<ImageBuffer> const& v, pyr_image_list) {
        //stuff
        std::vector<cv::Mat> this_level;
        BOOST_FOREACH(ImageBuffer const& ib, v) {
            this_level.push_back(ib.getMat());
        }
        result.push_back(createMxN(1, this_level.size(), this_level));
    }

    return result;
}

std::vector<cv::Mat> Alignment::getPyramidSingle(std::vector<ImageBuffer> const& pyr_images)
{
    std::vector<cv::Mat> result;
    BOOST_FOREACH(ImageBuffer const& ib, pyr_images) {
        result.push_back(ib.getMat());
    }
    return result;
}

void Alignment::setAlignDebugImages(bool value)
{
    params_alignment_.generate_debug_images = value;
}

void Alignment::getAlignDebugImages(std::vector<cv::Mat> & image_list)
{
    image_list = alignment_debug_images_;
}

void Alignment::getPyramidDebugImages(std::vector<cv::Mat> & image_list)
{
    image_list = pyramid_debug_images_;
}
