#pragma once

struct ParamsAlignment
{
	float icp_max_distance;
	float icp_max_normal;
	float weight_icp;
	float weight_color;
    float huber_icp; // switch to float (positive turns on)
    float huber_color; // switch to float (positive turns on)

	enum ImageChannelsT {IMAGE_ERROR_NONE, IMAGE_ERROR_Y, IMAGE_ERROR_CBCR, IMAGE_ERROR_YCBCR, IMAGE_ERROR_LAB} image_channels_t;
	int color_blur_size;

	float gn_min_change_to_continue;
	int gn_max_iterations;

	bool generate_debug_images;
	float debug_images_scale;

	float regularize_lambda;

	bool use_multiscale;
	int pyramid_levels; // takes effect if multiscale

	bool use_new_alignment;

	ParamsAlignment()
		: icp_max_distance(0.1f),
		icp_max_normal(45),
		weight_icp(10),
		weight_color(1),
        huber_icp(0.1),
        huber_color(0.1),
		image_channels_t(IMAGE_ERROR_Y),
		color_blur_size(5),
		gn_min_change_to_continue(0.0001f),
		gn_max_iterations(10),
		generate_debug_images(false),
		debug_images_scale(1),
		regularize_lambda(0),
		use_multiscale(true),
		pyramid_levels(4),
		use_new_alignment(false)
	{}
};
