#pragma once

#include <string>

enum ModelType {
	MODEL_NONE,
	MODEL_ERROR,
	MODEL_SINGLE_VOLUME,
	MODEL_GRID,
    MODEL_PATCH,
    MODEL_MOVING_VOLUME_GRID,
    MODEL_HISTOGRAM,
	MODEL_K_MEANS
};

struct ParamsVolumeModeler
{
	ModelType model_type;
	ModelType model_type_1; // alternate
	ModelType model_type_2; // alternate (assumes you used 1 already)
	bool use_features;
	bool verbose;
	bool first_frame_centroid; // only applies to single volume case, first frame is aligned to centroid of initial frame
    bool first_frame_origin; // center volume on origin (good for generated data)
	bool command_line_interface; // no images shown
	bool set_color_weights; // set color weights based on distance from depth edges
	bool set_color_weights_debug_images;
	float max_edge_sigmas;
	float max_distance_transform;
	std::string full_command_line;
	bool debug_show_nonzero_voxels;
	bool update_interface_view_pose;
	fs::path output;
	float scale_images; // general scaling of debug images...use when you like
    bool apply_vignette_model;

	ParamsVolumeModeler()
		: model_type(MODEL_SINGLE_VOLUME),
		model_type_1(MODEL_NONE),
		model_type_2(MODEL_NONE),
		use_features(false),
		verbose(false),
		first_frame_centroid(false),
        first_frame_origin(false),
		command_line_interface(false),
		set_color_weights(true),            // NOTE: CHANGED DEFAULT TO TRUE BECAUSE IT'S A GOOD IDEA
		set_color_weights_debug_images(false),
		max_edge_sigmas(10),
		max_distance_transform(5),
		full_command_line(),
		debug_show_nonzero_voxels(false),
		update_interface_view_pose(false),
		output(),
        scale_images(-1),
        apply_vignette_model(false)
	{}
};
