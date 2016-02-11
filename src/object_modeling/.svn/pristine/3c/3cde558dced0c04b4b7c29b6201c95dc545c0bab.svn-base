#pragma once

struct ParamsPatch
{
	float segments_max_depth_sigmas;
	float segments_max_angle;
	int segments_min_size;
	float border_create; // in meters
	float border_expand; // in meters
	bool segments_create_of_all_sizes;
	bool debug_patch_creation;
	bool test_patch_reorientation;
	bool test_patch_reorientation_glfw;

	ParamsPatch()
		: segments_max_depth_sigmas(3),
		segments_max_angle(30),
		segments_min_size(1000), // was 1000 for mapping patch volumes...probably want smaller for objects
		border_create(0.05f),
		border_expand(0.05f),
		segments_create_of_all_sizes(false),
		debug_patch_creation(false),
		test_patch_reorientation(false),
		test_patch_reorientation_glfw(false)
	{}
};
