#pragma once

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

struct ParamsMaskObject
{
	bool mask_object;

	fs::path histogram_hand_file;
	fs::path histogram_object_file;
	int bins_h;
	int bins_s;
	int bins_v;

	float initial_seed_rect_scale;
	int min_object_rect_side; // pixels
	int object_mask_erode_iterations;
	float max_connected_component_depth_difference;
	float max_disconnected_component_depth_difference;

	bool mask_hand;
	int mask_hand_erode_iterations;
	int mask_hand_dilate_iterations;
	int mask_hand_min_component_area_before_morphology; 
	int mask_hand_min_component_area_after_morphology;
	float mask_hand_backproject_threshold;
	bool mask_hand_floodfill;
	float mask_floodfill_expand_diff;

	ParamsMaskObject()
		:
		mask_object(false),
		bins_h(20),
		bins_s(10),
		bins_v(10),
		initial_seed_rect_scale(0.5f),
		min_object_rect_side(10),
		object_mask_erode_iterations(2),
		max_connected_component_depth_difference(0.02f), 
		max_disconnected_component_depth_difference(0.02f),
		mask_hand(false),
		mask_hand_erode_iterations(0),
		mask_hand_dilate_iterations(2),
		mask_hand_min_component_area_before_morphology(50),
		mask_hand_min_component_area_after_morphology(0),
		mask_hand_backproject_threshold(0.2f),
		mask_hand_floodfill(false),
		mask_floodfill_expand_diff(10.0f)
	{}
};