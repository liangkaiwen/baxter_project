#pragma once

struct ParamsMovingVolumeGrid
{
	float camera_center_distance; // meters
	int blocks_in_moving_volume;
	bool debug_no_moving_mesh;
	bool debug_disable_merge_on_shift;
	bool debug_fix_g2o_vertices_for_keyframes;
	bool debug_clipping;
    bool debug_delete_edges_for_merged_volumes;

	ParamsMovingVolumeGrid()
		: camera_center_distance(2.f),
		blocks_in_moving_volume(10),
		debug_no_moving_mesh(false),
		debug_disable_merge_on_shift(false),
		debug_fix_g2o_vertices_for_keyframes(false),
        debug_clipping(false),
        debug_delete_edges_for_merged_volumes(false)
	{}
};
