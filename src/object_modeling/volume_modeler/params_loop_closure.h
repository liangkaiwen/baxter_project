#pragma once

enum ActivationMode {
    ACTIVATION_MODE_ALL,
	ACTIVATION_MODE_AGE,
	ACTIVATION_MODE_FULL_GRAPH,
	ACTIVATION_MODE_KEYFRAME_GRAPH
};

struct ParamsLoopClosure
{
	bool loop_closure;
	float min_fraction;
	int optimize_iterations;
	float loop_closure_edge_strength;
	
	float keyframe_distance_create;
	float keyframe_angle_create;
	float keyframe_distance_match;
	float keyframe_angle_match;
	int keyframe_graph_distance;

	ActivationMode activation_mode;
	int activate_age;
	int activate_full_graph_depth;
	
	bool debug_optimize;

	bool debug_save_meshes;
	bool debug_disable_merge;
	bool debug_merge_show_points;

    bool use_dbow_place_recognition;

    // don't actually use this...just to hack in something that works
    int debug_min_keyframe_index_difference;
    int debug_max_total_loop_closures;

	ParamsLoopClosure()
		: loop_closure(false),
		min_fraction(0.5f),
		optimize_iterations(10),
		loop_closure_edge_strength(100),
		keyframe_distance_create(0.25f),
		keyframe_angle_create(20),
		keyframe_distance_match(1.5f),
		keyframe_angle_match(60),
		keyframe_graph_distance(5),
        activation_mode(ACTIVATION_MODE_ALL),
		activate_age(50),
		activate_full_graph_depth(3),
		debug_optimize(false),
		debug_save_meshes(false),
		debug_disable_merge(false),
        debug_merge_show_points(false),
        use_dbow_place_recognition(false),
        debug_min_keyframe_index_difference(200),
        debug_max_total_loop_closures(-1)
	{}
};
