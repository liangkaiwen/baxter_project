#pragma once

#include <string>
#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


struct Parameters {
	// overall
	std::string full_command_line;

	bool disable_cloud_viewer;
	bool disable_cout_buffer;

	bool save_input;
	bool save_input_only;
	float save_input_only_max_fps;
	bool save_input_images;
	bool save_objects_pcd;
	bool save_objects_png;
	bool save_tables_of_values;
	bool save_poses;
	bool save_cdi_images;
	bool save_viewer_screenshots;
	bool save_render_for_alignment;
	bool save_segment_debug_images;
	bool render_after;
	bool save_render_after;

	float max_depth_in_input_cloud;
	bool correct_input_camera_params;
	int process_frame_waitkey_delay;
	bool set_inactive_on_failure;
	bool show_axes;
	bool script_mode;
	bool white_background;

	// input options
	bool live_input_openni;

	// folder options
	fs::path folder_input;
	fs::path folder_output;
	int folder_frame_start;
	int folder_frame_increment;
	int folder_frame_end;
	bool folder_debug_keep_same_frame;
	bool folder_debug_pause_after_every_frame;
	bool folder_debug_pause_after_failure;
	bool folder_do_calibrate;
	bool folder_is_xyz_files;
	bool folder_is_rgbd_turntable_files;
	bool folder_is_oni_png_files;
	bool folder_is_raw_files;
	bool folder_is_freiburg_files;
	bool freiburg_associate_depth_first;
	bool folder_is_evan_files;
	fs::path file_input_object_poses;
	fs::path file_input_loop_poses;
	fs::path file_input_loop_which_pvs;
	bool load_object_poses;
	bool debug_do_load_pv_info;
	
	// opencl
	bool opencl_nvidia; // otherwise intel
	bool opencl_debug;
	bool opencl_context_cpu;
	bool opencl_context_gpu;
	fs::path opencl_cl_path;
	bool opencl_fast_math;

	// camera
	float camera_focal_x;
	float camera_focal_y;
	float camera_center_x;
	float camera_center_y;
	int camera_size_x;
	int camera_size_y;
	float camera_stereo_baseline;
	float camera_z_min;
	float camera_z_max;

	// render (for user initiated render)
	float render_min_depth;
	float render_max_depth;

	// which optimization
	bool use_features;
	bool use_combined_optimization;

	// how to init
	float initial_centroid_offset_x;
	float initial_centroid_offset_y;
	float initial_centroid_offset_z;
	bool initial_centroid_image_center;
	bool initial_centroid_fixed;
	bool initialize_with_previous_relative_transform;

	float max_depth_sigmas; // new depth-aware distance

	// feature params
	bool features_debug_images;
	bool features_frame_to_frame;

	// ransac params
	int ransac_min_inliers;
	float ransac_pixel_distance; // L2 norm, includes disparity
	float ransac_probability;
	int ransac_verbosity;
	int ransac_max_iterations;

	// icp params
	float icp_max_distance;
	float icp_normals_angle;
	float icp_min_translation_to_continue;
	float icp_min_rotation_to_continue;
	int icp_max_iterations;

	// normals frame
	int normals_smooth_iterations;
	bool normals_opencl_debug;

	// tsdf params
	int volume_cell_count_x;
	int volume_cell_count_y;
	int volume_cell_count_z;
	float volume_cell_size;
	float volume_max_weight;
	bool volume_debug_show_max_points;
	bool volume_debug_sphere;
	float volume_debug_sphere_radius;

	// combined optimization params
	float combined_weight_icp_points;
	float combined_weight_color;
	bool combined_pause_every_eval;
	bool combined_verbose;
	bool combined_debug_images;
	float combined_debug_images_scale;
	bool combined_gauss_newton;
	bool combined_gauss_newton_gpu_full;
	int combined_gauss_newton_max_iterations;
	float combined_gauss_newton_min_delta_to_continue;
	float combined_render_scale;
	int combined_octaves;
	enum ImageErrorT {IMAGE_ERROR_NONE, IMAGE_ERROR_Y, IMAGE_ERROR_CBCR, IMAGE_ERROR_YCBCR, IMAGE_ERROR_LAB};
	ImageErrorT combined_image_error;
	bool combined_debug_pause_after_icp_all;
	bool combined_debug_pause_after_icp_iteration;
	bool combined_debug_normal_eq;
	bool combined_debug_single_kernel;
	bool combined_compute_error_statistics;
	int combined_min_rendered_point_count;
	bool combined_show_render;

	// how to detect errors in optimization
	float error_max_t;
	float error_max_r;
	float error_change;
	float error_min_inlier_fraction_icp;
	float error_min_inlier_fraction_color;
	int error_min_rendered_points;
	int error_min_output_frames; // how many output frames required to check the others
	float error_min_eigenvalue;
	float error_rank_threshold;
	bool error_use_rank;

	// object masking
	bool mask_object;
	bool mask_hand;
	bool mask_input_if_present;
	bool mask_debug_images;
	bool mask_debug_every_component;
	int mask_object_min_size;
	float mask_initial_search_contraction_factor;
	float mask_search_contraction_factor;
	float mask_connected_max_depth_difference;
	float mask_disconnected_max_depth_difference;
	float mask_global_max_depth_difference;
	int mask_object_erode;
	bool mask_object_use_only_first_segment;
	bool mask_always_reset_search_region;
	bool mask_floodfill;
	float mask_floodfill_expand_diff;
	
	fs::path mask_hand_hist_filename;
	fs::path mask_object_hist_filename;
	fs::path mask_default_hand_hist_filename;
	fs::path mask_default_object_hist_filename;
	// new masking (learn)
	bool mask_hand_hist_learn;
	bool mask_object_hist_learn;
	int mask_hand_learn_hbins;
	int mask_hand_learn_sbins;
	int mask_hand_learn_vbins;
	float mask_hand_backproject_thresh;
	// hand we erode, then dilate, to get rid of spurious pixels
	int mask_hand_erode;
	int mask_hand_dilate;
	int mask_hand_min_component_area_before_morphology;
	int mask_hand_min_component_area_after_morphology;

	// color params
	int color_blur_size;
	bool color_blur_after_pyramid;

	// other stuff
	float display_min_z;
	float display_max_z;
	float render_viewer_scale;

	// mesh
	bool mesh_show;
	bool mesh_marching_cubes_weights;

	// segments
	float segments_max_angle;
	int segments_min_size;
	bool segment_debug_images;

	// patch volumes
	bool use_patch_volumes;
	float pv_initial_border_size;
	float pv_expand_border_size;
	bool pv_loop_closure;
	float pv_loop_min_frame_coverage;
	float pv_loop_max_normal_angle;
	bool pv_loop_debug_images;
	int pv_pose_graph_iterations;
	bool pv_pose_graph_verbose;
	int pv_max_age_before_considered_loop;
	int pv_age_to_deallocate;
	int pv_min_size_to_create_new;
	int pv_loop_icp_max_iterations;
	bool pv_show_volume_edges;
	bool pv_show_volume_mesh;
	bool pv_show_volume_normals;
	bool pv_show_graph_edges;
	float pv_mesh_alpha;
	float pv_edge_loop_information_factor;
	bool pv_debug_optimize_always;
	bool pv_debug_show_render_for_alignment;
	bool pv_verbose;
	int pv_max_side_voxel_count;
	int pv_split_voxel_overlap;
	int pv_loop_frame_skip;
	bool pv_loop_features;
	bool pv_debug_update_visualizer;
	float pv_keyframe_max_distance_create;
	float pv_keyframe_max_angle_create;
	float pv_keyframe_max_distance_match;
	float pv_keyframe_max_angle_match;
	bool pv_keyframe_debug;
	bool pv_debug_add_volumes;
	int pv_max_mb_allocated;
	bool pv_color_all_boxes;
	bool pv_use_covariance_to_create;
	bool pv_debug_print_edges;

	// patch grid?
	bool use_patch_grid;
	int pg_size;
	int pg_border;

	// Defaults defined here:
	Parameters() :
		full_command_line(""),
		disable_cloud_viewer(false),
		disable_cout_buffer(false),
		save_input(false),
		save_input_only(false),
		save_input_only_max_fps(-1),
		save_input_images(false),
		save_objects_pcd(false),
		save_objects_png(false),
		save_viewer_screenshots(false),
		save_render_for_alignment(false),
		save_segment_debug_images(false),
		render_after(false),
		save_render_after(false),
		max_depth_in_input_cloud(-1),
		correct_input_camera_params(false),
		process_frame_waitkey_delay(1),
		save_tables_of_values(false),
		set_inactive_on_failure(false),
		save_poses(false),
		save_cdi_images(false),
		show_axes(false),
		script_mode(false),
		white_background(false),
		live_input_openni(false),
		folder_input(),
		folder_output(),
		folder_frame_start(0),
		folder_frame_increment(1),
		folder_frame_end(-1),
		folder_debug_keep_same_frame(false),
		folder_debug_pause_after_every_frame(false),
		folder_debug_pause_after_failure(false),
		folder_do_calibrate(false),
		folder_is_xyz_files(false),
		folder_is_rgbd_turntable_files(false),
		folder_is_oni_png_files(false),
		folder_is_raw_files(false),
		folder_is_freiburg_files(false),
		freiburg_associate_depth_first(false),
		folder_is_evan_files(false),
		file_input_object_poses("object_poses.txt"),
		file_input_loop_poses("loop_poses.txt"),
		file_input_loop_which_pvs("loop_which_pvs.txt"),
		load_object_poses(false),
		debug_do_load_pv_info(false),

		opencl_nvidia(false),
		opencl_debug(false),
		opencl_context_cpu(false),
		opencl_context_gpu(false),
		opencl_cl_path(""),
		opencl_fast_math(false),

		camera_size_x(640),
		camera_size_y(480),
		camera_focal_x(525.0),
		camera_focal_y(525.0),
		camera_center_x( (camera_size_x - 1) / 2.),
		camera_center_y( (camera_size_y - 1) / 2.),
		camera_stereo_baseline(0.075),
		camera_z_min(0.4),
		camera_z_max(5), // could probably also use the other "max"

		render_min_depth(0.01), // gotta have something to make the frustum
		render_max_depth(30),

		use_features(false),
		use_combined_optimization(false),
		initialize_with_previous_relative_transform(true), // no param (true ok)
		initial_centroid_offset_x(0),
		initial_centroid_offset_y(0),
		initial_centroid_offset_z(0),
		initial_centroid_image_center(false),
		initial_centroid_fixed(false),

		max_depth_sigmas(3),

		features_debug_images(false),
		features_frame_to_frame(false),

		ransac_pixel_distance(3.0),
		ransac_min_inliers(10),
		ransac_probability(0.99),
		ransac_verbosity(3),
		ransac_max_iterations(2000),

		icp_max_distance(0.05),
		icp_normals_angle(45),
		icp_min_translation_to_continue(1e-4),
		icp_min_rotation_to_continue(1e-4),
		icp_max_iterations(1),

		normals_smooth_iterations(2),
		normals_opencl_debug(false),

		volume_cell_count_x(256),
		volume_cell_count_y(256),
		volume_cell_count_z(256),
		volume_cell_size(0.01),
		volume_max_weight(100),
		volume_debug_show_max_points(false),
		volume_debug_sphere(false),
		volume_debug_sphere_radius(1.0),

		// object mask
		mask_object(false),
		mask_hand(false),
		mask_input_if_present(false),
		mask_debug_images(false),
		mask_debug_every_component(false),
		mask_object_min_size(10),
		mask_initial_search_contraction_factor(0.25),
		mask_search_contraction_factor(0.5),
		mask_connected_max_depth_difference(0.02),
		mask_disconnected_max_depth_difference(0.02),
		mask_global_max_depth_difference(0.20),
		mask_object_erode(2),
		mask_object_use_only_first_segment(false),
		mask_always_reset_search_region(false),
		mask_hand_hist_filename(),
		mask_object_hist_filename(),
		mask_default_hand_hist_filename("hand.yml"),
		mask_default_object_hist_filename("object.yml"),
		mask_hand_hist_learn(false),
		mask_object_hist_learn(false),
		mask_hand_learn_hbins(20),
		mask_hand_learn_sbins(10),
		mask_hand_learn_vbins(10),
		mask_hand_backproject_thresh(0.2),
		mask_hand_erode(0),
		mask_hand_dilate(2),
		mask_hand_min_component_area_before_morphology(50),
		mask_hand_min_component_area_after_morphology(0),
		mask_floodfill(false),
		mask_floodfill_expand_diff(10),

		color_blur_size(5),
		color_blur_after_pyramid(false),
		combined_weight_icp_points(10),
		combined_weight_color(1),
		combined_pause_every_eval(false),
		combined_verbose(false),
		combined_debug_images(false),
		combined_debug_images_scale(1.0),
		combined_gauss_newton(false),
		combined_gauss_newton_gpu_full(true), // force true
		combined_gauss_newton_max_iterations(50),
		combined_gauss_newton_min_delta_to_continue(0.0001),
		combined_render_scale(1.0),
		combined_octaves(1),
		combined_image_error(IMAGE_ERROR_Y),
		combined_debug_pause_after_icp_all(false),
		combined_debug_pause_after_icp_iteration(false),
		combined_debug_normal_eq(false),
		combined_debug_single_kernel(false),
		combined_compute_error_statistics(false),
		combined_min_rendered_point_count(100), // note this...also note loop closure uses a different param
		combined_show_render(false),

		error_max_t(-1),
		error_max_r(-1),
		error_change(-1),
		error_min_inlier_fraction_icp(-1),
		error_min_inlier_fraction_color(-1),
		error_min_rendered_points(0),
		error_min_output_frames(2),
		error_min_eigenvalue(-1),
		error_rank_threshold(-1),
		error_use_rank(false),

		display_min_z(0.4),
		display_max_z(3.0),
		render_viewer_scale(1),

		mesh_show(false),
		mesh_marching_cubes_weights(false),

		segments_max_angle(30),
		segments_min_size(1000), // check this when doing objects
		segment_debug_images(false),

		use_patch_volumes(false),
		pv_initial_border_size(0.01),
		pv_expand_border_size(0.05),
		pv_loop_closure(false),
		pv_loop_min_frame_coverage(0.5),
		pv_loop_max_normal_angle(45), // if > 0, max angle between patch normal and direction to camera
		pv_loop_debug_images(false),
		pv_pose_graph_iterations(5),
		pv_pose_graph_verbose(true),
		pv_max_age_before_considered_loop(-10), // negative should turn off (less than -1 because I add 1 to it one place...lol)
		pv_age_to_deallocate(10),
		pv_min_size_to_create_new(1000),
		pv_loop_icp_max_iterations(1),
		pv_show_volume_edges(false),
		pv_show_volume_mesh(false),
		pv_show_volume_normals(false),
		pv_show_graph_edges(false),
		pv_mesh_alpha(0.5),
		pv_edge_loop_information_factor(1),
		pv_debug_optimize_always(false),
		pv_debug_show_render_for_alignment(false),
		pv_verbose(false),
		pv_max_side_voxel_count(-1),
		pv_split_voxel_overlap(0),
		pv_loop_frame_skip(0),
		pv_loop_features(false),
		pv_debug_update_visualizer(false),
		pv_keyframe_max_distance_create(0.5),
		pv_keyframe_max_angle_create(30),
		pv_keyframe_max_distance_match(1.5),
		pv_keyframe_max_angle_match(60),
		pv_keyframe_debug(false),
		pv_debug_add_volumes(false),
		pv_max_mb_allocated(-1),
		pv_color_all_boxes(false),
		pv_use_covariance_to_create(false),
		use_patch_grid(false),
		pg_size(32),
		pg_border(0),
		pv_debug_print_edges(false)
		{}
};