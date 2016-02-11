#include <stdio.h>

#include <iostream>
using std::cout;
using std::endl;

#include <boost/format.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

// opencl lib
#include "cll.h"
#include "KernelVignetteApplyModelPolynomial3Uchar4.h"
#include "KernelDepthImageToPoints.h"
#include "KernelTransformPoints.h"
#include "util.h" // poorly named, obviously

// opencv utilities
#include "opencv_utilities.h"

// volume modeler
#include "basic.h"
#include "frame.h"
#include "volume_modeler.h"
#include "vignette_calibration.h"

// frame provider
#include "frame_provider.h"
#include "pose_provider_standard.h"

#ifdef FRAME_PROVIDER_OPENNI2
#include "frame_provider_openni2.h"
#endif

#ifdef FRAME_PROVIDER_PCL
#include "frame_provider_pcd.h"
#endif

#ifdef VOLUME_MODELER_GLFW
#include "volume_modeler_glfw.h"
#endif


//////////////////////////////

int main(int argc, char* argv[])
{
    // can enable this if IO is really a problem...
#if 0
    // don't use printf/scanf (and expect it to be in sync)
    std::ios_base::sync_with_stdio(false);
#endif

#ifdef _WIN32
#if 1
    // windows: buffer stdout better!
    const int console_buffer_size = 4096;
    char buf[console_buffer_size];
    setvbuf(stdout, buf, _IOLBF, console_buffer_size);
#else
    cout << "WARNING: SLOW COUT" << endl;
#endif
#endif


#ifdef _WIN32
    const fs::path cl_path_default = "C:\\devlibs\\object_modeling\\OpenCLStaticLib";
#else
    const fs::path cl_path_default = "/home/peter/checkout/object_modeling/OpenCLStaticLib";
#endif


    const static std::string window_align_debug_images = "align_debug_images";
    const static std::string window_loop_closure_debug_images = "loop_closure_debug_images";
    const static std::string window_pyramid_debug_images = "pyramid_debug_images";

    // contains all params that VolumeModeler needs
    VolumeModelerAllParams params;

    // grab the whole command line to jog my memory later
    for (int i = 0; i < argc; ++i) {
        params.volume_modeler.full_command_line += std::string(argv[i]) + " ";
    }

    /////////////
    // boost program options\

    // basically assumes you have openni2 (recent change)
    FrameProviderOpenni2Params openni2_params;

    std::string model_type_string;
    std::string model_type_1_string;
    std::string model_type_2_string;
    fs::path input_png;
    fs::path input_png_depth;
    fs::path input_pcd;
    fs::path input_oni;
    fs::path input_handa;
    fs::path input_yuyin;
    fs::path input_freiburg;
    fs::path input_luis;
    fs::path input_arun;
    std::string yuyin_prefix = "NP1_";
    fs::path load_state;
    fs::path load_camera_poses;
    bool expect_evan_poses = false; // means expect poses, and that depth_factor is 1000.  Ha.  You could make a separate depth_factor param if it comes up
    bool expect_evan_camera_list = false; // means instead of one pose per frame, use camera_list.txt in folder.  Sigh.
    bool expect_luis_camera_list = false;
    bool empty = false;
    float max_depth = -1;
    int frame_start = -1;
    int frame_end = -1;
    int frame_increment = 1;
    bool pause_before_first_frame = false;
    bool pause_after_first_frame = false;
    bool pause_every_frame = false;
    bool pause_after_loop_closure = false;
    float test_set_weights = -1;
    int glfw_mesh_every_n = -1;
    bool align_debug_images = false;
    bool loop_closure_debug_images = false;
    bool loop_closure_save_meshes_always = false;
    bool cl_intel_cpu = false; // use intel opencl cpu instead of default nvidia gpu
    bool cl_amd_cpu = false; // use amd opencl cpu instead of default nvidia gpu
    bool test_align_weights = false;
    bool test_add_depth_weights = false;
    bool test_single_pose = false;
    bool test_set_value_in_sphere = false;
    bool test_set_value_in_box = false;
    bool glfw_join = false;
    bool glfw_join_full_mesh = false;
    float glfw_background_intensity = 0.5;
    int save_mesh_every_n = -1;
    int save_state_every_n = -1;
    bool save_state_at_end = false;
    bool save_input = false;
    bool save_images = false;
    bool save_masked_input = false;
    int circle_mask_pixel_radius = -1;
    int circle_mask_x = -1;
    int circle_mask_y = -1;
    int rect_mask_x = 0;
    int rect_mask_y = 0;
    int rect_mask_width = 0;
    int rect_mask_height = 0;
    bool suppress_debug_images = false;
    bool low_resolution = false;
    bool read_glfw_buffer = false;
    bool fixed_top_down_view = false;
    float fixed_top_down_view_height = 8.0;
    std::vector<float> fixed_view_pos_at_up;
    int glfw_width = 800;
    int glfw_height = 600;
    bool learn_histogram_hand = false;
    bool learn_histogram_object = false;
    bool debug_add_volume = false;
    bool test_nvidia_gpu_memory = false;
    bool loop_input = false;
    std::string activation_mode_string;
    float model_histogram_abs = -1;
    bool pick_pixel = false;
    fs::path cl_path;
    int extra_skip_hack = -1;
    int vxyz = -1;
    bool use_six_volumes = false;
    bool render_after_loaded_camera = false;
    bool use_f200_camera = false;
    int turntable_rotation_limit = -1;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("cl_path", po::value<fs::path>(&cl_path), "cl_path")
        ("load_state", po::value<fs::path>(&load_state), "load_state")
        ("load_camera_poses", po::value<fs::path>(&load_camera_poses), "load_camera_poses")
        ("render_after_loaded_camera", po::value<bool>(&render_after_loaded_camera)->zero_tokens(), "render_after_loaded_camera")
        ("empty", po::value<bool>(&empty)->zero_tokens(), "empty")
        ("png", po::value<fs::path>(&input_png), "input: folder with png files (see code for format)")
        ("png_depth", po::value<fs::path>(&input_png_depth), "input: folder with png depth files (probably generated)")
        ("pcd", po::value<fs::path>(&input_pcd), "input: folder with pcd files")
        ("oni", po::value<fs::path>(&input_oni), "input: oni file")
        ("handa", po::value<fs::path>(&input_handa), "input: handa folder")
        ("yuyin", po::value<fs::path>(&input_yuyin), "input: input_yuyin folder")
        ("yuyin_prefix", po::value<std::string>(&yuyin_prefix), "yuyin_prefix")
        ("freiburg", po::value<fs::path>(&input_freiburg), "input: freiburg associate.py file")

        ("luis", po::value<fs::path>(&input_luis), "input: luis folder")
        ("arun", po::value<fs::path>(&input_arun), "input: arun folder")
        ("use_f200_camera", po::value<bool>(&use_f200_camera)->zero_tokens(), "use_f200_camera")
        ("turntable_rotation_limit", po::value<int>(&turntable_rotation_limit), "turntable_rotation_limit")
        

        ("expect_evan_poses", po::value<bool>(&expect_evan_poses)->zero_tokens(), "expect_evan_poses")
        ("expect_evan_camera_list", po::value<bool>(&expect_evan_camera_list)->zero_tokens(), "expect_evan_camera_list")
        ("expect_luis_camera_list", po::value<bool>(&expect_luis_camera_list)->zero_tokens(), "expect_luis_camera_list")

        ("output", po::value<fs::path>(&params.volume_modeler.output), "output")

        ("f", po::value<bool>(&params.volume_modeler.use_features)->zero_tokens(), "use features")
        ("vs", po::value<float>(&params.volume.cell_size), "volume cell size")
        ("vx", po::value<int>(&params.volume.cell_count.x()), "volume x")
        ("vy", po::value<int>(&params.volume.cell_count.y()), "volume y")
        ("vz", po::value<int>(&params.volume.cell_count.z()), "volume z")
        ("vxyz", po::value<int>(&vxyz), "vxyz")
        ("mt", po::value<std::string>(&model_type_string), "model type string (see code)")
        ("mt1", po::value<std::string>(&model_type_1_string), "model_type_1_string (see code)")
        ("mt2", po::value<std::string>(&model_type_2_string), "model_type_2_string (see code)")
        ("max", po::value<float>(&max_depth), "max_depth")
        ("max_mb_gpu", po::value<int>(&params.grid.max_mb_gpu), "max_mb_gpu")
        ("max_mb_system", po::value<int>(&params.grid.max_mb_system), "max_mb_system")
        ("border_size", po::value<int>(&params.grid.border_size), "border_size")
        ("grid_size", po::value<int>(&params.grid.grid_size), "grid_size")
        ("v", po::value<bool>(&params.volume_modeler.verbose)->zero_tokens(), "verbose")

        ("pause_before_first_frame", po::value<bool>(&pause_before_first_frame)->zero_tokens(), "pause_before_first_frame")
        ("pause_after_first_frame", po::value<bool>(&pause_after_first_frame)->zero_tokens(), "pause_after_first_frame")
        ("pause_every_frame", po::value<bool>(&pause_every_frame)->zero_tokens(), "pause_every_frame")
        ("pause_after_loop_closure", po::value<bool>(&pause_after_loop_closure)->zero_tokens(), "pause_after_loop_closure")

        ("test_set_weights", po::value<float>(&test_set_weights), "test_set_weights") // test setting max weight
        ("first_frame_centroid", po::value<bool>(&params.volume_modeler.first_frame_centroid)->zero_tokens(), "first_frame_centroid")
        ("first_frame_origin", po::value<bool>(&params.volume_modeler.first_frame_origin)->zero_tokens(), "first_frame_origin")
        ("frame_increment", po::value<int>(&frame_increment), "frame_increment")
        ("frame_start", po::value<int>(&frame_start), "frame_start")
        ("frame_end", po::value<int>(&frame_end), "frame_end")
        ("cli", po::value<bool>(&params.volume_modeler.command_line_interface)->zero_tokens(), "command_line_interface")
        // set_color_weights now true by default
        //		("set_color_weights", po::value<bool>(&params.volume_modeler.set_color_weights)->zero_tokens(), "set_color_weights")

        // could enable again for grids (avoiding confusion with kmeans stuff)
        //("debug_render", po::value<bool>(&params.grid.debug_render)->zero_tokens(), "debug_render")
        ("cl_intel_cpu", po::value<bool>(&cl_intel_cpu)->zero_tokens(), "cl_intel_cpu")
        ("cl_amd_cpu", po::value<bool>(&cl_amd_cpu)->zero_tokens(), "cl_amd_cpu")
        ("temp_folder", po::value<fs::path>(&params.grid.temp_folder), "temp_folder")
        ("cam_fx", po::value<float>(&params.camera.focal.x()), "cam_fx")
        ("cam_fy", po::value<float>(&params.camera.focal.y()), "cam_fy")
        ("cam_sx", po::value<int>(&params.camera.size.x()), "cam_sx")
        ("cam_sy", po::value<int>(&params.camera.size.y()), "cam_sy")
        ("cam_cx", po::value<float>(&params.camera.center.x()), "cam_cx")
        ("cam_cy", po::value<float>(&params.camera.center.y()), "cam_cy")
        ("cam_min_depth", po::value<float>(&params.camera.min_max_depth[0]), "cam_min_depth")
        ("test_align_weights", po::value<bool>(&test_align_weights)->zero_tokens(), "test_align_weights")
        ("test_add_depth_weights", po::value<bool>(&test_add_depth_weights)->zero_tokens(), "test_add_depth_weights")
        ("test_single_pose", po::value<bool>(&test_single_pose)->zero_tokens(), "test_single_pose")
        ("test_set_value_in_sphere", po::value<bool>(&test_set_value_in_sphere)->zero_tokens(), "test_set_value_in_sphere")
        ("test_set_value_in_box", po::value<bool>(&test_set_value_in_box)->zero_tokens(), "test_set_value_in_box")
        ("glfw_join", po::value<bool>(&glfw_join)->zero_tokens(), "glfw_join")
        ("glfw_join_full_mesh", po::value<bool>(&glfw_join_full_mesh)->zero_tokens(), "glfw_join_full_mesh")
        ("save_mesh_every_n", po::value<int>(&save_mesh_every_n), "save_mesh_every_n")
        ("save_state_every_n", po::value<int>(&save_state_every_n), "save_state_every_n")
        ("save_state_at_end", po::value<bool>(&save_state_at_end)->zero_tokens(), "save_state_at_end")


        ("circle_mask_pixel_radius", po::value<int>(&circle_mask_pixel_radius), "circle_mask_pixel_radius")
        ("circle_mask_x", po::value<int>(&circle_mask_x), "circle_mask_x")
        ("circle_mask_y", po::value<int>(&circle_mask_y), "circle_mask_y")

        ("add_grids_in_frustum", po::value<bool>(&params.grid.add_grids_in_frustum)->zero_tokens(), "add_grids_in_frustum")

        ("align_debug_images", po::value<bool>(&align_debug_images)->zero_tokens(), "align_debug_images")
        ("loop_closure_debug_images", po::value<bool>(&loop_closure_debug_images)->zero_tokens(), "loop_closure_debug_images")
        ("glfw_mesh_every_n", po::value<int>(&glfw_mesh_every_n), "glfw_mesh_every_n")
        ("suppress_debug_images", po::value<bool>(&suppress_debug_images)->zero_tokens(), "suppress_debug_images")
        ("glfw_background_intensity", po::value<float>(&glfw_background_intensity), "glfw_background_intensity")


        ("debug_grid_motion", po::value<bool>(&params.grid.debug_grid_motion)->zero_tokens(), "debug_grid_motion")

        ("loop_closure", po::value<bool>(&params.loop_closure.loop_closure)->zero_tokens(), "loop_closure")
        ("loop_closure_save_meshes_always", po::value<bool>(&loop_closure_save_meshes_always)->zero_tokens(), "loop_closure_save_meshes_always")
        ("loop_closure_edge_strength", po::value<float>(&params.loop_closure.loop_closure_edge_strength), "loop_closure_edge_strength")
        ("optimize_iterations", po::value<int>(&params.loop_closure.optimize_iterations), "optimize_iterations")
        ("min_fraction", po::value<float>(&params.loop_closure.min_fraction), "min_fraction")
        ("keyframe_distance_create", po::value<float>(&params.loop_closure.keyframe_distance_create), "keyframe_distance_create")
        ("keyframe_angle_create", po::value<float>(&params.loop_closure.keyframe_angle_create), "keyframe_angle_create")
        ("keyframe_distance_match", po::value<float>(&params.loop_closure.keyframe_distance_match), "keyframe_distance_match")
        ("keyframe_angle_match", po::value<float>(&params.loop_closure.keyframe_angle_match), "keyframe_angle_match")

        ("activation_mode", po::value<std::string>(&activation_mode_string), "activation_mode")
        ("activate_age", po::value<int>(&params.loop_closure.activate_age), "activate_age")
        ("activate_full_graph_depth", po::value<int>(&params.loop_closure.activate_full_graph_depth), "activate_full_graph_depth")

        ("low_resolution", po::value<bool>(&low_resolution)->zero_tokens(), "low_resolution")

        ("align_weight_color", po::value<float>(&params.alignment.weight_color), "align_weight_color")
        ("align_weight_icp", po::value<float>(&params.alignment.weight_icp), "align_weight_icp")

        ("max_weight_icp", po::value<float>(&params.volume.max_weight_icp), "max_weight_icp")
        ("max_weight_color", po::value<float>(&params.volume.max_weight_color), "max_weight_color")
        ("use_most_recent_color", po::value<bool>(&params.volume.use_most_recent_color)->zero_tokens(), "use_most_recent_color")

        ("mask_object", po::value<bool>(&params.mask_object.mask_object)->zero_tokens(), "mask_object")
        ("mask_hand", po::value<bool>(&params.mask_object.mask_hand)->zero_tokens(), "mask_hand")
        ("histogram_hand_file", po::value<fs::path>(&params.mask_object.histogram_hand_file), "histogram_hand_file")
        ("histogram_object_file", po::value<fs::path>(&params.mask_object.histogram_object_file), "histogram_object_file")

        ("debug_show_nonzero_voxels", po::value<bool>(&params.volume_modeler.debug_show_nonzero_voxels)->zero_tokens(), "debug_show_nonzero_voxels")

        ("patch_segments_min_size", po::value<int>(&params.patch.segments_min_size), "patch_segments_min_size")
        ("patch_border_create", po::value<float>(&params.patch.border_create), "patch_border_create")
        ("patch_segments_create_of_all_sizes", po::value<bool>(&params.patch.segments_create_of_all_sizes)->zero_tokens(), "patch_segments_create_of_all_sizes")
        ("debug_patch_creation", po::value<bool>(&params.patch.debug_patch_creation)->zero_tokens(), "debug_patch_creation")

        ("update_interface_view_pose", po::value<bool>(&params.volume_modeler.update_interface_view_pose)->zero_tokens(), "update_interface_view_pose")
        ("read_glfw_buffer", po::value<bool>(&read_glfw_buffer)->zero_tokens(), "read_glfw_buffer")
        ("fixed_top_down_view", po::value<bool>(&fixed_top_down_view)->zero_tokens(), "fixed_top_down_view")
        ("fixed_top_down_view_height", po::value<float>(&fixed_top_down_view_height), "fixed_top_down_view_height")
        ("fixed_view_pos_at_up", po::value<std::vector<float> >(&fixed_view_pos_at_up)->multitoken(), "fixed_view_pos_at_up")

        ("glfw_width", po::value<int>(&glfw_width), "glfw_width")
        ("glfw_height", po::value<int>(&glfw_height), "glfw_height")

        ("learn_histogram_hand", po::value<bool>(&learn_histogram_hand)->zero_tokens(), "learn_histogram_hand")
        ("learn_histogram_object", po::value<bool>(&learn_histogram_object)->zero_tokens(), "learn_histogram_object")

        ("regularize_lambda", po::value<float>(&params.alignment.regularize_lambda), "regularize_lambda")

        ("mask_hand_dilate_iterations", po::value<int>(&params.mask_object.mask_hand_dilate_iterations), "mask_hand_dilate_iterations")

        ("debug_add_volume", po::value<bool>(&debug_add_volume)->zero_tokens(), "debug_add_volume")
        ("test_patch_reorientation", po::value<bool>(&params.patch.test_patch_reorientation)->zero_tokens(), "test_patch_reorientation")
        ("test_patch_reorientation_glfw", po::value<bool>(&params.patch.test_patch_reorientation_glfw)->zero_tokens(), "test_patch_reorientation_glfw")

        ("test_nvidia_gpu_memory", po::value<bool>(&test_nvidia_gpu_memory)->zero_tokens(), "test_nvidia_gpu_memory")

        ("loop_input", po::value<bool>(&loop_input)->zero_tokens(), "loop_input")

        ("use_multiscale", po::value<bool>(&params.alignment.use_multiscale)->zero_tokens(), "use_multiscale")
        ("pyramid_levels", po::value<int>(&params.alignment.pyramid_levels), "pyramid_levels")
        ("gn_max_iterations", po::value<int>(&params.alignment.gn_max_iterations), "gn_max_iterations")

        ("grid_free", po::value<bool>(&params.grid.grid_free)->zero_tokens(), "grid_free")

        ("debug_optimize", po::value<bool>(&params.loop_closure.debug_optimize)->zero_tokens(), "debug_optimize")
        ("debug_save_meshes", po::value<bool>(&params.loop_closure.debug_save_meshes)->zero_tokens(), "debug_save_meshes")
        ("debug_disable_merge", po::value<bool>(&params.loop_closure.debug_disable_merge)->zero_tokens(), "debug_disable_merge")
        ("debug_merge_show_points", po::value<bool>(&params.loop_closure.debug_merge_show_points)->zero_tokens(), "debug_merge_show_points")

        ("debug_no_moving_mesh", po::value<bool>(&params.moving_volume_grid.debug_no_moving_mesh)->zero_tokens(), "debug_no_moving_mesh")
        ("debug_disable_merge_on_shift", po::value<bool>(&params.moving_volume_grid.debug_disable_merge_on_shift)->zero_tokens(), "debug_disable_merge_on_shift")
        ("debug_fix_g2o_vertices_for_keyframes", po::value<bool>(&params.moving_volume_grid.debug_fix_g2o_vertices_for_keyframes)->zero_tokens(), "debug_fix_g2o_vertices_for_keyframes")
        ("debug_clipping", po::value<bool>(&params.moving_volume_grid.debug_clipping)->zero_tokens(), "debug_clipping")
        ("debug_delete_edges_for_merged_volumes", po::value<bool>(&params.moving_volume_grid.debug_delete_edges_for_merged_volumes)->zero_tokens(), "debug_delete_edges_for_merged_volumes")

        ("scale_images", po::value<float>(&params.volume_modeler.scale_images), "scale_images")

        ("model_histogram_bin_count", po::value<int>(&params.model_histogram.bin_count), "model_histogram_bin_count")
        ("model_histogram_abs", po::value<float>(&model_histogram_abs), "model_histogram_abs")
        ("debug_pick_pixel_depth_offset", po::value<float>(&params.model_histogram.debug_pick_pixel_depth_offset), "debug_pick_pixel_depth_offset")
        ("debug_points_along_ray", po::value<int>(&params.model_histogram.debug_points_along_ray), "debug_points_along_ray")

        ("pick_pixel", po::value<bool>(&pick_pixel)->zero_tokens(), "pick_pixel")
        ("extra_skip_hack", po::value<int>(&extra_skip_hack), "extra_skip_hack")
        ("minimum_relative_count", po::value<float>(&params.model_k_means.minimum_relative_count), "minimum_relative_count")

        ("normals_smooth_iterations", po::value<int>(&params.normals.smooth_iterations), "normals_smooth_iterations")

        ("use_six_volumes", po::value<bool>(&use_six_volumes)->zero_tokens(), "use_six_volumes")

        ("compatibility_add_max_angle_degrees", po::value<float>(&params.model_k_means.compatibility_add_max_angle_degrees), "compatibility_add_max_angle_degrees")
        ("compatibility_render_max_angle_degrees", po::value<float>(&params.model_k_means.compatibility_render_max_angle_degrees), "compatibility_render_max_angle_degrees")

        ("debug_rendering", po::value<bool>(&params.model_k_means.debug_rendering)->zero_tokens(), "debug_rendering")
        ("render_all_6", po::value<bool>(&params.model_k_means.render_all_6)->zero_tokens(), "render_all_6")
        ("render_all_6_from_canonical", po::value<bool>(&params.model_k_means.render_all_6_from_canonical)->zero_tokens(), "render_all_6_from_canonical")
        ("render_all_6_from_canonical_distance", po::value<float>(&params.model_k_means.render_all_6_from_canonical_distance), "render_all_6_from_canonical_distance")

        ("render_all_6_scale", po::value<float>(&params.model_k_means.render_all_6_scale), "render_all_6_scale")


        ("empty_always_included", po::value<bool>(&params.model_k_means.empty_always_included)->zero_tokens(), "empty_always_included")

        ("debug_slices", po::value<bool>(&params.model_k_means.debug_slices)->zero_tokens(), "debug_slices")
        ("slice_images_scale", po::value<float>(&params.model_k_means.slice_images_scale), "slice_images_scale")
        ("slice_color_max", po::value<float>(&params.model_k_means.slice_color_max), "slice_color_max")


        ("cos_weight", po::value<bool>(&params.model_k_means.cos_weight)->zero_tokens(), "cos_weight")
        ("min_truncation_distance", po::value<float>(&params.volume.min_truncation_distance), "min_truncation_distance")
        ("debug_meshes", po::value<bool>(&params.model_k_means.debug_meshes)->zero_tokens(), "debug_meshes")

        ("new_render", po::value<bool>(&params.grid.new_render)->zero_tokens(), "new_render")


        // only takes effect if the distance based on sigmas thing is true (it ISN'T and HASN'T BEEN)
        ("icp_max_distance", po::value<float>(&params.alignment.icp_max_distance), "icp_max_distance")
        ("icp_max_normal", po::value<float>(&params.alignment.icp_max_normal), "icp_max_normal")

        ("use_new_alignment", po::value<bool>(&params.alignment.use_new_alignment)->zero_tokens(), "use_new_alignment")

        ("huber_icp", po::value<float>(&params.alignment.huber_icp), "huber_icp")
        ("huber_color", po::value<float>(&params.alignment.huber_color), "huber_color")
        ("align_debug_images_scale", po::value<float>(&params.alignment.debug_images_scale), "params.alignment.debug_images_scale")

        ("apply_vignette_model", po::value<bool>(&params.volume_modeler.apply_vignette_model)->zero_tokens(), "apply_vignette_model")

        ("skip_single_mesh", po::value<bool>(&params.grid.skip_single_mesh)->zero_tokens(), "skip_single_mesh")

        ("use_dbow_place_recognition", po::value<bool>(&params.loop_closure.use_dbow_place_recognition)->zero_tokens(), "use_dbow_place_recognition")
        ("dbow_resources_folder", po::value<fs::path>(&params.dbow_place_recognition.resources_folder), "dbow_resources_folder")

        ("debug_allow_sequential_closures", po::value<bool>(&params.dbow_place_recognition.debug_allow_sequential_closures)->zero_tokens(), "debug_allow_sequential_closures")
        ("dbow_debug_images_save", po::value<bool>(&params.dbow_place_recognition.debug_images_save)->zero_tokens(), "dbow_debug_images_save")
        ("dbow_debug_images_show", po::value<bool>(&params.dbow_place_recognition.debug_images_show)->zero_tokens(), "dbow_debug_images_show")

            ("rect_mask_x", po::value<int>(&rect_mask_x), "rect_mask_x")
            ("rect_mask_y", po::value<int>(&rect_mask_y), "rect_mask_y")
            ("rect_mask_width", po::value<int>(&rect_mask_width), "rect_mask_width")
            ("rect_mask_height", po::value<int>(&rect_mask_height), "rect_mask_height")

            ("debug_max_total_loop_closures", po::value<int>(&params.loop_closure.debug_max_total_loop_closures), "debug_max_total_loop_closures")

            ("save_input", po::value<bool>(&save_input)->zero_tokens(), "save_input")
            ("save_images", po::value<bool>(&save_images)->zero_tokens(), "save_images")
            ("save_masked_input", po::value<bool>(&save_masked_input)->zero_tokens(), "save_masked_input")



        ;
    po::variables_map vm;
    try {
        //po::command_line_style::unix_style ^ po::command_line_style::allow_short
        po::store(po::parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm);
        po::notify(vm);
    }
    catch (std::exception & e) {
        cout << desc << endl;
        cout << e.what() << endl;
        return false;
    }
    if (vm.count("help")) {
        cout << "desc" << endl;
        return false;
    }
    if (!model_type_string.empty()) {
        params.volume_modeler.model_type = VolumeModeler::modelTypeFromString(model_type_string);
    }
    if (!model_type_1_string.empty()) {
        params.volume_modeler.model_type_1 = VolumeModeler::modelTypeFromString(model_type_1_string);
    }
    if (!model_type_2_string.empty()) {
        params.volume_modeler.model_type_2 = VolumeModeler::modelTypeFromString(model_type_2_string);
    }
    if (vm.count("activation_mode")) {
        if (activation_mode_string == "age") {
            params.loop_closure.activation_mode = ACTIVATION_MODE_AGE;
        }
        else if (activation_mode_string == "full_graph") {
            params.loop_closure.activation_mode = ACTIVATION_MODE_FULL_GRAPH;
        }
        else if (activation_mode_string == "keyframe_graph") {
            params.loop_closure.activation_mode = ACTIVATION_MODE_KEYFRAME_GRAPH;
        }
        else {
            cout << "Unknown activation_mode_string: " << activation_mode_string << endl;
            exit(1);
        }
    }
    if (model_histogram_abs > 0) {
        params.model_histogram.min_value = -model_histogram_abs;
        params.model_histogram.max_value = model_histogram_abs;
    }
    if (cl_path.empty()) {
        cl_path = cl_path_default;
    }
    if (vxyz > 0) {
        params.volume.cell_count = Eigen::Array3i::Constant(vxyz);
    }
    if (use_six_volumes) {
        // implies a couple of changes
        params.model_k_means.k = 6;
        params.model_k_means.store_normals = false;
    }
    // only do this here
    if (low_resolution) {
        openni2_params.resolution_x /= 2;
        openni2_params.resolution_y /= 2;
        params.camera.size /= 2;
        params.camera.focal /= 2;
        params.camera.setCenterFromSize();
    }

    ParamsCamera params_camera_f200_depth;
    Eigen::Affine3f f200_extrinsic = Eigen::Affine3f::Identity();
    if (use_f200_camera) {


#if 0
        // from dan_calib.xml (worked well for first camera)
        // though matlab is column major, this appears row major!
        params.camera.focal[0] = 613.4891;
        params.camera.focal[1] = 613.094;
        params.camera.center[0] = 333.7792;
        params.camera.center[1] = 247.0641;
        params.camera.size[0] = 640;
        params.camera.size[1] = 480;
        params.camera.min_max_depth[0] = 0.1; // or whatever...close!

        params_camera_f200_depth.focal[0] = 479.3389;
        params_camera_f200_depth.focal[1] = 479.0413;
        params_camera_f200_depth.center[0] = 322.3937;
        params_camera_f200_depth.center[1] = 231.3332;
        params_camera_f200_depth.size[0] = 640;
        params_camera_f200_depth.size[1] = 480;
        params_camera_f200_depth.min_max_depth = params.camera.min_max_depth;

        Eigen::Matrix3f rotation;
        rotation << 1 , -0.002929 , 0.0011267 , 0.0029431 , 0.99991 , -0.01272 , -0.0010893 , 0.012723 , 0.99992;
        f200_extrinsic.prerotate(rotation);
        Eigen::Vector3f translation;
        translation << 24.2003 , -0.496792 , 3.19385;
        translation /= 1000.f; // mm to m
        
        f200_extrinsic.pretranslate(translation);
#endif

        // from dan_calib.xml (worked well for first camera)
        // though matlab is column major, this appears row major!
        params.camera.focal[0] = 604.899;
        params.camera.focal[1] = 605.5411;
        params.camera.center[0] = 326.528;
        params.camera.center[1] = 248.0574;
        params.camera.size[0] = 640;
        params.camera.size[1] = 480;
        params.camera.min_max_depth[0] = 0.1; // or whatever...close!

        params_camera_f200_depth.focal[0] = 474.2684;
        params_camera_f200_depth.focal[1] = 474.6179;
        params_camera_f200_depth.center[0] = 311.0716;
        params_camera_f200_depth.center[1] = 235.5572;
        params_camera_f200_depth.size[0] = 640;
        params_camera_f200_depth.size[1] = 480;
        params_camera_f200_depth.min_max_depth = params.camera.min_max_depth;

        Eigen::Matrix3f rotation;
        rotation << 0.99994 , -0.0033716 , -0.010351 ,
                0.003302 , 0.99997 , -0.0067372 ,
                0.010374 , 0.0067027 , 0.99992;
        f200_extrinsic.prerotate(rotation);
        Eigen::Vector3f translation;
        translation << 23.8433 ,  0.182813 , 0.83563;
        translation /= 1000.f; // mm to m

        f200_extrinsic.pretranslate(translation);





        cout << "f200_extrinsic: " << endl;
        cout << f200_extrinsic.matrix() << endl;
    }


    // set camera params based on max depth??
    if (max_depth > 0) {
        params.camera.min_max_depth[1] = max_depth;
    }

    // create folders for saving
    // should maybe do inside volume modeler now?
    if (!params.volume_modeler.output.empty()) {
        if (!fs::exists(params.volume_modeler.output) && !fs::create_directories(params.volume_modeler.output)) {
            throw std::runtime_error("bad folder: " + params.volume_modeler.output.string());
        }
    }

    // obviously make a function for this:
    fs::path folder_save_input = params.volume_modeler.output/"save_input";
    if (!fs::exists(folder_save_input) && !fs::create_directories(folder_save_input)) {
        cout << "bad folder: " << folder_save_input << endl;
        exit(1);
    }
    fs::path folder_save_images = params.volume_modeler.output/"save_images";
    if (!fs::exists(folder_save_images) && !fs::create_directories(folder_save_images)) {
        cout << "bad folder: " << folder_save_images << endl;
        exit(1);
    }
    fs::path folder_save_masked_input = params.volume_modeler.output/"save_masked_input";
    if (!fs::exists(folder_save_masked_input) && !fs::create_directories(folder_save_masked_input)) {
        cout << "bad folder: " << folder_save_masked_input << endl;
        exit(1);
    }


    ///////////////
    // start by initializing opencl
    boost::scoped_ptr<CL> cl_ptr;
    OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
    OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;
    if (cl_intel_cpu) {
        platform_type = OPENCL_PLATFORM_INTEL;
        context_type = OPENCL_CONTEXT_CPU;
    }
    else if (cl_amd_cpu) {
        platform_type = OPENCL_PLATFORM_AMD;
        context_type = OPENCL_CONTEXT_CPU;
    }
    cl_ptr.reset(new CL(platform_type, context_type));
    if (!cl_ptr->isInitialized()) throw std::runtime_error ("Failed to initialize Open CL");





    /////////////////////
    ////// Learn histogram (independent of volume modeler)
    boost::shared_ptr<LearnHistogram> learn_histogram(new LearnHistogram(params.mask_object));
    fs::path learn_histogram_file;
    if (learn_histogram_hand) {
        if (params.mask_object.histogram_hand_file.empty()) {
            cout << "learn_histogram_hand && histogram_hand_file.empty()" << endl;
            exit(1);
        }
        learn_histogram_file = params.mask_object.histogram_hand_file;
    }
    if (learn_histogram_object) {
        if (params.mask_object.histogram_object_file.empty()) {
            cout << "learn_histogram_object && histogram_object_file.empty()" << endl;
            exit(1);
        }
        learn_histogram_file = params.mask_object.histogram_object_file;
    }
    cv::Mat learn_histogram_mat;
    if (!learn_histogram_file.empty() && fs::exists(learn_histogram_file)) {
        loadHistogram(learn_histogram_file.string(), learn_histogram_mat);
        learn_histogram->init(learn_histogram_mat);
    }

    /////////////////////
    ////////// 
    // Initialize volume modeler
    boost::shared_ptr<OpenCLAllKernels> all_kernels (new OpenCLAllKernels(*cl_ptr, cl_path));

    boost::shared_ptr<VolumeModeler> volume_modeler (new VolumeModeler(all_kernels, params));

    // for debugging purposes:
    // remove eventually
    boost::shared_ptr<VolumeModeler> alternate_volume_modeler;
    if (debug_add_volume) {
        alternate_volume_modeler.reset(new VolumeModeler(all_kernels, params));
    }

    // set initial debug
    volume_modeler->setAlignDebugImages(align_debug_images);

#ifdef VOLUME_MODELER_GLFW
    /*
    if (read_glfw_buffer) {
    // match camera???
    width = params.camera.size[0];
    height = params.camera.size[1];
    }
    */

    boost::shared_ptr<VolumeModelerGLFW> volume_modeler_glfw;
    if (!params.volume_modeler.command_line_interface) {
        ///////// disabled_keys
        std::set<std::string> disabled_keys;
        disabled_keys.insert(params.glfw_keys.volumes_active);
        disabled_keys.insert(params.glfw_keys.pose_graph_all);
        disabled_keys.insert(params.glfw_keys.cameras_all);

        volume_modeler_glfw.reset(new VolumeModelerGLFW(glfw_width, glfw_height, disabled_keys));
        volume_modeler_glfw->runInThread();
        volume_modeler->setUpdateInterface(volume_modeler_glfw);
        volume_modeler_glfw->setClearColorSync(glfw_background_intensity, glfw_background_intensity, glfw_background_intensity);
        volume_modeler_glfw->setEnableReadBuffer(read_glfw_buffer);
        if (fixed_top_down_view) {
            Eigen::Vector3f camera (0,-fixed_top_down_view_height,0);
            Eigen::Vector3f target (0,0,0);
            Eigen::Vector3f up (0,0,1);
            volume_modeler_glfw->setGluLookAt(camera, target, up);
        }
        else if (!fixed_view_pos_at_up.empty()) {
            if (fixed_view_pos_at_up.size() != 9) {
                cout << "Incorrect number of arguments for fixed_view_pos_at_up" << endl;
                exit(1);
            }
            {
                std::vector<float> const& v = fixed_view_pos_at_up;
                Eigen::Vector3f camera (v[0],v[1],v[2]);
                Eigen::Vector3f target (v[3],v[4],v[5]);
                Eigen::Vector3f up (v[6],v[7],v[8]);
                volume_modeler_glfw->setGluLookAt(camera, target, up);
            }
        }

        // also pick pixel only if not cli
        if (pick_pixel) {
            boost::shared_ptr<PickPixel> pick_pixel_ptr(new PickPixel());
            volume_modeler->setPickPixel(pick_pixel_ptr);
        }
    }
#endif


    ////////////////////////////////
    // load state
    if (!load_state.empty()) {
        cout << "Loading state from: " << load_state << endl;
        volume_modeler->load(load_state);
        cout << "Finished loading state" << endl;

        // look at them before first frame...(if you pause later)
        volume_modeler->refreshUpdateInterfaceForModels();
    }




    /////////////////////////////
    // this is an awkward hack:
    boost::shared_ptr<PoseProviderBase> pose_provider(new PoseProviderBase());
    if (!load_camera_poses.empty()) {
        cout << "Loading camera poses from: " << load_camera_poses << endl;
        pose_provider.reset(new PoseProviderStandard(load_camera_poses));
    }



    /////////////////////
    // assume that when you load state, you want to skip frames based on that state
    {
        size_t frames_added = volume_modeler->getFramesAdded();
        if (frames_added > 0) {
            frame_start = frames_added + 1;
            cout << "Set frame_start to: " << frame_start << endl;
            // do you also want to skip poses?
            // for now, skip the issue
            if (!load_camera_poses.empty()) {
                cout << "Right now can't load poses and load state" << endl;
                exit(1);
            }
        }
    }




    ///////////////////
    // how we get frames
    // note that this may use params!
    boost::scoped_ptr<FrameProviderBase> frame_provider;
    bool frame_provider_is_live = false;
    if (empty) {
        frame_provider.reset(new FrameProviderEmpty());
    }
    else if (!input_png.empty()) {
        boost::shared_ptr<FrameProviderPNG> core_frame_provider;
        if (expect_evan_poses) {
            core_frame_provider.reset(new FrameProviderPNG(input_png, 1000.f, true));
        }
        else if (expect_evan_camera_list) {
            core_frame_provider.reset(new FrameProviderPNG(input_png, 1000.f, false, "camera_list.txt"));
        }
        else {
            core_frame_provider.reset(new FrameProviderPNG(input_png));
        }
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_png_depth.empty()) {
        boost::shared_ptr<FrameProviderPNGDepth> core_frame_provider (new FrameProviderPNGDepth(input_png_depth, 10000.f, "camera_poses.txt"));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_pcd.empty()) {
#ifdef FRAME_PROVIDER_PCL
        boost::shared_ptr<FrameProviderPCD> core_frame_provider (new FrameProviderPCD(input_pcd));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
        add_frame_to_volume = true;
#else
        cout << "You must compile with FRAME_PROVIDER_PCL to enable PCD input" << endl;
        exit(1);
#endif
    }
    else if (!input_handa.empty()) {
        boost::shared_ptr<FrameProviderHanda> core_frame_provider (new FrameProviderHanda(input_handa, params.camera));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_yuyin.empty()) {
        boost::shared_ptr<FrameProviderYuyin> core_frame_provider (new FrameProviderYuyin(input_yuyin, 10000.f, yuyin_prefix));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_freiburg.empty()) {
        boost::shared_ptr<FrameProviderFreiburg> core_frame_provider (new FrameProviderFreiburg(input_freiburg));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_luis.empty()) {
        fs::path camera_list = fs::path();
        if (expect_luis_camera_list) {
            camera_list = "pose.txt";
        }
        boost::shared_ptr<FrameProviderLuis> core_frame_provider (new FrameProviderLuis(input_luis, 1000.f, camera_list));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_arun.empty()) {
        boost::shared_ptr<FrameProviderArun> core_frame_provider (new FrameProviderArun(input_arun, 1000.f));
        frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
    }
    else if (!input_oni.empty()) {
#ifdef FRAME_PROVIDER_OPENNI2
        openni2_params.file = input_oni;

        try {
            boost::shared_ptr<FrameProviderOpenni2> core_frame_provider (new FrameProviderOpenni2(openni2_params));
            frame_provider.reset(new FrameProviderFileWrapper(core_frame_provider, frame_increment, frame_start, frame_end));
        }
        catch (std::runtime_error & e) {
            cout << "Error initializing frame provider: " << e.what() << endl;
            exit(1);
        }
#else
        cout << "You must compile with FRAME_PROVIDER_OPENNI2 to enable oni file input" << endl;
        exit(1);
#endif
    }
    else {
#ifdef FRAME_PROVIDER_OPENNI2
        try {
            frame_provider.reset(new FrameProviderOpenni2(openni2_params));
        }
        catch (std::runtime_error & e) {
            cout << "Error initializing frame provider: " << e.what() << endl;
            exit(1);
        }
        frame_provider_is_live = true;
#else
        cout << "You must compile with FRAME_PROVIDER_OPENNI2 to enable live input" << endl;
        exit(1);
#endif
    }




    ////////////////////
    // process frames
    bool process_next_frame = true; // could turn off inside
    bool openni_auto = true;
    bool add_frame_to_volume = !frame_provider_is_live;
    boost::timer timer;
    int frame_counter = 0;
    int key = -1;
    std::vector<std::string> filename_to_pose_list;
    while(process_next_frame) {
        boost::timer t_frame;

        bool pause_after_this_frame = pause_every_frame;

        if (pause_after_first_frame && frame_counter == 0) {
            pause_after_this_frame = true;
        }

        if (pause_before_first_frame) {
            cout << "pause_before_first_frame..." << endl;
            cv::waitKey();
        }

        boost::shared_ptr<Frame> frame_ptr(new Frame(all_kernels));
        bool frame_valid = frame_provider->getNextFrame(frame_ptr->mat_color_bgra, frame_ptr->mat_depth);
        // ok to have ros_timestamp_valid invalid...will just be 0,0
        bool ros_timestamp_valid = frame_provider->getLastROSTimestamp(frame_ptr->ros_timestamp);
        frame_counter++;
        if (params.volume_modeler.verbose) {
            cout << "---" << endl;
            cout << "Frame: " << frame_counter << endl;
            cout << "Valid: " << frame_valid << endl;
            cout << "ROS Timestamp: " << frame_ptr->ros_timestamp << endl;
        }

        //////////////////
        // can do something here after the frames run out
        if (!frame_valid) {
            if (loop_input) {
                // grab camera pose as a cheat?  no...assume loopy input

                frame_provider->reset();
                continue;
            }
            else {
                cout << "break on !frame_valid" << endl;
                break; 
            }
        }

        ////////////////////////
        // mess with the frame
        // note that setColorWeights happens inside addFrame() now if param is true

        // hack in all arun f200 settings hardcoded
        if (use_f200_camera) {
            KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels);
            KernelTransformPoints _KernelTransformPoints(*all_kernels);
            //KernelPointsToDepthImage _KernelPointsToDepthImage(*all_kernels);

            ImageBuffer depth_source(all_kernels->getCL());
            depth_source.setMat(frame_ptr->mat_depth);
            ImageBuffer points_source(all_kernels->getCL());
            _KernelDepthImageToPoints.runKernel(depth_source, points_source, params_camera_f200_depth.focal, params_camera_f200_depth.center);

            ImageBuffer points_transformed(all_kernels->getCL());
            _KernelTransformPoints.runKernel(points_source, points_transformed, f200_extrinsic);

            cv::Mat points_transformed_mat = points_transformed.getMat();
            cv::Mat replacement_depth;
            cv::Mat original_pixels_ignore;
            projectPixels(params.camera, points_transformed_mat, replacement_depth, original_pixels_ignore);

            // try dilating the depths?
            cv::Mat replacement_depth_dilated;
            cv::dilate(replacement_depth, replacement_depth_dilated, cv::Mat());

            frame_ptr->mat_depth = replacement_depth_dilated;
        }


        frame_ptr->reduceToMaxDepth(max_depth);

        // apply vignette model?
        if (params.volume_modeler.apply_vignette_model)
        {
            boost::timer t;
            // todo: param
            const Eigen::Array3f vignette_model(-1.14405, 1.82316, -1.11109);

            // todo: opencl version
            //frame_ptr->mat_color_bgra = VignetteCalibration::applyVignetteModelPolynomial3<cv::Vec4b>(frame_ptr->mat_color_bgra, params.camera, vignette_model);
            ImageBuffer ib_color(all_kernels->getCL());
            ib_color.setMat(frame_ptr->mat_color_bgra);
            KernelVignetteApplyModelPolynomial3Uchar4 _KernelVignetteApplyModelPolynomial3Uchar4(*all_kernels);
            ImageBuffer ib_color_v(all_kernels->getCL());
            _KernelVignetteApplyModelPolynomial3Uchar4.runKernel(ib_color, ib_color_v, params.camera.center, vignette_model);
            frame_ptr->mat_color_bgra = ib_color_v.getMat();

            cout << "TIME apply_vignette_model: " << t.elapsed() << endl;
        }

        // smarter circle masking for Luis
        if (circle_mask_pixel_radius > 0) {
            int center_x = circle_mask_x >= 0 ? circle_mask_x : params.camera.center[0];
            int center_y = circle_mask_y >= 0 ? circle_mask_y : params.camera.center[1];
            frame_ptr->circleMaskPixelRadius(circle_mask_pixel_radius, center_y, center_x);
        }

        // rect masking as well (arun)
        if (rect_mask_width > 0 && rect_mask_height > 0) {
            cv::Rect rect_mask(rect_mask_x, rect_mask_y, rect_mask_width, rect_mask_height);
            cv::Mat old_depth = frame_ptr->mat_depth;
            frame_ptr->mat_depth = cv::Mat(old_depth.size(), CV_32F, cv::Scalar::all(0));
            old_depth(rect_mask).copyTo(frame_ptr->mat_depth(rect_mask));
        }

        // this is a bit hacky..avoids crashes in getting object mask without histograms
        if (!learn_histogram_file.empty()) {
            cv::Mat mat_color_bgr;
            cv::cvtColor(frame_ptr->mat_color_bgra, mat_color_bgr, CV_BGRA2BGR);
            bool learn_result = learn_histogram->learn(mat_color_bgr);
            if (learn_result) {
                learn_histogram_mat = learn_histogram->getHistogram();
                saveHistogram(learn_histogram_file.string(), learn_histogram_mat);
            }
            add_frame_to_volume = false;
        }

        if (test_add_depth_weights) {
            frame_ptr->mat_add_depth_weights = cv::Mat(frame_ptr->mat_depth.size(), CV_32F, cv::Scalar(1));
            // set upper left quadrant to 0...
            cv::Mat zero_region = frame_ptr->mat_add_depth_weights(cv::Rect(0, 0, frame_ptr->mat_add_depth_weights.cols / 2, frame_ptr->mat_add_depth_weights.rows / 2));
            zero_region.setTo(0);

            cv::imshow("mat_add_depth_weights", frame_ptr->mat_add_depth_weights);
        }

        if (test_align_weights) {
            frame_ptr->mat_align_weights = cv::Mat(frame_ptr->mat_depth.size(), CV_32F, cv::Scalar(1));
            // set upper left quadrant to 0...
            cv::Mat zero_region = frame_ptr->mat_align_weights(cv::Rect(0, 0, frame_ptr->mat_align_weights.cols / 2, frame_ptr->mat_align_weights.rows / 2));
            zero_region.setTo(0);

            cv::imshow("mat_align_weights", frame_ptr->mat_align_weights);
        }

        if (test_set_value_in_sphere) {
            Eigen::Vector3f center(0,0,0);
            float radius = 0.5;
            Eigen::Array4ub color(255,255,255,255);
            volume_modeler->setValueInSphere(1, 1, color, 1, center, radius);
            add_frame_to_volume = false; // avoid other work
        }


        if (test_set_value_in_box) {
            Eigen::Affine3f box_pose;
            box_pose = Eigen::Affine3f::Identity();
            box_pose.scale(Eigen::Vector3f(0.5,0.2,0.1));
            box_pose.pretranslate(Eigen::Vector3f(0.1,0.1,0.1));
            Eigen::Array4ub color(255,255,255,255);
            volume_modeler->setValueInBox(1, 1, color, 1, box_pose);
            add_frame_to_volume = false; // avoid other work
        }

        // save the input?
        if (save_input) {
            fs::path filename_color = folder_save_input / (boost::format("input_color_%05.png") % volume_modeler->getFramesAdded()).str();
            fs::path filename_depth = folder_save_input / (boost::format("input_depth_%05.png") % volume_modeler->getFramesAdded()).str();
            cv::imwrite(filename_color.string(), frame_ptr->mat_color_bgra);
            cv::imwrite(filename_depth.string(), frame_ptr->getPNGDepth());
        }

        if (save_masked_input) {
            cv::Mat masked_color(frame_ptr->mat_color_bgra.size(), frame_ptr->mat_color_bgra.type(), cv::Scalar::all(0));
            cv::Mat mask = frame_ptr->mat_depth > 0;
            frame_ptr->mat_color_bgra.copyTo(masked_color, mask);

            fs::path filename_masked_color = folder_save_masked_input / (boost::format("%05d_input_masked_color.png") % volume_modeler->getFramesAdded()).str();
            cv::imwrite(filename_masked_color.string(), masked_color);
        }

        if (!params.volume_modeler.command_line_interface) {
            // show the input

            //cv::imshow("input_color", frame_ptr->mat_color);
            //cv::imshow("input_depth", frame_ptr->getPrettyDepth());
            cv::Mat depth_4c;
            cv::cvtColor(frame_ptr->getPrettyDepth(), depth_4c, CV_GRAY2BGRA);
            cv::Mat input_both = create1x2(frame_ptr->mat_color_bgra, depth_4c);
            if (params.volume_modeler.scale_images > 0) {
                cv::Mat input_both_scaled;
                cv::resize(input_both, input_both_scaled, cv::Size(), params.volume_modeler.scale_images, params.volume_modeler.scale_images, cv::INTER_NEAREST);
                cv::imshow("input_both_scaled", input_both_scaled);
            }
            else {
                cv::imshow("input_both", input_both);
            }
            if (save_images) {
                fs::path filename = folder_save_images / (boost::format("input_both_%05d.png") % volume_modeler->getFramesAdded()).str();
                cv::imwrite(filename.string(), input_both);
            }

            // get the key
            int this_key = cv::waitKey(20); // This is in ms of added lag right here :)    <------ added lag

            // also check for glfw key (only here?)
#ifdef VOLUME_MODELER_GLFW
            if (volume_modeler_glfw) {
                int glfw_key = volume_modeler_glfw->getKeyLowerCaseSync();
                if (glfw_key > 0) {
                    this_key = glfw_key;
                    volume_modeler_glfw->clearKeySync();
                }
            }
#endif

            if (key < 0) key = this_key;
        }

        if (key == 'a' && frame_provider_is_live) {
            // hacky
#ifdef FRAME_PROVIDER_OPENNI2
            FrameProviderOpenni2* openni2_ptr = dynamic_cast<FrameProviderOpenni2*>(frame_provider.get());
            openni_auto = !openni_auto;
            cout << "Setting auto values to: " << openni_auto << endl;
            openni2_ptr->setAutoExposure(openni_auto);
            openni2_ptr->setAutoWhiteBalance(openni_auto);
#endif
        }
        else if (key == 's' && frame_provider_is_live) {
            add_frame_to_volume = !add_frame_to_volume;
            cout << "Setting add_frame_to_volume: " << add_frame_to_volume << endl;
        }
        else if (key == 'r' && frame_provider_is_live) {
            cout << "Reset..." << endl;
            volume_modeler->reset();
        }
        else if (key == 'p') {
            cout << "Pausing after this frame..." << endl;
            pause_after_this_frame = true;
        }
        else if (key == 'm') {
            cout << "Generating mesh..." << endl;
            fs::path folder_for_frame = (boost::format("mesh_%05d") % volume_modeler->getFramesAdded()).str();
            volume_modeler->generateAndSaveAllMeshes(params.volume_modeler.output / folder_for_frame);
            volume_modeler->saveCameraPoses(params.volume_modeler.output / folder_for_frame / "camera_poses.txt");
            cout << "Done with mesh" << endl;

            // save mesh equivalent:
#if 0
            MeshVertexVector vertex_list;
            TriangleVector triangle_list;
            volume_modeler->generateMesh(vertex_list, triangle_list);
            volume_modeler->saveMesh(vertex_list, triangle_list, save_file);
#endif
        }
        else if (key == 'g') {
#ifdef VOLUME_MODELER_GLFW
            if (volume_modeler_glfw) {
                MeshPtr mesh_ptr (new Mesh);
                volume_modeler->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
                volume_modeler_glfw->updateMesh(params.glfw_keys.mesh, mesh_ptr);
            }
#else
            cout << "not compiled with VOLUME_MODELER_GLFW" << endl;
#endif
        }
        else if (key == 'h') {
#ifdef VOLUME_MODELER_GLFW
            if (volume_modeler_glfw) {
                MeshPtr mesh_ptr (new Mesh);
                std::vector<bool> vertex_validity;
                std::vector<bool> triangle_validity;
                volume_modeler->generateMeshAndValidity(mesh_ptr->vertices, mesh_ptr->triangles, vertex_validity, triangle_validity);
                volume_modeler_glfw->updateMesh(params.glfw_keys.mesh, mesh_ptr);
            }
#else
            cout << "not compiled with VOLUME_MODELER_GLFW" << endl;
#endif
        }
        else if (key == 'c') {
            align_debug_images = !align_debug_images;
            cout << "align_debug_images: " << align_debug_images << endl;
        }
        else if (key == 'd') {
            loop_closure_debug_images = !loop_closure_debug_images;
            cout << "loop_closure_debug_images: " << loop_closure_debug_images << endl;
        }
        else if (key == 'u') {
            volume_modeler->debugCheckOverlap();
        }
        else if (key == 'x') {
            fs::path folder_for_frame = (boost::format("save_state_%05d") % volume_modeler->getFramesAdded()).str();
            cout << "saving state and quitting: " << folder_for_frame << endl;
            volume_modeler->save(params.volume_modeler.output / folder_for_frame);
            // and quit?
            exit(0);
        }
        else if (key == 'q') {
            cout << "quitting..." << endl;
            exit(0);
        }
        key = -1;

        ////////////////////////////////
        // actually align and add the frame!
        if (add_frame_to_volume) {
            volume_modeler->setAlignDebugImages(align_debug_images);
            bool volume_success = false;

            Eigen::Affine3f loaded_camera_pose;
            // assume pose_provider overrides those from the frame_provider
            bool load_pose_success = pose_provider->getNextPose(loaded_camera_pose);
            if (load_pose_success) {
                cout << "Got pose from pose_provider..." << endl;
            }
            else {
                // hack in the evan poses here...in general, might want to do this elsewhere
                load_pose_success = frame_provider->getLastFramePose(loaded_camera_pose);
                if (load_pose_success) {
                    cout << "Got pose from frame_provider..." << endl;
                }
            }
            if (load_pose_success) {
                if (extra_skip_hack > 0) {
                    static int extra_skip_hack_counter = extra_skip_hack;
                    if (extra_skip_hack_counter-- > 0) continue;
                    else extra_skip_hack_counter = extra_skip_hack;
                }

                volume_modeler->addFrame(*frame_ptr, loaded_camera_pose);
                if (render_after_loaded_camera) {
                    volume_modeler->render(loaded_camera_pose);
                }
                volume_success = true;
            }
            else {
                ///////// this is the real behavior
                boost::timer t_align_add_overall;
                volume_success = volume_modeler->alignAndAddFrame(*frame_ptr);
                if (params.volume_modeler.verbose) cout << "TIME alignAndAddFrame: " << t_align_add_overall.elapsed() << endl;
            }
            cv::Mat render_color, render_normals;
            volume_modeler->getLastRenderPretty(render_color, render_normals);

            if (!params.volume_modeler.command_line_interface) {
                //cv::imshow("render_color", render_color);
                //cv::imshow("render_normals", render_normals);
                cv::Mat render_both = create1x2(render_color, render_normals);
                if (params.volume_modeler.scale_images > 0) {
                    cv::Mat render_both_scaled;
                    cv::resize(render_both, render_both_scaled, cv::Size(), params.volume_modeler.scale_images, params.volume_modeler.scale_images, cv::INTER_NEAREST);
                    cv::imshow("render_both_scaled", render_both_scaled);
                }
                else {
                    cv::imshow("render_both", render_both);
                }

                // also extra ones
                // also other models here?
                // gotta do it after getLastRenderPretty unless you fix stuff...
                std::vector<cv::Mat> extra_color, extra_normals;
                volume_modeler->renderAllExtraModelsPretty(volume_modeler->getLastCameraPose(), extra_color, extra_normals);
                for (size_t i = 0; i < extra_color.size(); ++i) {
                    cv::Mat render_both = create1x2(extra_color[i], extra_normals[i]);
                    // could scale..
                    std::string name = (boost::format("render_both_%d") % i).str();
                    cv::imshow(name, render_both);
                }


                if (save_images) {
                    {
                        fs::path filename = folder_save_images / (boost::format("render_both_%05d.png") % volume_modeler->getFramesAdded()).str();
                        cv::imwrite(filename.string(), render_both);
                    }

                    {
                        // also hack in all four
                        // some of this is dup
                        cv::Mat color_4c = frame_ptr->mat_color_bgra.clone(); // necessary?
                        cv::Mat depth_4c;
                        cv::cvtColor(frame_ptr->getPrettyDepth(), depth_4c, CV_GRAY2BGRA);

                        std::vector<cv::Mat> v;
                        v.push_back(color_4c);
                        v.push_back(depth_4c);
                        v.push_back(render_color);
                        v.push_back(render_normals);

                        cv::Mat input_and_render = createMxN(2,2,v);
                        cv::imshow("input_and_render", input_and_render);
                        fs::path filename = folder_save_images / (boost::format("input_and_render_%05d.png") % volume_modeler->getFramesAdded()).str();
                        cv::imwrite(filename.string(), input_and_render);
                    }
                }
            }

            if (params.grid.debug_render) {
                std::vector<cv::Mat> debug_render_images = volume_modeler->getDebugRenderImages();
                for (size_t i = 0; i < debug_render_images.size(); ++i) {
                    cv::imshow("debug_render", debug_render_images[i]);
                    int this_key = cv::waitKey(0);
                    if (key < 0) key = this_key;
                }

            }

            if (align_debug_images) {
                const static bool pyramid_images = false;
                if (pyramid_images){
                    std::vector<cv::Mat> pyramid_images;
                    volume_modeler->getPyramidDebugImages(pyramid_images);
                    // todo: show these better
                    for (size_t i = 0; i < pyramid_images.size(); ++i) {
                        std::string window_name = window_pyramid_debug_images + "_" + boost::lexical_cast<std::string>(i);
                        cv::imshow(window_name, pyramid_images[i]);
                    }
                }

                std::vector<cv::Mat> debug_images;
                volume_modeler->getAlignDebugImages(debug_images);
                for (size_t i = 0; i < debug_images.size(); ++i) {
                    if (save_images) {
                        fs::path filename = folder_save_images / (boost::format("align_debug_images_%05d_%05d.png") % volume_modeler->getFramesAdded() % i).str();
                        cv::imwrite(filename.string(), debug_images[i]);
                    }
                    else {
                        cv::imshow(window_align_debug_images, debug_images[i]);
                        int this_key = cv::waitKey(1000/20);
                        if (key < 0) key = this_key;
                    }
                }
            }
            else {
                cv::destroyWindow(window_align_debug_images);
            }

            if (params.volume_modeler.verbose) {
                cout << "Volume modeler info before (possibly) loop closure:" << endl;
                cout << volume_modeler->getSummaryString() << endl;
            }

            // loop closure!
            if (volume_success && params.loop_closure.loop_closure) {
                // right before loop closure

                // awkward: now only show loop closure debug images on specifically that variable
                volume_modeler->setAlignDebugImages(loop_closure_debug_images);

                bool loop_success = volume_modeler->loopClosure(*frame_ptr);

                // after loop closure
                volume_modeler->setAlignDebugImages(align_debug_images); // reset

                cv::Mat render_color, render_normals;
                volume_modeler->getLastRenderPretty(render_color, render_normals);
                if (!params.volume_modeler.command_line_interface) {
                    //cv::imshow("render_color_loop", render_color);
                    //cv::imshow("render_normals_loop", render_normals);
                    cv::Mat render_loop_both = create1x2(render_color, render_normals);
                    cv::imshow("render_loop_both", render_loop_both);
                    if (save_images) {
                        fs::path filename = folder_save_images / (boost::format("render_loop_both_%05d.png") % volume_modeler->getFramesAdded()).str();
                        cv::imwrite(filename.string(), render_loop_both);
                    }
                }

                if (loop_success) {
                    if (loop_closure_debug_images) {
                        std::vector<cv::Mat> debug_images;
                        volume_modeler->getAlignDebugImages(debug_images);
                        for (size_t i = 0; i < debug_images.size(); ++i) {
                            cv::imshow(window_loop_closure_debug_images, debug_images[i]);
                            int this_key = cv::waitKey(1000/20);
                            if (key < 0) key = this_key;
                        }
                    }
                    else {
                        cv::destroyWindow(window_loop_closure_debug_images);
                    }

                    if (loop_closure_save_meshes_always) {
                        cout << "loop_closure_save_meshes_always..." << endl;
                        fs::path folder_for_frame = (boost::format("%05d_loop") % volume_modeler->getFramesAdded()).str();
                        volume_modeler->generateAndSaveAllMeshes(params.volume_modeler.output / folder_for_frame);
                        cout << "done with loop_closure_save_meshes_always" << endl;
                    }

                    if (pause_after_loop_closure) {
                        pause_after_this_frame = true;
                    }
                }

                // summary at the end
                if (params.volume_modeler.verbose) {
                    cout << "Volume modeler info after loop closure:" << endl;
                    cout << volume_modeler->getSummaryString() << endl;
                }
            }

            if (params.volume_modeler.verbose) {
                if (volume_success) {
                    cout << "Added frame: " << volume_modeler->getFramesAdded() << endl;
                }
                else {
                    cout << "FAILED to add frame" << endl;
                }
            }

            // any other debug images...
            if (!params.volume_modeler.command_line_interface && !suppress_debug_images) {
                std::map<std::string, cv::Mat> debug_images = volume_modeler->getDebugImages();
                for (std::map<std::string, cv::Mat>::const_iterator iter = debug_images.begin(); iter != debug_images.end(); ++iter) {
                    cv::imshow(iter->first, iter->second);
#if 0
                    // grab ones to save here?
                    // again, probably a better way to do this...
                    if (!save_images.empty()) {
                        if (iter->first == "segments") {
                            fs::path filename = params.volume_modeler.output / save_images / (boost::format("segments_%05d.png") % volume_modeler->getFramesAdded()).str();
                            cv::imwrite(filename.string(), iter->second);
                        }
                    }
#endif
                    if (save_images) {
                        fs::path filename = folder_save_images / (boost::format("%s_%05d.png") % iter->first % volume_modeler->getFramesAdded()).str();
                        cv::imwrite(filename.string(), iter->second);
                    }
                }
            }

            // get the camera poses associated with input filenames
            if (volume_success) {
                std::string last_frame_filename;
                bool got_filename = frame_provider->getLastFilename(last_frame_filename);
                if (frame_provider->getLastFilename(last_frame_filename)) {
                    std::string line_for_frame = (boost::format("%s %s") % last_frame_filename % EigenUtilities::transformToString(volume_modeler->getLastCameraPose())).str();
                    filename_to_pose_list.push_back(line_for_frame);
                }
            }

        } // done adding frame to volume

        if (test_nvidia_gpu_memory) {
            int total, available;
            volume_modeler->getNvidiaGPUMemoryUsage(total, available);
            cout << "test_nvidia_gpu_memory: " << "Total: " << total << " Available: " << available << endl;
        }

        if (test_set_weights >= 0) {
            volume_modeler->setMaxWeightInVolume(test_set_weights);
        }

        if (debug_add_volume) {
            Eigen::Affine3f last_camera_pose = volume_modeler->getLastCameraPose();

            std::vector<cv::Mat> debug_image_v;
            {
                // render the original
                volume_modeler->render(last_camera_pose);
                cv::Mat color, depth;
                volume_modeler->getLastRenderPretty(color, depth);
                //cv::Mat both = create1x2(color, depth);
                //cv::imshow("debug_add_volume true", both);
                debug_image_v.push_back(color);
                debug_image_v.push_back(depth);
            }

            // update alternate
            alternate_volume_modeler->addFrame(*frame_ptr, last_camera_pose);

            // should do this:
            alternate_volume_modeler->debugAddVolume();

            // render for comparison?
            // need pose to do so...
            {
                alternate_volume_modeler->render(last_camera_pose);
                cv::Mat color, depth;
                alternate_volume_modeler->getLastRenderPretty(color, depth);
                //both = create1x2(color, depth);
                //cv::imshow("debug_add_volume alternate", both);
                debug_image_v.push_back(color);
                debug_image_v.push_back(depth);
            }

            cv::Mat debug_image = createMxN(2,2,debug_image_v);
            cv::imshow("debug_add_volume", debug_image);

            if (save_images) {
                fs::path filename = folder_save_images / (boost::format("debug_add_volume_%05d.png") % volume_modeler->getFramesAdded()).str();
                cv::imwrite(filename.string(), debug_image);
            }
        }

        ///////
        // end of frame activities
        if (params.volume_modeler.verbose) cout << "TIME entire frame: " << t_frame.elapsed() << endl;


#ifdef VOLUME_MODELER_GLFW
        if (volume_modeler_glfw) {
            if (glfw_mesh_every_n > 0) {
                int frames_added = volume_modeler->getFramesAdded();
                if (frames_added > 0 && frames_added % glfw_mesh_every_n == 0) {
                    MeshPtr mesh_ptr (new Mesh);
                    volume_modeler->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
                    volume_modeler_glfw->updateMesh(params.glfw_keys.mesh, mesh_ptr);
                }
            }

            if (read_glfw_buffer) {
                // should I sleep to "make sure" that it renders?
                // I should have a "wait for next render"...
                // in fact, readBuffer should "wait for next render"
                cv::Mat read_buffer = volume_modeler_glfw->readBuffer();
                if (!read_buffer.empty()) {
                    cv::imshow("read_buffer", read_buffer);
                    if (save_images) {
                        fs::path filename = folder_save_images / (boost::format("glfw_%05d.png") % volume_modeler->getFramesAdded()).str();
                        cv::imwrite(filename.string(), read_buffer);
                    }
                }
            }
        }
#endif

        if (save_mesh_every_n > 0) {
            int frames_added = volume_modeler->getFramesAdded();
            if (frames_added > 0 && frames_added % save_mesh_every_n == 0) {
                cout << "save_mesh_every_n: " << frames_added << endl;
                if (false) {
                    // could just do the simple one
                    fs::path file_for_frame = (boost::format("mesh_%05d.ply") % frames_added).str();
                    volume_modeler->generateAndSaveMesh(params.volume_modeler.output / file_for_frame);
                }
                else {
                    fs::path folder_for_frame = (boost::format("mesh_%05d") % frames_added).str();
                    volume_modeler->generateAndSaveAllMeshes(params.volume_modeler.output / folder_for_frame);
                    //volume_modeler->saveCameraPoses(params.volume_modeler.output / folder_for_frame / "camera_poses.txt");
                }

                cout << "Done with save_mesh_every_n" << endl;
            }
        }

        if (save_state_every_n > 0) {
            int frames_added = volume_modeler->getFramesAdded();
            if (frames_added > 0 && frames_added % save_state_every_n == 0) {
                cout << "save_state_every_n: " << frames_added << endl;
                fs::path folder_for_frame = (boost::format("save_state_%05d") % volume_modeler->getFramesAdded()).str();
                volume_modeler->save(params.volume_modeler.output / folder_for_frame);
            }
        }

        if (turntable_rotation_limit > 0) {
            // you might make this a function of the modeler...
            typedef boost::shared_ptr<Eigen::Affine3f> PosePtr;
            typedef std::vector<PosePtr > PosePtrList;
            PosePtrList camera_poses;
            volume_modeler->getAllCameraPoses(camera_poses);
            float accumulated_angle_degrees = 0;
            PosePtr previous_pose;
            for (PosePtrList::iterator iter = camera_poses.begin(); iter != camera_poses.end(); ++iter) {
                if (previous_pose) {
                    // do stuff
                    float angle, distance;
                    EigenUtilities::getCameraPoseDifference(*previous_pose, **iter, angle, distance);
                    accumulated_angle_degrees += angle;
                }
                previous_pose = *iter;
            }

            if (accumulated_angle_degrees > turntable_rotation_limit) {
                process_next_frame = false;
            }

        }

        if (pause_after_this_frame) {
            cout << "paused..." << endl;
            int this_key = cv::waitKey(0);
            if (key < 0) key = this_key;
        }
    } // loop over all input

    // special external filenames to camera poses
    if (!filename_to_pose_list.empty()) {
        fs::path folder = params.volume_modeler.output; // could be subfolder?
        fs::path filename = folder / "filename_and_pose_list.txt";
        std::fstream file(filename.string().c_str(), std::ios::out);
        BOOST_FOREACH(std::string const& s, filename_to_pose_list) {
            file << s << endl;
        }
        cout << "Saved filename_to_pose_list to: " << filename << endl;
    }

    // mesh and camera poses at the end of file input
    {
        boost::timer t;
        cout << "Generating mesh after frame loop..." << endl;
        static const fs::path folder = "mesh_final";
        volume_modeler->generateAndSaveAllMeshes(params.volume_modeler.output / folder);
        volume_modeler->saveCameraPoses(params.volume_modeler.output / folder / "camera_poses.txt");
        volume_modeler->saveGraphs(params.volume_modeler.output / folder);
        cout << "TIME generateAndSaveAllMeshes: " << t.elapsed() << endl;
    }
    cout << "Done with mesh" << endl;

    // state as well?
    if (save_state_at_end) {
        int frames_added = volume_modeler->getFramesAdded();
        fs::path folder_for_frame = (boost::format("save_state_at_end_%d") % volume_modeler->getFramesAdded()).str();
        volume_modeler->save(params.volume_modeler.output / folder_for_frame);
        cout << "Saved state to: " << folder_for_frame << endl;
    }


#ifdef VOLUME_MODELER_GLFW
    if (volume_modeler_glfw) {
        if (glfw_join) {
            // try to destroy useless opencv windows
            cout << "Attempting to cv::destroyAllWindows..." << endl;
            cv::destroyAllWindows();
            //cv::waitKey(1000); // still doesn't work with this in

            // put final mesh in the viewer
            cout << "final mesh to viewer..." << endl;
            if (glfw_join_full_mesh) {
                MeshPtr mesh_ptr (new Mesh);
                std::vector<bool> vertex_validity;
                std::vector<bool> triangle_validity;
                volume_modeler->generateMeshAndValidity(mesh_ptr->vertices, mesh_ptr->triangles, vertex_validity, triangle_validity);
                volume_modeler_glfw->updateMesh(params.glfw_keys.mesh, mesh_ptr);
            }
            else {
                MeshPtr mesh_ptr (new Mesh);
                volume_modeler->generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
                volume_modeler_glfw->updateMesh(params.glfw_keys.mesh, mesh_ptr);
            }
            cout << "joining..." << endl;
            volume_modeler_glfw->join();
        }
        else {
            volume_modeler_glfw->destroy();
        }
    }
#endif

    return 0;
}
