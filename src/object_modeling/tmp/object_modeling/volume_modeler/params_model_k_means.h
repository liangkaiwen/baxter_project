#pragma once

#include "EigenUtilities.h"

struct ParamsModelKMeans
{
	int k;
	float minimum_relative_count;
    bool store_normals;
    float compatibility_add_max_angle_degrees;
    float compatibility_render_max_angle_degrees;
    bool debug_rendering;
    bool render_all_6;
    bool render_all_6_from_canonical;
	float render_all_6_from_canonical_distance;
    float render_all_6_scale;
    Eigen::Array4ub default_mesh_color;
	bool empty_always_included; // for addifcompatible (at least)
    bool debug_slices;
    int slice_axis;
    float slice_images_scale;
    float slice_color_max;
    float fixed_normal_view_length;
	bool cos_weight;
    bool debug_meshes;


	ParamsModelKMeans()
		: k(2),
        minimum_relative_count(0.f),
        store_normals(true),
        compatibility_add_max_angle_degrees(90),
        compatibility_render_max_angle_degrees(90),
        debug_rendering(false),
        render_all_6(false),
        render_all_6_from_canonical(false),
		render_all_6_from_canonical_distance(1),
        render_all_6_scale(1.f),
        default_mesh_color(Eigen::Array4ub::Constant(200)),
        empty_always_included(false),
        debug_slices(false),
        slice_axis(1),
        slice_images_scale(1),
        slice_color_max(0.01),
        fixed_normal_view_length(0.1),
        cos_weight(false),
        debug_meshes(false)
	{}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
