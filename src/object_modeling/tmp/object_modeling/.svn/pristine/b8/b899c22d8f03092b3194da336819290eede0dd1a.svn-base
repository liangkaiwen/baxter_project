#pragma once

struct ParamsModelHistogram
{
	int bin_count;
	float min_value;
	float max_value;
	float debug_pick_pixel_depth_offset;
    int debug_points_along_ray;
	int mat_height;
    int mat_width_per_bin;
    int peak_finding_bin_range;

	ParamsModelHistogram()
		: bin_count(10),
		min_value(-0.1),
		max_value(0.1),
        debug_pick_pixel_depth_offset(0),
        debug_points_along_ray(0),
		mat_height(50),
        mat_width_per_bin(10),
        peak_finding_bin_range(2)
	{}
};
