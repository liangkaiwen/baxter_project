#pragma once

struct ParamsVolume
{
	Eigen::Array3i cell_count;
	float cell_size;
	float max_weight_icp;
	float max_weight_color;
	bool use_most_recent_color;
    float min_truncation_distance; // half of full truncation range..need to make all models respect this...

	ParamsVolume()
        : cell_count(256,256,256),
		cell_size(0.01f),
		max_weight_icp(100),
		max_weight_color(100),
        use_most_recent_color(false),
        min_truncation_distance(0)
	{}
};
