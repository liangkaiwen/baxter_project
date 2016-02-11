#pragma once

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

struct ParamsGrid
{
	int grid_size;
	int border_size;
	int max_mb_gpu;
	int max_mb_system;
	bool debug_render;
	bool debug_grid_motion;
	fs::path temp_folder;
	bool add_grids_in_frustum;
	bool grid_free;
	bool new_render;
    bool skip_single_mesh;

	ParamsGrid()
		: grid_size(32),
		border_size(2), // was 1
		max_mb_gpu(500),
		max_mb_system(-1),
		debug_render(false),
		debug_grid_motion(false),
		temp_folder("temp_grid"),
		add_grids_in_frustum(false),
		grid_free(false),
        new_render(false),
        skip_single_mesh(false)
	{}
};
