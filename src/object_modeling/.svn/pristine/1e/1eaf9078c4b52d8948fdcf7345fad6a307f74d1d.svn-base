// see Noise.cpp (dup)
float simpleAxial(float z) {
	float w = z - 0.4;
	return 0.0012 + 0.0019 * w * w;
}

// copied from TSDF.cl
float4 applyPoseToPoint(float16 pose, float4 point)
{
	point.s3 = 1.0f;
	float4 result;
	result.s0 = dot(pose.s0123, point);
	result.s1 = dot(pose.s4567, point);
	result.s2 = dot(pose.s89ab, point);
	result.s3 = 1.0f;
	return result;
}

// copied from TSDF.cl
float4 applyPoseToNormal(float16 pose, float4 normal)
{
	float4 result;
	normal.s3 = 0;
	result.s0 = dot(pose.s0123, normal);
	result.s1 = dot(pose.s4567, normal);
	result.s2 = dot(pose.s89ab, normal);
	result.s3 = 0.0f;
	return result;
}

int getImageIndex(int2 image_dims, int2 pixel)
{
	return pixel.s1 * image_dims.s0 + pixel.s0;
}

float4 depthToPoint(int2 pixel, float depth, float2 camera_c, float2 camera_f)
{
	float4 result = nan((uint4)0);
	if (depth > 0) {
		float2 pixel_float = convert_float2(pixel);
		result.s01 = (pixel_float - camera_c) * depth / camera_f;
		result.s2 = depth;
		result.s3 = 1;
	}
	return result;
}

__kernel void computeNormals(
	__global float4 *points,
	__global float4 *result_normals,
	__local float4 *local_points,
	const int2 image_dims,
	float max_sigmas
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);

	bool valid = all(global_index < image_dims);

	int2 local_index_logical = (int2)(get_local_id(0), get_local_id(1));
	int2 border_offset = (int2)(1,1);
	int2 local_index_with_border = local_index_logical + border_offset;

	int2 local_mem_dims;
	local_mem_dims.s0 = get_local_size(0) + 2;
	local_mem_dims.s1 = get_local_size(1) + 2;

	const float4 nan_point = nan((uint4)0);

	// load local memory
	local_points[getImageIndex(local_mem_dims, local_index_with_border)] = valid ? points[getImageIndex(image_dims, global_index)] : nan_point;
	int2 offset;
	if (get_local_id(0) == 0) {
		offset = (int2)(-1,0);
		local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)] = (valid && global_index.s0 > 0) ? points[getImageIndex(image_dims, global_index + offset)] : nan_point;
	}
	if (get_local_id(0) == get_local_size(0) - 1) {
		offset = (int2)(1,0);
		local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)] = (valid && global_index.s0 < image_dims.s0 - 1) ? points[getImageIndex(image_dims, global_index + offset)] : nan_point;
	}
	if (get_local_id(1) == 0) {
		offset = (int2)(0,-1);
		local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)] = (valid && global_index.s1 > 0) ? points[getImageIndex(image_dims, global_index + offset)] : nan_point;
	}
	if (get_local_id(1) == get_local_size(1) - 1) {
		offset = (int2)(0,1);
		local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)] = (valid && global_index.s1 < image_dims.s1 - 1) ? points[getImageIndex(image_dims, global_index + offset)] : nan_point;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute simple normal
	float4 normal_to_write = nan_point;
	offset = (int2)(0,0);
	float4 p = local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)];
	offset = (int2)(-1,0);
	float4 px0 = local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)];
	offset = (int2)(1,0);
	float4 px1 = local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)];
	offset = (int2)(0,-1);
	float4 py0 = local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)];
	offset = (int2)(0,1);
	float4 py1 = local_points[getImageIndex(local_mem_dims, local_index_with_border + offset)];
	// count on valid math of everything creating nans when nans arise?
	bool within_distance = true;
	float max_distance = max_sigmas * simpleAxial(p.s2);
	float max_distance_squared = max_distance * max_distance;
	float4 diff_vec;
	diff_vec = p - px0;
	within_distance = within_distance && dot(diff_vec, diff_vec) <= max_distance_squared;
	diff_vec = p - px1;
	within_distance = within_distance && dot(diff_vec, diff_vec) <= max_distance_squared;
	diff_vec = p - py0;
	within_distance = within_distance && dot(diff_vec, diff_vec) <= max_distance_squared;
	diff_vec = p - py1;
	within_distance = within_distance && dot(diff_vec, diff_vec) <= max_distance_squared;
	if (within_distance) {
		normal_to_write = normalize(cross(py1 - py0, px1 - px0));
	}

	// write result
	if (valid) {
		result_normals[getImageIndex(image_dims, global_index)] = normal_to_write;
	}
}

__kernel void smoothNormals(
	__global float4 *input_points,
	__global float4 *input_normals,
	__global float4 *result_normals,
	__local float4 *local_points,
	__local float4 *local_normals,
	const int2 image_dims,
	const float max_sigmas
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);

	bool valid = all(global_index < image_dims);

	int2 local_index_logical = (int2)(get_local_id(0), get_local_id(1));
	int2 border_offset = (int2)(1,1);
	int2 local_index_with_border = local_index_logical + border_offset;

	int2 local_mem_dims;
	local_mem_dims.s0 = get_local_size(0) + 2;
	local_mem_dims.s1 = get_local_size(1) + 2;

	const float4 nan_point = nan((uint4)0);

	// try local memory a different way 
	int local_pointer = getImageIndex(local_mem_dims, local_index_logical);
	int2 global_index_offset = global_index - border_offset;
	// always check global valid
	if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
		int global_pointer = getImageIndex(image_dims, global_index_offset);
		local_points[local_pointer] = input_points[global_pointer];
		local_normals[local_pointer] = input_normals[global_pointer];
	}
	else {
		local_points[local_pointer] = local_normals[local_pointer] = nan_point;
	}
	// wrap for rightmost
	if ( local_index_logical.s0 < 2 ) {
		int2 wrap_offset = (int2)(get_local_size(0),0);
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_points[local_pointer] = input_points[global_pointer];
			local_normals[local_pointer] = input_normals[global_pointer];
		}
		else {
			local_points[local_pointer] = local_normals[local_pointer] = nan_point;
		}
	}
	// wrap for bottommost
	if ( local_index_logical.s1 < 2 ) {
		int2 wrap_offset = (int2)(0, get_local_size(0));
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_points[local_pointer] = input_points[global_pointer];
			local_normals[local_pointer] = input_normals[global_pointer];
		}
		else {
			local_points[local_pointer] = local_normals[local_pointer] = nan_point;
		}
	}
	// wrap for bottom/right corner
	if ( local_index_logical.s0 < 2 && local_index_logical.s1 < 2 ) {
		int2 wrap_offset = (int2)(get_local_size(0), get_local_size(0));
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_points[local_pointer] = input_points[global_pointer];
			local_normals[local_pointer] = input_normals[global_pointer];
		}
		else {
			local_points[local_pointer] = local_normals[local_pointer] = nan_point;
		}
	}

	// make sure all local memory written
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 normal_to_write = nan_point;

	float4 p_center = local_points[getImageIndex(local_mem_dims, local_index_with_border)];
	float max_distance = max_sigmas * simpleAxial(p_center.s2);
	float max_distance_squared = max_distance * max_distance;

	// matching CPU code, only proceed if center point valid
	if (isfinite(p_center.s2)) {
		float4 normal_sum = (float4)0;
		bool any_valid_normals = false;
		for (int row_d = -1; row_d <= 1; row_d++) {
			for (int col_d = -1; col_d <= 1; col_d++) {
				int2 offset = (int2)(col_d,row_d);
				local_pointer = getImageIndex(local_mem_dims, local_index_with_border + offset);
				float4 p_other = local_points[local_pointer];
				bool use_other = true;
				float4 diff_vec = p_center - p_other;
				use_other = use_other && dot(diff_vec,diff_vec) <= max_distance_squared;
				float4 n_other = local_normals[local_pointer];
				use_other = use_other && isfinite(n_other.s2); // why does s0 vs s2 make a difference???
				if (use_other) {
					normal_sum += n_other;
					any_valid_normals = true;
				}
			}
		}
		if (any_valid_normals) {
			normal_to_write = normalize(normal_sum);
		}
	}

	// write result
	if (valid) {
		result_normals[getImageIndex(image_dims, global_index)] = normal_to_write;
	}
}

