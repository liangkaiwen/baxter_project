// see Noise.cpp (dup)
float simpleAxial(float z) {
	float w = z - 0.4;
	return 0.0012 + 0.0019 * w * w;
}
#define Z_MIN 0.4

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

// copied from TSDF.cl
float2 projectPoint(float2 f, float2 c, float4 point)
{
	return (point.s01 * f / point.s2 + c);
}

int getImageIndex(int2 image_dims, int x, int y)
{
	return  y * image_dims.s0 + x;
}

int getImageIndexChannels(int2 image_dims, int image_channels, int channel, int x, int y)
{
	return image_channels * (y * image_dims.s0 + x) + channel;
}

int getImageIndexList(int2 image_dims, int which_image, int x, int y)
{
	return image_dims.s0 * image_dims.s1 * which_image + y * image_dims.s0 + x;
}

// still taking image_channels as an argument though if getImageIndexList it's not needed
float interpolateImage(__global float *frame_image, int2 frame_image_dims, int image_channels, int channel, float2 p_proj)
{
	int2 p_proj_image_floor = convert_int2(p_proj);
	int i = p_proj_image_floor.s0;
	int j = p_proj_image_floor.s1;

	// note this is not bounds checked:
	#if 0
	float i00 = frame_image[getImageIndexChannels(frame_image_dims, image_channels, channel, i, j)];
	float i01 = frame_image[getImageIndexChannels(frame_image_dims, image_channels, channel, i, j+1)];
	float i10 = frame_image[getImageIndexChannels(frame_image_dims, image_channels, channel, i+1, j)];
	float i11 = frame_image[getImageIndexChannels(frame_image_dims, image_channels, channel, i+1, j+1)];
	#endif
	float i00 = frame_image[getImageIndexList(frame_image_dims, channel, i, j)];
	float i01 = frame_image[getImageIndexList(frame_image_dims, channel, i, j+1)];
	float i10 = frame_image[getImageIndexList(frame_image_dims, channel, i+1, j)];
	float i11 = frame_image[getImageIndexList(frame_image_dims, channel, i+1, j+1)];
	float2 offset = p_proj - convert_float2(p_proj_image_floor);
	float off_x = offset.s0;
	float off_y = offset.s1;
	float result =		  i00 * (1 - off_x) * (1 - off_y)
						+ i01 * (1 - off_x) * (off_y)
						+ i10 * (off_x) * (1 - off_y)
						+ i11 * (off_x) * (off_y);
	return result;
}

float interpolateImageSafe(__global float *frame_image, int2 frame_image_dims, int image_channels, int channel, float2 p_proj) 
{
	// redundant evaluation of this so I can call the same inner function
	int2 p_proj_image_floor = convert_int2(p_proj);
	int i = p_proj_image_floor.s0;
	int j = p_proj_image_floor.s1;

	// new: return 0 if any will be outside image bounds
    // todo: Interestingly this keeps the error function from getting vectorized!! (on the Intel CPU anyway...)
	if (i < 0 || j < 0 || i + 1 >= frame_image_dims.s0 || j + 1 >= frame_image_dims.s1) return 0;

	return interpolateImage(frame_image, frame_image_dims, image_channels, channel, p_proj);
}




////////////////////////////////////
// todo: split into icp and color separately...probably worth it for simplicity
// also worth it is separating out the reduction from the computation
// a bit slower, but will KEEP YOU SANE
__kernel void computeErrorAndGradientReduced(
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	__global float *rendered_image,
	__global float4 *frame_points,
	__global float4 *frame_normals,
	__global float *frame_image,
	__global float *frame_gradient_x,
	__global float *frame_gradient_y,
	__global float *frame_weights,
	__global float *result_reduced_LHS, // 21 upper diagonal elements per work-group.
	__global float *result_reduced_RHS, // 6 RHS elements per work-group.
	__local float *local_LHS,
	__local float *local_RHS,
	const int write_result_error_vector,
	__global float *result_error_vector,
	const int write_result_error_matrix,
	__global float *result_error_matrix,
	const int write_result_robust_weights,
	__global float *result_robust_weights,
	const int4 render_rect, // describes how to index into rendered
	const int2 frame_image_dims, // how to index into frame
	const float16 x_transform,
	const float max_distance,
	const float min_normal_dot_product,
	const float weight_icp,
	const float weight_color,
	const int image_channel_count,
	const float2 focal_lengths_render,
	const float2 camera_centers_render,
	const float2 focal_lengths_frame,
	const float2 camera_centers_frame,
	const int error_point_count,
	const int huber_icp,
	const int huber_color
	)
{
	unsigned int i = get_global_id(0);
	unsigned int tid = get_local_id(0);
	int error_channel_count = 1 + image_channel_count;
	const float4 empty_point = (float4)(0,0,0,0);

	////////// Make sure to set all local memory later!
	// init local memory to 0, so the last few straglers are set to 0
	for (int a = 0; a < 21; ++a) local_LHS[a * get_local_size(0) + tid] = 0;
	for (int a = 0; a < 6; ++a) local_RHS[a * get_local_size(0) + tid] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	// will be set to 0 by anything that declares this point invalid
	// It's ok to keep processing such points as long as they fall within array bounds
	// will check valid before accessing any global arrays
	// TODO: use global_in_bounds where safe...
	bool global_in_bounds = (i < error_point_count);
	bool valid = global_in_bounds;

	/////////// local set to 0
	// store ICP and color errors together
	// assumes no more than 3 color error channels
	#define MAX_ERROR_CHANNELS 3
	float this_error_array[MAX_ERROR_CHANNELS];
	float this_weight_array[MAX_ERROR_CHANNELS];
	for (int c = 0; c < MAX_ERROR_CHANNELS; ++c) {
		this_error_array[c] = 0;
		this_weight_array[c] = 0;
	}
	float this_gradient_array[MAX_ERROR_CHANNELS][6];
	for (int c = 0; c < MAX_ERROR_CHANNELS; ++c) {
		for (int col = 0; col < 6; ++col) {
			this_gradient_array[c][col] = 0;
		}
	}

	
	//////// global set to 0
	// must fill with zeros initially?
	// NOPE!  TODO: keep "in bounds" a separate idea from "valid", then just do this once after computation
	// This is a relic from when you used to bomb out early instead of reducing
	// NOTE: images are based on error_point_count, NOT get_global_size
	if (valid) {
		if (write_result_error_vector) {
			for (int c = 0; c < error_channel_count; ++c) { 
				result_error_vector[c*error_point_count + i] = 0;
			}
		}
		if (write_result_error_matrix) {
			for (int c = 0; c < error_channel_count; ++c) {
				for (size_t col = 0; col < 6; col++) {
					result_error_matrix[c*6*error_point_count + col*error_point_count + i] = 0;
				}
			}
		}
		if (write_result_robust_weights) {
			for (int c = 0; c < error_channel_count; ++c) {
				result_robust_weights[c*error_point_count + i] = 0;
			}
		}
	}

	//////// associate
	float4 rendered_point = valid ? rendered_points[i] : empty_point;
	float4 rendered_normal = valid ? rendered_normals[i] : empty_point;
	if (!isfinite(rendered_point.z) || !isfinite(rendered_normal.z)) valid = false;
	float4 p = applyPoseToPoint(x_transform, rendered_point);
	float4 p_normal = applyPoseToNormal(x_transform, rendered_normal);
	float2 p_proj = projectPoint(focal_lengths_frame, camera_centers_frame, p);
	// check projection within object rect of frame
	int2 p_proj_floor = convert_int2_rtn(p_proj); // to negative infinity (always down)
	//if (any(p_proj_floor < object_rect.s01) || any(p_proj_floor + 1 >= object_rect.s01 + object_rect.s23)) valid = false;
	if (any(p_proj_floor < (int2)(0)) || any(p_proj_floor + (int2)(1) >= frame_image_dims)) valid = false;
	// do nearest neighbor for point
	int2 p_proj_int = convert_int2_rte(p_proj);
	int frame_index = getImageIndex(frame_image_dims, p_proj_int.x, p_proj_int.y);
	float4 frame_point = valid ? frame_points[frame_index] : empty_point;
	if (!isfinite(frame_point.z)) valid = false;
	// end associate

	// need these for gradient:
	float x_2 = 2*p.x;
	float y_2 = 2*p.y;
	float z_2 = 2*p.z;
	// 3x6
	float j_3d_x[18] = {1,0,0, 0   , z_2,-y_2,
						0,1,0, -z_2, 0  , x_2,
						0,0,1,  y_2,-x_2, 0};

	#define HUBER_COLOR_E 0.1 // parameter (0.02 problem, 
	#define HUBER_ICP_E 0.01 // parameter
	#define WEIGHT_BY_DISTANCE true
	#define OUTLIER_BY_DISTANCE false // was false
	#define OUTLIER_BY_DISTANCE_SIGMAS 12 // was 6 (only relevant if OUTLIER_BY_DISTANCE)

	//////////////////
	// ICP does additional filtering (filterForICP)
	bool icp_valid = false;
	float4 frame_normal = valid ? frame_normals[frame_index] : empty_point;
	float4 icp_vector = frame_point - p;
	float max_distance_to_use = OUTLIER_BY_DISTANCE ? OUTLIER_BY_DISTANCE_SIGMAS * simpleAxial(frame_point.s2) : max_distance;
	if (length(icp_vector) <= max_distance_to_use && isfinite(frame_normal.z)) {
		float dot_prod = dot(frame_normal, p_normal);
		if (dot_prod >= min_normal_dot_product) {
			icp_valid = true;
			// this weight carries through and affects color as well
			//this_weight_array[0] = 1;
			this_weight_array[0] = frame_weights[frame_index];
			float icp_error = dot(icp_vector, p_normal);

			// get the weight by distance
			if (WEIGHT_BY_DISTANCE) {
				float distance_weight_icp = simpleAxial(Z_MIN) / simpleAxial(frame_point.s2);
				this_weight_array[0] *= distance_weight_icp;
			}

			// huber
			// MAYBE YOU SHOULD CHANGE weight_icp AND TRANSFER TO COLOR
			// weight_icp should be the overall 10/1 thingy
			// should separately keep the huber downweight from icp and apply to color
			if (huber_icp) {
				float squared_icp_error = icp_error * icp_error;
				float huber_weight = 1.f;
				if (squared_icp_error > HUBER_ICP_E * HUBER_ICP_E) {
					huber_weight = HUBER_ICP_E / sqrt(squared_icp_error);
				}
				// because we apply this to both the error and the jacobian row, need to sqrt so it comes out right for the normal eq
				this_weight_array[0] *= sqrt(huber_weight);
			}

			// set error part
			float this_error_icp = this_weight_array[0] * weight_icp * icp_error; 
			this_error_array[0] = valid ? this_error_icp : 0;

			// gradient part
			// set the ICP jacobian row
			float j_rot_x[18] = {0,0,0,  0  , z_2,-y_2,
								 0,0,0, -z_2, 0  , x_2,
								 0,0,0,  y_2,-x_2, 0};
			// Eigen::MatrixXf jacobian_row_icp_unweighted = - ( n_3d.transpose() * j_3d_x ) + ( (p_frame - p_3d).transpose() * j_rot_x );
			for (size_t col = 0; col < 6; col++) {
				float to_store = - ( p_normal.s0 * j_3d_x[col+0] + p_normal.s1 * j_3d_x[col+6] + p_normal.s2 * j_3d_x[col+12] );
				to_store += icp_vector.s0 * j_rot_x[col+0] + icp_vector.s1 * j_rot_x[col+6] + icp_vector.s2 * j_rot_x[col+12];
				to_store *= this_weight_array[0] * weight_icp;
				this_gradient_array[0][col] = valid ? to_store : 0;
			}
		}
	}


	///////////////////
	// Color
	// only if icp valid?
	if (icp_valid) {
		float inverse_depth = 1.0f / p.z;
		// 2x3
		float4 j_proj_3d_r0 = (float4) ( (focal_lengths_frame.s0 * inverse_depth), (0), (-p.x * focal_lengths_frame.s0 * inverse_depth * inverse_depth), 0 );
		float4 j_proj_3d_r1 = (float4) ( (0), (focal_lengths_frame.s1 * inverse_depth), (-p.y * focal_lengths_frame.s1 * inverse_depth * inverse_depth), 0 );


		for (int c = 0; c < image_channel_count; ++c) {
			// error part
			float frame_image_value = valid ? interpolateImage(frame_image, frame_image_dims, image_channel_count, c, p_proj) : 0;
			float rendered_image_value = valid ? rendered_image[render_rect.s2 * render_rect.s3 * c + i] : 0;
			float image_error = (frame_image_value - rendered_image_value);
			
			// If you wanted to actually compute the error, you'd use this:
			#if 0
			float image_error_robust = image_error;
			if (huber_color) {
				if (image_error > HUBER_COLOR_E) {
					image_error_robust = sqrt(2 * HUBER_COLOR_E * image_error - HUBER_COLOR_E * HUBER_COLOR_E);
				}
				else if (image_error < -HUBER_COLOR_E) {
					image_error_robust = -sqrt(2 * HUBER_COLOR_E * -image_error - HUBER_COLOR_E * HUBER_COLOR_E);
				}
			}
			#endif

			// Instead:
			//this_weight_array[c+1] = 1;
			// start with weight from icp (new)
			this_weight_array[c+1] = this_weight_array[0];
			if (huber_color) {
				float squared_image_error = image_error * image_error;
				float huber_weight_color = 1;
				if (squared_image_error > HUBER_COLOR_E * HUBER_COLOR_E) {
					huber_weight_color = HUBER_COLOR_E / sqrt(squared_image_error);
				}
				// because we apply this to both the error and the jacobian row, need to sqrt so it comes out right for the normal eq
				this_weight_array[c+1] *= sqrt(huber_weight_color);
			}

			// set the error part:
			float this_error_color = this_weight_array[c+1] * weight_color * image_error;
			this_error_array[c+1] = valid ? this_error_color : 0;

			// gradient part:
			// 1x2
			float2 j_error_proj = valid ? 
				(float2) ( interpolateImage(frame_gradient_x, frame_image_dims, image_channel_count, c, p_proj),
						   interpolateImage(frame_gradient_y, frame_image_dims, image_channel_count, c, p_proj) ) : 0;

			// Eigen::MatrixXf jacobian_row_color = weight_color * j_error_proj * j_proj_3d * j_3d_x;
			// 1x3 (j_error_proj * j_proj_3d)
			float4 j_error_3d = (float4) ( (j_error_proj.s0 * j_proj_3d_r0.s0 + j_error_proj.s1 * j_proj_3d_r1.s0),
										   (j_error_proj.s0 * j_proj_3d_r0.s1 + j_error_proj.s1 * j_proj_3d_r1.s1),
										   (j_error_proj.s0 * j_proj_3d_r0.s2 + j_error_proj.s1 * j_proj_3d_r1.s2),
										   0 );

			for (size_t col = 0; col < 6; col++) {
				// dupish:
				float to_store = j_error_3d.s0 * j_3d_x[col+0] + j_error_3d.s1 * j_3d_x[col+6] + j_error_3d.s2 * j_3d_x[col+12];
				to_store *= this_weight_array[c+1] * weight_color;
				this_gradient_array[c+1][col] = valid ? to_store : 0;
			}
		}
	}

	//// write results to global memory if requested
	if (valid) {
		if (write_result_error_vector) {
			for (int c = 0; c < error_channel_count; ++c) {
				result_error_vector[c*error_point_count + i] = this_error_array[c];
			}
		}
		if (write_result_error_matrix) {
			// by error_channel, column, row (to coallesce memory accesses)
			for (int c = 0; c < error_channel_count; ++c) {
				for (size_t col = 0; col < 6; col++) {
					result_error_matrix[c*6*error_point_count + col*error_point_count + i] = this_gradient_array[c][col];
				}
			}
		}
		if (write_result_robust_weights) {
			for (int c = 0; c < error_channel_count; ++c) {
				result_robust_weights[c*error_point_count + i] = this_weight_array[c];
			}
		}
	}

	//////////////////
	// By use of "valid?" we should have correct values for this_gradient_array and this_error_array
	// in particular 0's for invalid points
	// and we need to make sure all of the following code runs for all threads

	////////////////////////////////////
	// reduce:

	// fill in "my" local memory
	// since this is memory specific to THIS error point, need to zero if not valid
	// See beginning of the kernel to see if I'm still setting local memory to zero up there...
	// LHS
	int LHS_counter = 0; // index to pack in the upper triangle...must end at 21
	for (int row = 0; row < 6; ++row) {
		for (int col = row; col < 6; ++col) {
			// need to do dot products of my values
			float dot_product = 0;
			for (int c = 0; c < error_channel_count; ++c) {
				dot_product += this_gradient_array[c][row] * this_gradient_array[c][col];
			}
			local_LHS[LHS_counter * get_local_size(0) + tid] = valid ? dot_product : 0;
			LHS_counter++;
		}
	}

	// RHS
	for (int col = 0; col < 6; ++col) {
		float dot_product = 0;
		for (int c = 0; c < error_channel_count; ++c) {
			dot_product += this_gradient_array[c][col] * this_error_array[c];
		}
		local_RHS[col * get_local_size(0) + tid] = valid ? dot_product : 0;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
	for(unsigned int s = get_local_size(0)/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			// loop over all LHS things
			for (int a = 0; a < 21; ++a) {
				local_LHS[a * get_local_size(0) + tid] += local_LHS[a * get_local_size(0) + tid + s];
				//barrier(CLK_LOCAL_MEM_FENCE);//remove (but these did change things!?)
			}
			// and over RHS things
			for (int a = 0; a < 6; ++a) {
				local_RHS[a * get_local_size(0) + tid] += local_RHS[a * get_local_size(0) + tid + s];
				//barrier(CLK_LOCAL_MEM_FENCE);//remove (but these did change things!?)
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) {
		// loop over all LHS things
		for (int a = 0; a < 21; ++a) {
			result_reduced_LHS[a * get_num_groups(0) + get_group_id(0)] = local_LHS[a * get_local_size(0)];
		}
		// and over RHS things
		for (int a = 0; a < 6; ++a) {
			result_reduced_RHS[a * get_num_groups(0) + get_group_id(0)] = local_RHS[a * get_local_size(0)];
		}
	}
}

// need to reduce the results from single kernel
__kernel void reduceErrorAndGradient(__global float *LHS_in, __global float *RHS_in, __global float *LHS_out, __global float *RHS_out, __local float* local_LHS, __local float* local_RHS, int rows)
{
	// load shared mem
	unsigned int tid = get_local_id(0);
	unsigned int i = get_global_id(0);
	
	// fill in "my" local memory
	// LHS
	for (int col = 0; col < 21; ++col) {
		local_LHS[col * get_local_size(0) + tid] = (i < rows) ? LHS_in[col * rows + i] : 0;
	}
	// RHS
	for (int col = 0; col < 6; ++col) {
		local_RHS[col * get_local_size(0) + tid] = (i < rows) ? RHS_in[col * rows + i] : 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
	for(unsigned int s=get_local_size(0)/2; s>0; s>>=1) 
	{
		if (tid < s) 
		{
			// LHS
			for (int col = 0; col < 21; ++col) {
				local_LHS[col * get_local_size(0) + tid] += local_LHS[col * get_local_size(0) + tid + s];
			}
			// RHS
			for (int col = 0; col < 6; ++col) {
				local_RHS[col * get_local_size(0) + tid] += local_RHS[col * get_local_size(0) + tid + s];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global mem
	if (tid == 0) {
		// LHS
		for (int col = 0; col < 21; ++col) {
			LHS_out[col * get_num_groups(0) + get_group_id(0)] = local_LHS[col * get_local_size(0)];
		}
		// RHS
		for (int col = 0; col < 6; ++col) {
			RHS_out[col * get_num_groups(0) + get_group_id(0)] = local_RHS[col * get_local_size(0)];
		}
	}
}
