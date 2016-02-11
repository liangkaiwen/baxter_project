int getImageIndex(int2 image_dims, int2 pixel)
{
	return pixel.s1 * image_dims.s0 + pixel.s0;
}

float getY(float4 pixel)
{
	// rgba:
	//return (1.f / 255) * dot( (float4)(0.299f, 0.587f, 0.114f, 0), pixel );
	// bgra:
	return (1.f / 255) * dot( (float4)(0.114f, 0.587f, 0.299f, 0), pixel );
}

// from http://stackoverflow.com/questions/6973115/opencv-convert-rgb-to-ycbcr-and-vice-versa-by-hand-visual-c
// when I was searching wrongly for ycbcr!!
float getCb(float4 pixel)
{
	// rgba
	//return (1.f / 255) * (128 + dot( (float4)(-0.168736f, -0.331264f, 0.5f, 0), pixel ) );
	// bgra
	return (1.f / 255) * (128 + dot( (float4)(0.5f, -0.331264f, -0.168736f, 0), pixel ) );
}

float getCr(float4 pixel)
{
	// rgba
	//return (1.f / 255) * (128 + dot( (float4)(0.5f, -0.418688f, -0.081312f, 0), pixel ) );
	// bgra
	return (1.f / 255) * (128 + dot( (float4)(-0.081312f, -0.418688f, 0.5f, 0), pixel ) );
}
	

__kernel void extractYFloat(
	__global uchar4 *input_uchar4,
	__global float *output_float1,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);
	uchar4 input_pixel = input_uchar4[image_index];
	output_float1[image_index] = getY(convert_float4(input_pixel));
}

__kernel void extractCrFloat(
	__global uchar4 *input_uchar4,
	__global float *output_float1,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);
	uchar4 input_pixel = input_uchar4[image_index];
	output_float1[image_index] = getCr(convert_float4(input_pixel));
}

__kernel void extractCbFloat(
	__global uchar4 *input_uchar4,
	__global float *output_float1,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);
	uchar4 input_pixel = input_uchar4[image_index];
	output_float1[image_index] = getCb(convert_float4(input_pixel));
}



// old interlaced style y, cr, cb, ...
__kernel void extractYCrCbFloat(
	__global uchar4 *input_uchar4,
	__global float *output_float,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);
	uchar4 input_pixel = input_uchar4[image_index];
	float4 input_pixel_float = convert_float4(input_pixel);
	float y_float = getY(input_pixel_float);
	float cr_float = getCr(input_pixel_float);
	float cb_float = getCb(input_pixel_float);
	output_float[3 * image_index] = y_float;
	output_float[3 * image_index + 1] = cr_float;
	output_float[3 * image_index + 2] = cb_float;
}

__kernel void splitFloat3(
	__global float *input_floats,
	__global float *output_float1,
	__global float *output_float2,
	__global float *output_float3,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);

	output_float1[image_index] = input_floats[3 * image_index];
	output_float2[image_index] = input_floats[3 * image_index + 1];
	output_float3[image_index] = input_floats[3 * image_index + 2];
}

__kernel void mergeFloat3(
	__global float *input_float1,
	__global float *input_float2,
	__global float *input_float3,
	__global float *output_floats,
	const int2 image_dims
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, global_index);

	output_floats[3 * image_index] = input_float1[image_index];
	output_floats[3 * image_index + 1] = input_float2[image_index];
	output_floats[3 * image_index + 2] = input_float3[image_index];
}

#if 0
// untested
__kernel void convolutionFloat(
	__global float *input_float,
	__global float *output_float,
	const int2 image_dims,
	__constant float * filter_kernel,
	const int kernel_side,
	__local float *local_input
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);

	bool valid = all(global_index < image_dims);

	int2 kernel_dims = (int2)kernel_side;
	int half_kernel_side = kernel_side / 2;
	int kernel_size_less_1 = kernel_side - 1;

	int2 local_index_logical = (int2)(get_local_id(0), get_local_id(1));
	int2 border_offset = (int2)(half_kernel_side,half_kernel_side);
	int2 local_index_with_border = local_index_logical + border_offset;

	int2 local_mem_dims;
	local_mem_dims.s0 = get_local_size(0) + kernel_size_less_1;
	local_mem_dims.s1 = get_local_size(1) + kernel_size_less_1;

	// initialize local memory
	int local_pointer = getImageIndex(local_mem_dims, local_index_logical);
	int2 global_index_offset = global_index - border_offset;
	// always check global valid
	// note that this currently sets all invalid to 0
	// probably want to extend border
	// but I'm not smart enough right now
	if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
		int global_pointer = getImageIndex(image_dims, global_index_offset);
		local_input[local_pointer] = input_float[global_pointer];
	}
	else {
		local_input[local_pointer] = 0;
	}
	// wrap for rightmost
	if ( local_index_logical.s0 < kernel_size_less_1) {
		int2 wrap_offset = (int2)(get_local_size(0),0);
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_input[local_pointer] = input_float[global_pointer];
		}
		else {
			local_input[local_pointer] = 0;
		}
	}
	// wrap for bottommost
	if ( local_index_logical.s1 < kernel_size_less_1) {
		int2 wrap_offset = (int2)(0, get_local_size(0));
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_input[local_pointer] = input_float[global_pointer];
		}
		else {
			local_input[local_pointer] = 0;
		}
	}
	// wrap for bottom/right corner
	if ( local_index_logical.s0 < kernel_size_less_1 && local_index_logical.s1 < kernel_size_less_1 ) {
		int2 wrap_offset = (int2)(get_local_size(0), get_local_size(0));
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;
		if ( valid && all(global_index_offset >= (int2)0) && all(global_index_offset < image_dims) ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_input[local_pointer] = input_float[global_pointer];
		}
		else {
			local_input[local_pointer] = 0;
		}
	}

	// make sure all local memory written
	barrier(CLK_LOCAL_MEM_FENCE);

	if (valid) {
		float result = 0;
		for (int row_d = -half_kernel_side; row_d <= half_kernel_side; ++row_d) {
			for (int col_d = -half_kernel_side; col_d <= half_kernel_side; ++col_d) {
				int2 offset = (int2)(col_d,row_d);
				local_pointer = getImageIndex(local_mem_dims, local_index_with_border + offset);
				int kernel_pointer = getImageIndex(kernel_dims, border_offset + offset); 
				result += local_input[local_pointer] * filter_kernel[kernel_pointer];
			}
		}

		output_float[getImageIndex(image_dims, global_index)] = result;
	}
}
#endif

__kernel void convolutionFloatHorizontal(
	__global float *input_float,
	__global float *output_float,
	const int2 image_dims,
	__constant float * filter_kernel,
	const int kernel_side,
	__local float *local_input
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);

	// can go ahead and fill in local memory even if !valid
	bool valid = all(global_index < image_dims);

	int half_kernel_side = kernel_side / 2;
	int kernel_size_less_1 = kernel_side - 1;
	int2 border_offset = (int2)(half_kernel_side, 0);
	int2 local_index_logical = (int2)(get_local_id(0), get_local_id(1));
	int2 local_index_with_border = local_index_logical + border_offset;

	// initialize local memory
	// use the same local mem as if 2d array (16x16 workgroup size)
	int2 local_mem_dims;
	local_mem_dims.s0 = get_local_size(0) + kernel_size_less_1;
	local_mem_dims.s1 = get_local_size(1) + kernel_size_less_1;
	int local_pointer = getImageIndex(local_mem_dims, local_index_logical);
	int2 global_index_offset = global_index - border_offset;

	// possible to be left, right, down OOB
	bool not_too_left = global_index_offset.s0 >= 0;
	bool not_too_right = global_index_offset.s0 < image_dims.s0;
	bool not_too_down = global_index_offset.s1 < image_dims.s1;
	if ( not_too_left && not_too_right && not_too_down ) {
		int global_pointer = getImageIndex(image_dims, global_index_offset);
		local_input[local_pointer] = input_float[global_pointer];
	}
	else if ( not_too_left && not_too_down) {
		// must be too right and not too down
		int global_pointer = getImageIndex(image_dims, (int2) (image_dims.s0 - 1, global_index_offset.s1) );
		local_input[local_pointer] = input_float[global_pointer];
	}
	else if ( not_too_right && not_too_down) {
		// must be too left and not too down
		int global_pointer = getImageIndex(image_dims, (int2) (0, global_index_offset.s1) );
		local_input[local_pointer] = input_float[global_pointer];
	}
	else {
		// must be too down
		local_input[local_pointer] = 0;
	}

	// wrap for rightmost
	if ( local_index_logical.s0 < kernel_size_less_1) {
		int2 wrap_offset = (int2)(get_local_size(0),0);
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;

		//// dup:
		bool not_too_left = global_index_offset.s0 >= 0;
		bool not_too_right = global_index_offset.s0 < image_dims.s0;
		bool not_too_down = global_index_offset.s1 < image_dims.s1;
		if ( not_too_left && not_too_right && not_too_down ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_input[local_pointer] = input_float[global_pointer];
		}
		else if ( not_too_left && not_too_down) {
			// must be too right and not too down
			int global_pointer = getImageIndex(image_dims, (int2) (image_dims.s0 - 1, global_index_offset.s1) );
			local_input[local_pointer] = input_float[global_pointer];
		}
		else if ( not_too_right && not_too_down) {
			// must be too left and not too down
			int global_pointer = getImageIndex(image_dims, (int2) (0, global_index_offset.s1) );
			local_input[local_pointer] = input_float[global_pointer];
		}
		else {
			// must be too down
			local_input[local_pointer] = 0;
		}
	}


	// make sure all local memory written
	barrier(CLK_LOCAL_MEM_FENCE);

	if (valid) {
		float result = 0;
		for (int d = -half_kernel_side; d <= half_kernel_side; ++d) {
			int2 offset = (int2)(d, 0);
			local_pointer = getImageIndex(local_mem_dims, local_index_with_border + offset);
			result += local_input[local_pointer] * filter_kernel[half_kernel_side + d];
		}

		output_float[getImageIndex(image_dims, global_index)] = result;
	}
}


__kernel void convolutionFloatVertical(
	__global float *input_float,
	__global float *output_float,
	const int2 image_dims,
	__constant float * filter_kernel,
	const int kernel_side,
	__local float *local_input
	)
{
	int2 global_index;
	global_index.s0 = get_global_id(0);
	global_index.s1 = get_global_id(1);

	// can go ahead and fill in local memory even if !valid
	bool valid = all(global_index < image_dims);

	int half_kernel_side = kernel_side / 2;
	int kernel_size_less_1 = kernel_side - 1;
	int2 border_offset = (int2)(0, half_kernel_side); // dup change
	int2 local_index_logical = (int2)(get_local_id(0), get_local_id(1));
	int2 local_index_with_border = local_index_logical + border_offset;

	// initialize local memory
	// use the same local mem as if 2d array (16x16 workgroup size)
	int2 local_mem_dims;
	local_mem_dims.s0 = get_local_size(0) + kernel_size_less_1;
	local_mem_dims.s1 = get_local_size(1) + kernel_size_less_1;
	int local_pointer = getImageIndex(local_mem_dims, local_index_logical);
	int2 global_index_offset = global_index - border_offset;

	// possible to be up, down, right OOB
	bool not_too_up = global_index_offset.s1 >= 0;
	bool not_too_down = global_index_offset.s1 < image_dims.s1;
	bool not_too_right = global_index_offset.s0 < image_dims.s0;
	if ( not_too_up && not_too_down && not_too_down ) {
		int global_pointer = getImageIndex(image_dims, global_index_offset);
		local_input[local_pointer] = input_float[global_pointer];
	}
	else if ( not_too_up && not_too_right) {
		// must be too down and not too right
		int global_pointer = getImageIndex(image_dims, (int2) (global_index_offset.s0, image_dims.s1 - 1) );
		local_input[local_pointer] = input_float[global_pointer];
	}
	else if ( not_too_down && not_too_right) {
		// must be too up and not too right
		int global_pointer = getImageIndex(image_dims, (int2) (global_index_offset.s0, 0) );
		local_input[local_pointer] = input_float[global_pointer];
	}
	else {
		// must be too right
		local_input[local_pointer] = 0;
	}

	// wrap for bottommost
	if ( local_index_logical.s1 < kernel_size_less_1) {
		int2 wrap_offset = (int2)(0, get_local_size(1));
		local_pointer = getImageIndex(local_mem_dims, local_index_logical + wrap_offset);
		global_index_offset = global_index - border_offset + wrap_offset;

		//// dup:
		// possible to be up, down, right OOB
		bool not_too_up = global_index_offset.s1 >= 0;
		bool not_too_down = global_index_offset.s1 < image_dims.s1;
		bool not_too_right = global_index_offset.s0 < image_dims.s0;
		if ( not_too_up && not_too_down && not_too_down ) {
			int global_pointer = getImageIndex(image_dims, global_index_offset);
			local_input[local_pointer] = input_float[global_pointer];
		}
		else if ( not_too_up && not_too_right) {
			// must be too down and not too right
			int global_pointer = getImageIndex(image_dims, (int2) (global_index_offset.s0, image_dims.s1 - 1) );
			local_input[local_pointer] = input_float[global_pointer];
		}
		else if ( not_too_down && not_too_right) {
			// must be too up and not too right
			int global_pointer = getImageIndex(image_dims, (int2) (global_index_offset.s0, 0) );
			local_input[local_pointer] = input_float[global_pointer];
		}
		else {
			// must be too right
			local_input[local_pointer] = 0;
		}
	}


	// make sure all local memory written
	barrier(CLK_LOCAL_MEM_FENCE);

	if (valid) {
		float result = 0;
		for (int d = -half_kernel_side; d <= half_kernel_side; ++d) {
			int2 offset = (int2)(0, d);
			local_pointer = getImageIndex(local_mem_dims, local_index_with_border + offset);
			result += local_input[local_pointer] * filter_kernel[half_kernel_side + d];
		}

		output_float[getImageIndex(image_dims, global_index)] = result;
	}
}

__kernel void halfSizeImage(
	__global float *input_float,
	__global float *output_float,
	const int2 input_dims
	)
{
	int2 output_index;
	output_index.s0 = get_global_id(0);
	output_index.s1 = get_global_id(1);

	int2 output_dims = input_dims / 2;
	int2 input_index = output_index * 2;

	output_float[getImageIndex(output_dims, output_index)] = input_float[getImageIndex(input_dims, input_index)];
}

__kernel void halfSizeFloat4(
	__global float4 *input_float4,
	__global float4 *output_float4,
	const int2 input_dims
	)
{
	int2 output_index;
	output_index.s0 = get_global_id(0);
	output_index.s1 = get_global_id(1);

	int2 output_dims = input_dims / 2;
	int2 input_index = output_index * 2;

	output_float4[getImageIndex(output_dims, output_index)] = input_float4[getImageIndex(input_dims, input_index)];
}

__kernel void halfSizeFloat4Mean(
	__global float4 *input_float4,
	__global float4 *output_float4,
	const int2 input_dims
	)
{
	int2 output_index;
	output_index.s0 = get_global_id(0);
	output_index.s1 = get_global_id(1);

	int2 output_dims = input_dims / 2;

	int2 input_indices[4];
	input_indices[0] = output_index * 2;
	input_indices[1] = input_indices[0] + (int2)(1,0);
	input_indices[2] = input_indices[0] + (int2)(0,1);
	input_indices[3] = input_indices[0] + (int2)(1,1);

	float4 input[4];
	for (int i = 0; i < 4; ++i) {
		input[i] = input_float4[getImageIndex(input_dims, input_indices[i])];
	}

	float4 total = (float4)(0);
	int valid_count = 0;
	for (int i = 0; i < 4; ++i) {
		if (isfinite(input[i].z)) {
			total += input[i];
			valid_count += 1;
		}
	}

	float4 result = valid_count ? total / valid_count : input[0];

	output_float4[getImageIndex(output_dims, output_index)] = result;
}