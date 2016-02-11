#define MIN_VOXELS_IN_DISTRIBUTION_RAMP 4 // should probably match other...should also be able to be increased...right?
#define MIN_VOXELS_IN_TSDF_RAMP 4 // was 3

// means use noise model (also takes into account min_voxels_...)
#define USE_SIGMA 1 // PETER REMMEMBER THAT YOU CHANGED THIS!@!!!!!!
#define SIGMA_COUNT 6 // was 3 forever

// see Noise.cpp (dup)
float simpleAxial(float z) {
	float w = z - 0.4;
	return 0.0012 + 0.0019 * w * w;
}
#define Z_MIN 0.4

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

float2 projectPoint(float2 f, float2 c, float4 point)
{
	return (point.s01 * f / point.s2 + c);
}

float4 pixelOnFocal(float2 f, float2 c, int2 pixel)
{
	float4 result;
	result.s01 = (convert_float2(pixel) - c) / f;
	result.s23 = 1.0f;
	return result;
}

float4 voxelToWorldFloat(float4 voxel_sizes, int4 voxel_ijk4)
{
	return convert_float4(voxel_ijk4) * voxel_sizes;
}

float4 worldToVoxelFloat(float4 voxel_sizes, float4 volume_point)
{
	return (1.0f / voxel_sizes) * volume_point;
}

// for things involving nearest voxel
bool checkVoxelWithinVolume(int4 volume_dims, int4 voxel_coords)
{
    return (voxel_coords.s0 >= 0 && voxel_coords.s0 <= (volume_dims.s0 - 1) &&
            voxel_coords.s1 >= 0 && voxel_coords.s1 <= (volume_dims.s1 - 1) &&
            voxel_coords.s2 >= 0 && voxel_coords.s2 <= (volume_dims.s2 - 1) );
}

// for things involving interpolation
bool checkSurroundingVoxelsAreWithinVolume(int4 volume_dims, int4 voxel_coords_floor)
{
	return (voxel_coords_floor.s0 >= 0 && voxel_coords_floor.s0 < (volume_dims.s0 - 1) &&
		voxel_coords_floor.s1 >= 0 && voxel_coords_floor.s1 < (volume_dims.s1 - 1) &&
		voxel_coords_floor.s2 >= 0 && voxel_coords_floor.s2 < (volume_dims.s2 - 1) );
}

int getImageIndex(int2 image_dims, int2 pixel)
{
	return pixel.s1 * image_dims.s0 + pixel.s0;
}

int getVIndex(int4 volume_dims, int i, int j, int k)
{
	return k * volume_dims.s0 * volume_dims.s1 + j * volume_dims.s0  + i;
}

int getVIndexInt4(int4 volume_dims, int4 voxel)
{
	return getVIndex(volume_dims, voxel.s0, voxel.s1, voxel.s2);
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


bool rayWillNeverHitVolume(int4 volume_dims, float4 id_f, float4 ray_unit_v)
{
	bool result = false;
	float4 dims_f = convert_float4(volume_dims);
	result = result || (id_f.s0 < 0 && ray_unit_v.s0 < 0);
	result = result || (id_f.s0 >= dims_f.s0 && ray_unit_v.s0 > 0);
	result = result || (id_f.s1 < 0 && ray_unit_v.s1 < 0);
	result = result || (id_f.s1 >= dims_f.s1 && ray_unit_v.s1 > 0);
	result = result || (id_f.s2 < 0 && ray_unit_v.s2 < 0);
	result = result || (id_f.s2 >= dims_f.s2 && ray_unit_v.s2 > 0);
	return result;
}

float getMinValueSurroundingVoxels(int4 volume_dims, int4 voxel_coords_floor, __global float *V)
{
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	float min_value = V[getVIndex(volume_dims, i, j, k)];
	min_value = min(min_value, V[getVIndex(volume_dims, i, j, k+1)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i, j+1, k)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i, j+1, k+1)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i+1, j, k)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i+1, j, k+1)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i+1, j+1, k)]);
	min_value = min(min_value, V[getVIndex(volume_dims, i+1, j+1, k+1)]);
	return min_value;
}

bool checkAllHaveWeights(int4 volume_dims, int4 voxel_coords_floor, __global float *W)
{
	bool result = true;
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	result = result && W[getVIndex(volume_dims, i, j, k)]>0;
	result = result && W[getVIndex(volume_dims, i, j, k+1)]>0;
	result = result && W[getVIndex(volume_dims, i, j+1, k)]>0;
	result = result && W[getVIndex(volume_dims, i, j+1, k+1)]>0;
	result = result && W[getVIndex(volume_dims, i+1, j, k)]>0;
	result = result && W[getVIndex(volume_dims, i+1, j, k+1)]>0;
	result = result && W[getVIndex(volume_dims, i+1, j+1, k)]>0;
	result = result && W[getVIndex(volume_dims, i+1, j+1, k+1)]>0;
	return result;
}

bool checkAnyHaveWeights(int4 volume_dims, int4 voxel_coords_floor, __global float *W)
{
	bool result = false;
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	result = result || W[getVIndex(volume_dims, i, j, k)]>0;
	result = result || W[getVIndex(volume_dims, i, j, k+1)]>0;
	result = result || W[getVIndex(volume_dims, i, j+1, k)]>0;
	result = result || W[getVIndex(volume_dims, i, j+1, k+1)]>0;
	result = result || W[getVIndex(volume_dims, i+1, j, k)]>0;
	result = result || W[getVIndex(volume_dims, i+1, j, k+1)]>0;
	result = result || W[getVIndex(volume_dims, i+1, j+1, k)]>0;
	result = result || W[getVIndex(volume_dims, i+1, j+1, k+1)]>0;
	return result;
}

float trilinearInterpolate(int4 volume_dims, float4 voxel_coords_f, __global float *D)
{
	int4 voxel_coords_floor = convert_int4(voxel_coords_f);
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;
	float d000 = D[getVIndex(volume_dims, i, j, k)];
	float d001 = D[getVIndex(volume_dims, i, j, k+1)];
	float d010 = D[getVIndex(volume_dims, i, j+1, k)];
	float d011 = D[getVIndex(volume_dims, i, j+1, k+1)];
	float d100 = D[getVIndex(volume_dims, i+1, j, k)];
	float d101 = D[getVIndex(volume_dims, i+1, j, k+1)];
	float d110 = D[getVIndex(volume_dims, i+1, j+1, k)];
	float d111 = D[getVIndex(volume_dims, i+1, j+1, k+1)];

	float4 offset = voxel_coords_f - convert_float4(voxel_coords_floor);
	float off_x = offset.s0;
	float off_y = offset.s1;
	float off_z = offset.s2;

	float result =		  d000 * (1 - off_x) * (1 - off_y) * (1 - off_z)
		+ d001 * (1 - off_x) * (1 - off_y) * (off_z)
		+ d010 * (1 - off_x) * (off_y) * (1 - off_z)
		+ d011 * (1 - off_x) * (off_y) * (off_z)
		+ d100 * (off_x) * (1 - off_y) * (1 - off_z)
		+ d101 * (off_x) * (1 - off_y) * (off_z)
		+ d110 * (off_x) * (off_y) * (1 - off_z)
		+ d111 * (off_x) * (off_y) * (off_z);

	return result;
}

float nearestNeighbor(int4 volume_dims, float4 voxel_coords_f, __global float *D)
{
	int4 nn = convert_int4_rte(voxel_coords_f);
	return D[getVIndex(volume_dims, nn.x, nn.y, nn.z)];
}

uchar4 trilinearInterpolateColor(int4 volume_dims, float4 voxel_coords_f, __global uchar4 *C)
{
	int4 voxel_coords_floor = convert_int4(voxel_coords_f);
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	float4 c000 = convert_float4(C[getVIndex(volume_dims, i, j, k)]);
	float4 c001 = convert_float4(C[getVIndex(volume_dims, i, j, k+1)]);
	float4 c010 = convert_float4(C[getVIndex(volume_dims, i, j+1, k)]);
	float4 c011 = convert_float4(C[getVIndex(volume_dims, i, j+1, k+1)]);
	float4 c100 = convert_float4(C[getVIndex(volume_dims, i+1, j, k)]);
	float4 c101 = convert_float4(C[getVIndex(volume_dims, i+1, j, k+1)]);
	float4 c110 = convert_float4(C[getVIndex(volume_dims, i+1, j+1, k)]);
	float4 c111 = convert_float4(C[getVIndex(volume_dims, i+1, j+1, k+1)]);

	float4 offset = voxel_coords_f - convert_float4(voxel_coords_floor);
	float off_x = offset.s0;
	float off_y = offset.s1;
	float off_z = offset.s2;

	float4 result =		  c000 * (1 - off_x) * (1 - off_y) * (1 - off_z)
		+ c001 * (1 - off_x) * (1 - off_y) * (off_z)
		+ c010 * (1 - off_x) * (off_y) * (1 - off_z)
		+ c011 * (1 - off_x) * (off_y) * (off_z)
		+ c100 * (off_x) * (1 - off_y) * (1 - off_z)
		+ c101 * (off_x) * (1 - off_y) * (off_z)
		+ c110 * (off_x) * (off_y) * (1 - off_z)
		+ c111 * (off_x) * (off_y) * (off_z);

	return convert_uchar4_sat_rte(result);
}


void trilinearInterpolateColorAndWeight(int4 volume_dims, float4 voxel_coords_f, __global uchar4 *C, __global float *W, uchar4 * result_color, float * result_weight)
{
	// SET INITIAL SAFE RETURN VALUES
	*result_color = (uchar4)(0);
	*result_weight = 0;

	int4 voxel_coords_floor = convert_int4(voxel_coords_f);
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	// get weights
	float w000 = W[getVIndex(volume_dims, i, j, k)];
	float w001 = W[getVIndex(volume_dims, i, j, k+1)];
	float w010 = W[getVIndex(volume_dims, i, j+1, k)];
	float w011 = W[getVIndex(volume_dims, i, j+1, k+1)];
	float w100 = W[getVIndex(volume_dims, i+1, j, k)];
	float w101 = W[getVIndex(volume_dims, i+1, j, k+1)];
	float w110 = W[getVIndex(volume_dims, i+1, j+1, k)];
	float w111 = W[getVIndex(volume_dims, i+1, j+1, k+1)];

	float4 c000 = convert_float4(C[getVIndex(volume_dims, i, j, k)]);
	float4 c001 = convert_float4(C[getVIndex(volume_dims, i, j, k+1)]);
	float4 c010 = convert_float4(C[getVIndex(volume_dims, i, j+1, k)]);
	float4 c011 = convert_float4(C[getVIndex(volume_dims, i, j+1, k+1)]);
	float4 c100 = convert_float4(C[getVIndex(volume_dims, i+1, j, k)]);
	float4 c101 = convert_float4(C[getVIndex(volume_dims, i+1, j, k+1)]);
	float4 c110 = convert_float4(C[getVIndex(volume_dims, i+1, j+1, k)]);
	float4 c111 = convert_float4(C[getVIndex(volume_dims, i+1, j+1, k+1)]);

	float4 offset = voxel_coords_f - convert_float4(voxel_coords_floor);
	float off_x = offset.s0;
	float off_y = offset.s1;
	float off_z = offset.s2;

	float offset_w000 = (1 - off_x) * (1 - off_y) * (1 - off_z);
	float offset_w001 = (1 - off_x) * (1 - off_y) * (off_z);
	float offset_w010 = (1 - off_x) * (off_y) * (1 - off_z);
	float offset_w011 = (1 - off_x) * (off_y) * (off_z);
	float offset_w100 = (off_x) * (1 - off_y) * (1 - off_z);
	float offset_w101 = (off_x) * (1 - off_y) * (off_z);
	float offset_w110 = (off_x) * (off_y) * (1 - off_z);
	float offset_w111 = (off_x) * (off_y) * (off_z);

	float ww000 = w000 > 0 ? offset_w000 : 0;
	float ww001 = w001 > 0 ? offset_w001 : 0;
	float ww010 = w010 > 0 ? offset_w010 : 0;
	float ww011 = w011 > 0 ? offset_w011 : 0;
	float ww100 = w100 > 0 ? offset_w100 : 0;
	float ww101 = w101 > 0 ? offset_w101 : 0;
	float ww110 = w110 > 0 ? offset_w110 : 0;
	float ww111 = w111 > 0 ? offset_w111 : 0;
	float ww_sum = ww000 + ww001 + ww010 + ww011 + ww100 + ww101 + ww110 + ww111;
	if (ww_sum < 0.5) return; // ok now

	float4 result_c = ww000 * c000 +
		ww001 * c001 +
		ww010 * c010 +
		ww011 * c011 +
		ww100 * c100 +
		ww101 * c101 +
		ww110 * c110 +
		ww111 * c111;
	result_c = result_c / ww_sum;

	float result_w = ww000 * w000 +
		ww001 * w001 +
		ww010 * w010 +
		ww011 * w011 +
		ww100 * w100 +
		ww101 * w101 +
		ww110 * w110 +
		ww111 * w111;
	result_w = result_w / ww_sum;

	// doesn't explicitly check 255 bounds...
	*result_color = convert_uchar4_sat_rte(result_c);
	*result_weight = result_w;
}


void trilinearInterpolateDistanceAndWeight(int4 volume_dims, float4 voxel_coords_f, __global float *D, __global float *W, float * result_distance, float * result_weight)
{
	// SET INITIAL RETURN VALUES!!
	*result_distance = -1;
	*result_weight = 0;

	int4 voxel_coords_floor = convert_int4(voxel_coords_f);
	int i = voxel_coords_floor.s0;
	int j = voxel_coords_floor.s1;
	int k = voxel_coords_floor.s2;

	// get weights
	float w000 = W[getVIndex(volume_dims, i, j, k)];
	float w001 = W[getVIndex(volume_dims, i, j, k+1)];
	float w010 = W[getVIndex(volume_dims, i, j+1, k)];
	float w011 = W[getVIndex(volume_dims, i, j+1, k+1)];
	float w100 = W[getVIndex(volume_dims, i+1, j, k)];
	float w101 = W[getVIndex(volume_dims, i+1, j, k+1)];
	float w110 = W[getVIndex(volume_dims, i+1, j+1, k)];
	float w111 = W[getVIndex(volume_dims, i+1, j+1, k+1)];

	float d000 = D[getVIndex(volume_dims, i, j, k)];
	float d001 = D[getVIndex(volume_dims, i, j, k+1)];
	float d010 = D[getVIndex(volume_dims, i, j+1, k)];
	float d011 = D[getVIndex(volume_dims, i, j+1, k+1)];
	float d100 = D[getVIndex(volume_dims, i+1, j, k)];
	float d101 = D[getVIndex(volume_dims, i+1, j, k+1)];
	float d110 = D[getVIndex(volume_dims, i+1, j+1, k)];
	float d111 = D[getVIndex(volume_dims, i+1, j+1, k+1)];

	float4 offset = voxel_coords_f - convert_float4(voxel_coords_floor);
	float off_x = offset.s0;
	float off_y = offset.s1;
	float off_z = offset.s2;

	float offset_w000 = (1 - off_x) * (1 - off_y) * (1 - off_z);
	float offset_w001 = (1 - off_x) * (1 - off_y) * (off_z);
	float offset_w010 = (1 - off_x) * (off_y) * (1 - off_z);
	float offset_w011 = (1 - off_x) * (off_y) * (off_z);
	float offset_w100 = (off_x) * (1 - off_y) * (1 - off_z);
	float offset_w101 = (off_x) * (1 - off_y) * (off_z);
	float offset_w110 = (off_x) * (off_y) * (1 - off_z);
	float offset_w111 = (off_x) * (off_y) * (off_z);

	const float min_weight = 0; // 1e-6 didn't help
	float ww000 = w000 > min_weight ? offset_w000 : 0;
	float ww001 = w001 > min_weight ? offset_w001 : 0;
	float ww010 = w010 > min_weight ? offset_w010 : 0;
	float ww011 = w011 > min_weight ? offset_w011 : 0;
	float ww100 = w100 > min_weight ? offset_w100 : 0;
	float ww101 = w101 > min_weight ? offset_w101 : 0;
	float ww110 = w110 > min_weight ? offset_w110 : 0;
	float ww111 = w111 > min_weight ? offset_w111 : 0;
	float ww_sum = ww000 + ww001 + ww010 + ww011 + ww100 + ww101 + ww110 + ww111;
	if (ww_sum < 0.5) return; // ok now with initial values set

	float result_d = ww000 * d000 +
		ww001 * d001 +
		ww010 * d010 +
		ww011 * d011 +
		ww100 * d100 +
		ww101 * d101 +
		ww110 * d110 +
		ww111 * d111;
	result_d = result_d / ww_sum;

	float result_w = ww000 * w000 +
		ww001 * w001 +
		ww010 * w010 +
		ww011 * w011 +
		ww100 * w100 +
		ww101 * w101 +
		ww110 * w110 +
		ww111 * w111;
	result_w = result_w / ww_sum;

	*result_distance = result_d;
	*result_weight = result_w;
}

// from http://www.cs.utah.edu/~awilliam/box/
float2 box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
{
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	if (ray_unit_v.s0 >= 0) {
		tmin = (box_min.s0 - camera_center.s0) / ray_unit_v.s0;
		tmax = (box_max.s0 - camera_center.s0) / ray_unit_v.s0;
	}
	else {
		tmin = (box_max.s0 - camera_center.s0) / ray_unit_v.s0;
		tmax = (box_min.s0 - camera_center.s0) / ray_unit_v.s0;
	}
	if (ray_unit_v.s1 >= 0) {
		tymin = (box_min.s1 - camera_center.s1) / ray_unit_v.s1;
		tymax = (box_max.s1 - camera_center.s1) / ray_unit_v.s1;
	}
	else {
		tymin = (box_max.s1 - camera_center.s1) / ray_unit_v.s1;
		tymax = (box_min.s1 - camera_center.s1) / ray_unit_v.s1;
	}
	if ( (tmin > tymax) || (tymin > tmax) ) {
		return nan((uint2)0);
	}
	if (tymin > tmin) {
		tmin = tymin;
	}
	if (tymax < tmax) {
		tmax = tymax;
	}
	if (ray_unit_v.s2 >= 0) {
		tzmin = (box_min.s2 - camera_center.s2) / ray_unit_v.s2;
		tzmax = (box_max.s2 - camera_center.s2) / ray_unit_v.s2;
	}
	else {
		tzmin = (box_max.s2 - camera_center.s2) / ray_unit_v.s2;
		tzmax = (box_min.s2 - camera_center.s2) / ray_unit_v.s2;
	}
	if ( (tmin > tzmax) || (tzmin > tmax) ) {
		return nan((uint2)0);
	}
	if (tzmin > tmin) {
		tmin = tzmin;
	}
	if (tzmax < tmax) {
		tmax = tzmax;
	}
	if ( (tmin < max_t) && (tmax > min_t) ) {
		float real_min_t = max(tmin, min_t);
		float real_max_t = min(tmax, max_t);
		return (float2)(real_min_t, real_max_t);
	}
	else {
		return nan((uint2)0);
	}
}


__kernel void addFrame(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	__global float *depth_image,
	__global uchar4 *color_image,
	__global int *segments,
	__global float *depth_weight,
	__global float *color_weight,
	const int which_segment,
	const int4 volume_dims,
	const float voxel_size,
	const float w_max_icp,
	const float w_max_color,
	const float16 pose,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int2 image_dims,
    const int use_most_recent_color,
    const float min_truncation_distance
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);

	float4 voxel_sizes = (float4)(voxel_size);

	// get world coordinates
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
	float4 world_point = applyPoseToPoint(pose, voxel_world_point);
	if (world_point.s2 < 0) return;
	float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
	int2 pixel_int = convert_int2_rte(pixel);
	// make sure they fall in the image
	bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 && 
		pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
	if (!on_image) return;
	int pixel_index = getImageIndex(image_dims, pixel_int);
	float depth = depth_image[pixel_index];
	if (! (depth > 0)) return;
	// new, check segment:
	if (which_segment > 0) {
		if (segments[pixel_index] != which_segment) {
			return;
		}
	}

#define USE_3D_DISTANCE 1
	// This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
	float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
	float depth_norm = length((float4) (xy, depth, 0));
	float voxel_norm = length((float4) (world_point.xyz, 0));
	float d = depth_norm - voxel_norm;
#else
	float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
	float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif


	if (d < d_min) return; // voxel is past depth



	////////////////////////////
	// Actually change voxel values
	size_t v_index = getVIndex(volume_dims, i, j, k);

	//// depth:
	float d_old = D[v_index];
	float dw_old = DW[v_index];
	float d_weight = depth_weight[pixel_index];

	// IS THIS REALLY RIGHT?
	// note that d_max is variable with USE_SIGMA
	float d_new = min(d, d_max);

	float dw_new = d_weight;
#ifdef USE_SIGMA
	// before this complication, just try a bigger ramp, dude!
	//float exponent = (-2 / 3.14159) * ((d*d) / (sigma_z*sigma_z));
	//float d_new = sign(d)*sqrt(1 - exp(exponent));
	// weight contains axial and lateral terms to downweight distance points
	d_weight *= (simpleAxial(Z_MIN) / sigma_z) * (Z_MIN*Z_MIN) / (depth*depth);
#endif
	float dw_sum = dw_old + dw_new;
	D[v_index] = (d_old * dw_old + d_new * dw_new) / dw_sum; 
	DW[v_index] = min(w_max_icp, dw_sum);


	//// color:
	uchar4 c_old = C[v_index];
	float cw_old = CW[v_index];

	// only update color within truncation region
	if (d < d_max) {
		uchar4 c_new = color_image[pixel_index];
		float c_weight = color_weight[pixel_index];
		float cw_new = c_weight * max(0.f, dw_new * (1 - fabs(d) / d_max));
		// could also use gaussian...

		float cw_sum = cw_old + cw_new;
		C[v_index] = (use_most_recent_color) ? c_new : convert_uchar4_sat_rte((convert_float4(c_old) * cw_old + convert_float4(c_new) * cw_new) / cw_sum);
		CW[v_index] = (use_most_recent_color) ? 1 : min(w_max_color, cw_sum);
	}
}


// original render
#if 0
__kernel void renderFrame(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	__global int *rendered_mask,
	__global uchar4 *rendered_color_image,
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose,
	const float16 inverse_pose,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int4 render_rect,
	const float step_size,
	const float min_render_depth,
	const float max_render_depth,
	const int replace_only_if_nearer,
	const int mask_value
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	float4 voxel_sizes = (float4)(voxel_size);

	int2 pixel = (int2)(i,j);
	int image_index = getImageIndex(render_rect.s23, pixel);

	float4 camera_center = applyPoseToPoint(inverse_pose, (float4)(0));
	float4 pixel_on_focal = pixelOnFocal(focal_lengths, camera_centers, pixel + render_rect.s01);
	float4 pixel_transformed = applyPoseToPoint(inverse_pose, pixel_on_focal);
	float4 ray_unit_v = (pixel_transformed - camera_center);
	ray_unit_v.s3 = 0;
	ray_unit_v = normalize(ray_unit_v);
	bool found_surface_voxel = false;
	const float reset_d = 1.0;
	float mat_d = reset_d;
	float mat_d_previous = reset_d;
	float t_previous = 0; // will be set if surface is found
	float t = 0; // initialized in loop

        // replace if nearer check
	float previous_distance = 0;
	if (replace_only_if_nearer) {
		// read previous
		float4 previous_point = rendered_points[image_index];
		if (isfinite(previous_point.s2)) {
			// from later: final_point = applyPoseToPoint(pose, t_star_point);
			// was:
			// previous_depth = previous_point.s2;
			// to think about...points behind camera?
			previous_distance = distance(camera_center, applyPoseToPoint(inverse_pose, previous_point));
			//printf("previous distance:%f\n", previous_distance);
		}
	}

	// new initial box check code
	// box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
#define SLABTEST 1
#ifdef SLABTEST
	float4 box_min = voxelToWorldFloat(voxel_sizes, (int4)(-1,-1,-1,0)); // -1 to be "safe"
	float4 box_max = voxelToWorldFloat(voxel_sizes, volume_dims); // volume dims is already "+1"
	float2 t_bounds = box_intersect(camera_center, ray_unit_v, box_min, box_max, min_render_depth, max_render_depth);
	if (!isfinite(t_bounds.s0) || !isfinite(t_bounds.s1)) {
		// only really need to check the first one
		return;
	}
#else
	float2 t_bounds = (float2)(min_render_depth, max_render_depth);
#endif

	//for (t = min_render_depth; t <= max_render_depth; t += step_size) {
	for (t = t_bounds.s0; t <= t_bounds.s1; t += step_size) {
		float4 volume_point = camera_center + t * ray_unit_v;

		// only if replace_only_if_nearer
		if (replace_only_if_nearer && previous_distance > 0) {
			if (t > previous_distance) {
				break;
			}
		}

		float4 id_f = worldToVoxelFloat(voxel_sizes, volume_point);
		int4 id_floor = convert_int4(id_f);

		// could "check for never intersecting volume (just an optimization):"
		// This can be commented out without affecting result
#ifndef SLABTEST
		if (rayWillNeverHitVolume(volume_dims, id_f, ray_unit_v)) break;
#endif

		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, id_floor)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		if (!checkAllHaveWeights(volume_dims, id_floor, DW)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		mat_d = trilinearInterpolate(volume_dims, id_f, D);

		// backside:
		if (mat_d > 0 && mat_d_previous < 0) break;

		// frontside:
		if (mat_d < 0 && mat_d_previous > 0 && mat_d_previous < reset_d) {
			found_surface_voxel = true;
			break;
		}

		// these 2 go together:
		mat_d_previous = mat_d;
		t_previous = t;
	}

	if (found_surface_voxel) {
		// DEBUG
		// "paper":
		//float t_star =  t_previous - (t - t_previous) * mat_d_previous / (mat_d - mat_d_previous);
		// stupid:
		//float t_star = t;
		// peter algebra (matches paper I believe)
		float t_star = t - mat_d * (t - t_previous) / (mat_d - mat_d_previous);
		float4 t_star_point = camera_center + t_star * ray_unit_v;
		float4 id_t_star = worldToVoxelFloat(voxel_sizes, t_star_point);

		const float normals_volume_delta = 1.f;
		const float4 normal_delta_v = (float4)(normals_volume_delta, normals_volume_delta, normals_volume_delta, 0);
		const float4 normal_delta_v_x = (float4)(normals_volume_delta, 0, 0, 0);
		const float4 normal_delta_v_y = (float4)(0, normals_volume_delta, 0, 0);
		const float4 normal_delta_v_z = (float4)(0, 0, normals_volume_delta, 0);

		float4 id_t_star_check_pos = id_t_star + normal_delta_v;
		float4 id_t_star_x_pos = id_t_star + normal_delta_v_x;
		float4 id_t_star_y_pos = id_t_star + normal_delta_v_y;
		float4 id_t_star_z_pos = id_t_star + normal_delta_v_z;

		// bomb out if normal check exits volume
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star))) return;
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star_check_pos))) return;

		// bomb out if any of the normal  of the normal checks don't have weights
		// this is the new check that solves the rendering bug
		// note there are redundant checks in here
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_x_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_y_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_z_pos), DW)) return;

		// could also assume this d_t_star is 0 (which is of course the goal)
		float d_t_star = trilinearInterpolate(volume_dims, id_t_star, D);
		float d_t_star_x_pos = trilinearInterpolate(volume_dims, id_t_star_x_pos, D);
		float d_t_star_y_pos = trilinearInterpolate(volume_dims, id_t_star_y_pos, D);
		float d_t_star_z_pos = trilinearInterpolate(volume_dims, id_t_star_z_pos, D);

		float4 normal = (float4)
			(
			(d_t_star_x_pos - d_t_star),
			(d_t_star_y_pos - d_t_star),
			(d_t_star_z_pos - d_t_star),
			0
			);
		normal = normalize(normal);
		// got normal

		// whew!  Can finally set some output values
		float4 t_star_point_relative_to_camera = applyPoseToPoint(pose, t_star_point);
		float4 normal_relative_to_camera = applyPoseToNormal(pose, normal);

		// check that normal points towards camera...if not, we've got a "back" surface (no point flipping it)
		if (normal_relative_to_camera.s2 > 0) return;

		uchar4 color = trilinearInterpolateColor(volume_dims, id_t_star, C);
		//uchar4 color = trilinearInterpolateColorWithWeights(volume_dims, id_t_star, C, CW);

		// HERE WE ACTUALLY RENDER SOMETHING
		rendered_mask[image_index] = mask_value;
		rendered_color_image[image_index] = color;
		rendered_points[image_index] = t_star_point_relative_to_camera;
		rendered_normals[image_index] = normal_relative_to_camera;
	}
}
#endif

__kernel void addVolume(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	__global float *D2,
	__global float *DW2,
	__global uchar4 *C2,
	__global float *CW2,
	const int4 volume_dims,
	const int4 volume_dims2,
	const float voxel_size,
	const float voxel_size2,
	const float w_max_icp,
	const float w_max_color,
	const float16 pose_to_v2
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);
	float4 voxel_sizes = (float4)(voxel_size);
	float4 voxel_sizes2 = (float4)(voxel_size2);
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
	float4 voxel_world_point_2 = applyPoseToPoint(pose_to_v2, voxel_world_point);

	// the lookup in volume 2
	float4 id_f = worldToVoxelFloat(voxel_sizes2, voxel_world_point_2);
	int4 id_floor = convert_int4(id_f);

	// bomb out if can't interpolate here
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims2, id_floor)) return;

	// Peter new note:  should not be necessary!  Should count on weight to do the work for us...
	// see if we're interpolating anything meaningful
	// note this is ANY have weights
	//if (!checkAnyHaveWeights(volume_dims2, id_floor, DW2)) return;
	// "any" causes "black" to sneak in...do "all" instead?
	//if (!checkAllHaveWeights(volume_dims2, id_floor, W2)) return;

	// get current values
	size_t v_index = getVIndex(volume_dims, i, j, k);
	float d_old = D[v_index];
	float dw_old = DW[v_index]; // was INT up until 4/22/2014!!!
	uchar4 c_old = C[v_index];
	float cw_old = CW[v_index];

	float d2;
	float dw2;
	trilinearInterpolateDistanceAndWeight(volume_dims2, id_f, D2, DW2, &d2, &dw2);

	float dw_sum = dw_old + dw2;
	if (dw_sum <= 0) return; // meaningless depth => meaningless color
	D[v_index] = (dw_old * d_old + dw2 * d2) / dw_sum;
	DW[v_index] = min(w_max_icp, dw_sum);

	uchar4 c2;
	float cw2;
	trilinearInterpolateColorAndWeight(volume_dims2, id_f, C2, CW2, &c2, &cw2);

	float cw_sum = cw_old + cw2;
	if (cw_sum <= 0) return; // already updated depth values
	C[v_index] = convert_uchar4_sat_rte((convert_float4(c_old) * cw_old + convert_float4(c2) * cw2) / cw_sum);
	CW[v_index] = min(w_max_color, cw_sum);
}


__kernel void setVolumeToSphere(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	const int4 volume_dims,
	const float voxel_size,
	const float sphere_radius
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);
	float4 voxel_sizes = (float4)(voxel_size);
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
	size_t v_index = getVIndex(volume_dims, i, j, k);

	float r = length( (float4) (voxel_world_point.s012, 0) );
	float d = r - sphere_radius;

	D[v_index] = d;
	C[v_index] = (uchar4) (200,200,200,0);
	DW[v_index] = 1.0;
	CW[v_index] = 1.0;
}

__kernel void setMaxWeightInVolume(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	const int4 volume_dims,
	const float max_weight
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	size_t v_index = getVIndex(volume_dims, i, j, k);
	DW[v_index] = min(DW[v_index], max_weight);
	CW[v_index] = min(CW[v_index], max_weight);
}

// takes color buffers as arguments, though doesn't use it...
__kernel void setValueInSphere(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	const int4 volume_dims,
	const float voxel_size,
	const float d_value,
	const float dw_value,
	const uchar4 c_value,
	const float cw_value,
	const float4 circle_center,
	const float circle_radius
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	size_t v_index = getVIndex(volume_dims, i, j, k);

	int4 ijk4 = (int4)(i,j,k,0);
	float4 voxel_sizes = (float4)(voxel_size);
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);

	float check_d = distance(voxel_world_point, circle_center);
	if (check_d < circle_radius) {
		D[v_index] = d_value;
		DW[v_index] = dw_value;
		C[v_index] = c_value;
		CW[v_index] = cw_value;
	}
}

__kernel void setValueInBox(
	__global float *D,
	__global float *DW,
	__global uchar4 *C,
	__global float *CW,
	const int4 volume_dims,
	const float voxel_size,
	const float d_value,
	const float dw_value,
	const uchar4 c_value,
	const float cw_value,
	const float16 pose,
	const float16 pose_inverse
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	size_t v_index = getVIndex(volume_dims, i, j, k);

	int4 ijk4 = (int4)(i,j,k,0);
	float4 voxel_sizes = (float4)(voxel_size);
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);

	// inverse here?  Idea is pose is applied to box
	float4 p = applyPoseToPoint(pose_inverse, voxel_world_point);

	// check against 0-1 box...exclusive?
	bool check_box = all(p.s012 > 0) && all(p.s012 < 1);
	if (check_box) {
		D[v_index] = d_value;
		DW[v_index] = dw_value;
		C[v_index] = c_value;
		CW[v_index] = cw_value;
	}
}


__kernel void setPointsInsideBoxTrue(
	__global float * depth_image,
	__global uchar * inside_image,
	const float2 camera_c,
	const float2 camera_f,
	const int2 image_dims,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose
	)
{
	int2 pixel;
	pixel.s0 = get_global_id(0);
	pixel.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, pixel);
	float d = depth_image[image_index];

	float4 p = depthToPoint(pixel, d, camera_c, camera_f);
	float4 p_in_box = applyPoseToPoint(pose, p);
	float4 box_max = convert_float4(volume_dims) * voxel_size;

	bool is_inside = all(p_in_box.s012 > (float3)(0)) && all(p_in_box.s012 < box_max.s012);
	if (is_inside) {
		inside_image[image_index] = 0xFF;
	}
}

__kernel void doesBoxContainSurface(
	__global float *D,
	__global float *DW,
	__global int *result_buffer,
	const int4 volume_dims,
	const int4 box_origin,
	const int4 box_size
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 this_index = (int4)(i,j,k,0) + box_origin;

	if ( any(this_index.s012 < 0) || any(this_index.s012 >= volume_dims.s012) ) return;

	size_t v_index = getVIndex(volume_dims, this_index.s0, this_index.s1, this_index.s2);
	float d = D[v_index];
	float dw = DW[v_index];
	if (dw > 0 && d < 0) {
		result_buffer[0] = true;
	}
}

__kernel void KernelSetFloat(__global float * mem, __private float val) {
	mem[get_global_id(0)]=val;
}

__kernel void KernelSetUChar(__global unsigned char * mem, __private unsigned char val) {
	mem[get_global_id(0)]=val;
}

__kernel void KernelSetInt(__global int * mem, __private int val) {
        mem[get_global_id(0)]=val;
}

__kernel void KernelSetFloat4(__global float4 * mem, __private float4 val) {
    mem[get_global_id(0)]=val;
}


__kernel void KernelAddFrameToHistogram(
	__global float *histogram_bin,
	__global float *depth_image,
	__global int *segments,
	const int which_segment,
	const float bin_min,
	const float bin_max,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose,
	const float2 focal_lengths,
	const float2 camera_centers,
    const int2 image_dims,
    const float min_truncation_distance
	)
{
	// should consider making this consistent part a function, obviously
	/////// UNCHANGED FROM addFrame
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);

	float4 voxel_sizes = (float4)(voxel_size);

	// get world coordinates
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
	float4 world_point = applyPoseToPoint(pose, voxel_world_point);
	if (world_point.s2 < 0) return;
	float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
	int2 pixel_int = convert_int2_rte(pixel);
	// make sure they fall in the image
	bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 && 
		pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
	if (!on_image) return;
	int pixel_index = getImageIndex(image_dims, pixel_int);
	float depth = depth_image[pixel_index];
	if (! (depth > 0)) return;
	// new, check segment:
	if (which_segment > 0) {
		if (segments[pixel_index] != which_segment) {
			return;
		}
	}

#define USE_3D_DISTANCE 1
	// This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
	float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
	float depth_norm = length((float4) (xy, depth, 0));
	float voxel_norm = length((float4) (world_point.xyz, 0));
	float d = depth_norm - voxel_norm;
#else
	float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

	if (d < d_min) return; // voxel is past depth



	/////// END: UNCHANGED FROM addFrame

	////////////////////////////
        // Actually change voxel values
        size_t v_index = getVIndex(volume_dims, i, j, k);

	// empty space uses d_max...
        float d_new = min(d, d_max);

        // skip d_max values for ALL bins?  Probably want to store them somewhere...
        if (d_new < d_max) {

            // update this bin if needed
            if (d_new >= bin_min && d_new < bin_max) {
                float count_old = histogram_bin[v_index];
                float count_new = count_old + 1.f;
                histogram_bin[v_index] = count_new;
            }
         }
}


__kernel void KernelHistogramSum(
	__global float *histogram_bin,
	__global float *sum,
	__global float *count,
	const float bin_min,
	const float bin_max,
	const int4 volume_dims
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);

	size_t v_index = getVIndex(volume_dims, i, j, k);

	float bin_count = histogram_bin[v_index];
	float to_add = bin_count * (bin_min + bin_max) / 2;
	sum[v_index] += to_add;
	count[v_index] += bin_count;
}



__kernel void KernelHistogramSumCheckIndex(
        __global float *histogram_bin,
        __global int *index_buffer,
        __global float *sum,
        __global float *count,
        const int this_index,
        const int index_range,
        const float bin_min,
        const float bin_max,
        const int4 volume_dims
        )
{
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        size_t v_index = getVIndex(volume_dims, i, j, k);

        int index_from_buffer = index_buffer[v_index];

        if (this_index >= index_from_buffer - index_range && this_index <= index_from_buffer + index_range) {
            float bin_count = histogram_bin[v_index];
            float to_add = bin_count * (bin_min + bin_max) / 2;
            sum[v_index] += to_add;
            count[v_index] += bin_count;
        }
}

__kernel void KernelMaxFloats(
        __global float *running_values,
        __global int *running_index,
        __global float *new_values,
        const int new_index
        )
{
        size_t i = get_global_id(0);

		float current_value = running_values[i];
		int current_index = running_index[i];
		float new_value = new_values[i];

		if (current_index < 0 || new_value > current_value) {
			running_values[i] = new_value;
			running_index[i] = new_index;
		}
}

__kernel void KernelMinFloats(
        __global float *running_values,
        __global int *running_index,
        __global float *new_values,
        const int new_index
        )
{
        size_t i = get_global_id(0);

		float current_value = running_values[i];
		int current_index = running_index[i];
		float new_value = new_values[i];

		if (current_index < 0 || new_value < current_value) {
			running_values[i] = new_value;
			running_index[i] = new_index;
		}
}


__kernel void KernelPickIfIndexFloats(
        __global float *running_buffer,
        __global int *possible_index,
        __global float *possible_values,
        const int index
        )
{
        size_t i = get_global_id(0);

		int possible_value_index = possible_index[i];
		if (possible_value_index == index) {
			running_buffer[i] = possible_values[i];
		}
}

__kernel void KernelPickIfIndexFloat4(
        __global float4 *running_buffer,
        __global int *possible_index,
        __global float4 *possible_values,
        const int index
        )
{
        size_t i = get_global_id(0);

        int possible_value_index = possible_index[i];
        if (possible_value_index == index) {
            running_buffer[i] = possible_values[i];
        }
}


__kernel void KernelHistogramMax(
        __global float *histogram_bin,
        __global int *volume_index,
        __global float *volume_value,
        const int4 volume_dims,
        const int bin_index
        )
{
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        size_t v_index = getVIndex(volume_dims, i, j, k);

        float bin_count = histogram_bin[v_index];
        float previous_max = volume_value[v_index];

        if (bin_count > previous_max) {
            volume_index[v_index] = bin_index;
            volume_value[v_index] = bin_count;
        }
}


__kernel void KernelHistogramMaxCheckIndex(
        __global float *histogram_bin,
        __global int *previous_index,
        __global int *volume_index,
        __global float *volume_value,
        const int4 volume_dims,
        const int bin_index,
        const int index_range
        )
{
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        size_t v_index = getVIndex(volume_dims, i, j, k);

        // only check this one if OUTSIDE index range of previous index
        int previous_index_value = previous_index[v_index];

        if (bin_index < previous_index_value - index_range || bin_index > previous_index_value + index_range) {

            float bin_count = histogram_bin[v_index];
            float previous_max = volume_value[v_index];

            if (bin_count > previous_max) {
                volume_index[v_index] = bin_index;
                volume_value[v_index] = bin_count;
            }
       }
}


__kernel void KernelHistogramVariance(
        __global float *histogram_bin,
        __global float *mean,
        __global float *count,
        __global float *variance,
        const float bin_min,
        const float bin_max,
        const int4 volume_dims
        )
{
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);
        size_t v_index = getVIndex(volume_dims, i, j, k);

        float bin_count = histogram_bin[v_index];

        // implies count_value > 0
        if (bin_count > 0) {
            float mean_value = mean[v_index];
            float count_value = count[v_index];
            float bin_value = (bin_min + bin_max) / 2;

            // fine:
            //float to_add = (bin_count / count_value) * bin_value * bin_value - mean_value * mean_value;

            // also fine
            float diff = bin_value - mean_value;
            float to_add = bin_count / count_value * diff * diff;
            variance[v_index] += to_add;
        }
}



__kernel void KernelDivideFloats(
	__global float *buffer,
	__global float *divisors
	)
{
	size_t i = get_global_id(0);

	float divisor = divisors[i];
        float new_value = divisor > 0 ? buffer[i] / divisor : 0;
        buffer[i] = new_value;
}

__kernel void KernelAddFloats(
        __global float *buffer,
        __global float *to_be_added
        )
{
        size_t i = get_global_id(0);

        float to_add = to_be_added[i];
        float new_value = buffer[i] + to_add;
        buffer[i] = new_value;
}

__kernel void KernelAddFloatsWithWeights(
        __global float *running_values,
        __global float *running_weights,
        __global float *to_add_values,
        __global float *to_add_weights
        )
{
        size_t i = get_global_id(0);

        float old_value = running_values[i];
        float old_weight = running_weights[i];
        float to_add_value = to_add_values[i];
        float to_add_weight = to_add_weights[i];

        float new_weight = old_weight + to_add_weight;
        float new_value = (old_value * old_weight + to_add_value * to_add_weight) / new_weight;

		// right!!  gotta do this:
		if (new_weight > 0) {
			running_values[i] = new_value;
			running_weights[i] = new_weight;
		}
}

__kernel void KernelAddFloatsWithWeightsExternalWeight(
        __global float *running_values,
        __global float *running_weights,
        __global float *to_add_values,
        __global float *to_add_weights,
		const float external_weight
        )
{
        size_t i = get_global_id(0);

        float old_value = running_values[i];
        float old_weight = running_weights[i];
        float to_add_value = to_add_values[i];
        float to_add_weight = to_add_weights[i] * external_weight;

        float new_weight = old_weight + to_add_weight;
        float new_value = (old_value * old_weight + to_add_value * to_add_weight) / new_weight;

		// right!!  gotta do this:
		if (new_weight > 0) {
			running_values[i] = new_value;
			running_weights[i] = new_weight;
		}
}

__kernel void KernelMinAbsFloatsWithWeights(
        __global float *running_values,
        __global float *running_weights,
        __global float *to_min_values,
        __global float *to_min_weights
        )
{
    size_t i = get_global_id(0);

    float old_value = running_values[i];
    float old_weight = running_weights[i];
    float to_min_value = to_min_values[i];
    float to_min_weight = to_min_weights[i];

    if (old_weight <= 0 || ( fabs(to_min_value) < fabs(old_value) ) ) {
        running_values[i] = to_min_value;
        running_weights[i] = to_min_weight;
    }
}

__kernel void KernelMinAbsFloatsWithWeightsRecordIndex(
        __global float *running_values,
        __global float *running_weights,
        __global int *running_index,
        __global float *to_min_values,
        __global float *to_min_weights,
		const int this_index
        )
{
    size_t i = get_global_id(0);

    float old_value = running_values[i];
    float old_weight = running_weights[i];
    float to_min_value = to_min_values[i];
    float to_min_weight = to_min_weights[i];

    if (old_weight <= 0 || ( fabs(to_min_value) < fabs(old_value) ) ) {
        running_values[i] = to_min_value;
        running_weights[i] = to_min_weight;
		running_index[i] = this_index;
    }
}

__kernel void KernelMinAbsFloatsWithWeightsAndMinimumWeightFraction(
        __global float *running_values,
        __global float *running_weights,
        __global float *to_min_values,
        __global float *to_min_weights,
		__global float *max_weights,
		const float min_weight_fraction
        )
{
    size_t i = get_global_id(0);

    float old_value = running_values[i];
    float old_weight = running_weights[i];
    float to_min_value = to_min_values[i];
    float to_min_weight = to_min_weights[i];

	// note that run correctly, to_min_weight will equal max_weights once for each voxel...
	float weight_fraction = to_min_weight / max_weights[i];

    if (old_weight <= 0 || ( (fabs(to_min_value) < fabs(old_value) && weight_fraction >= min_weight_fraction) ) ) {
        running_values[i] = to_min_value;
        running_weights[i] = to_min_weight;
    }
}

float gaussian_pdf(float mean, float variance, float x_value)
{
    float normalize = 1.f / sqrt(variance * 2 * 3.14159265359f);
    float diff = x_value - mean;
    float pdf_value = normalize * exp( -diff * diff / (2 * variance));
    return pdf_value;
}

__kernel void KernelGaussianPDF(
        __global float *means,
        __global float *variances,
        __global float *x_values,
        __global float *pdf_values
        )
{
        size_t i = get_global_id(0);

        float mean = means[i];
        float variance = variances[i];
        float x_value = x_values[i];
        pdf_values[i] = gaussian_pdf(mean, variance, x_value);
}

__kernel void KernelGaussianPDFConstantX(
        __global float *means,
        __global float *variances,
        const float x_value,
        __global float *pdf_values
        )
{
        size_t i = get_global_id(0);

        float mean = means[i];
        float variance = variances[i];
        pdf_values[i] = gaussian_pdf(mean, variance, x_value);
}



__kernel void KernelExtractVolumeSlice(
	__global float *volume,
	__global float *result,
	const int4 volume_dims,
	const int axis,
	const int position
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	int4 v_index4;
	int2 image_dims;
	if (axis == 0) {
		v_index4 = (int4)(position, i, j, 0);
		image_dims = (int2)(volume_dims.s1, volume_dims.s2);
	}
	else if (axis == 1) {
		v_index4 = (int4)(i, position, j, 0);
		image_dims = (int2)(volume_dims.s0, volume_dims.s2);
	}
	else if (axis == 2) {
		v_index4 = (int4)(i, j, position, 0);
		image_dims = (int2)(volume_dims.s0, volume_dims.s1);
	}
	size_t v_index = getVIndex(volume_dims, v_index4.s0, v_index4.s1, v_index4.s2);
	float v = volume[v_index];

	size_t i_index = getImageIndex(image_dims, (int2)(i,j));
	result[i_index] = v;
}

__kernel void KernelExtractVolumeSliceFloat4(
    __global float4 *volume,
    __global float4 *result,
    const int4 volume_dims,
    const int axis,
    const int position
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    int4 v_index4;
    int2 image_dims;
    if (axis == 0) {
        v_index4 = (int4)(position, i, j, 0);
        image_dims = (int2)(volume_dims.s1, volume_dims.s2);
    }
    else if (axis == 1) {
        v_index4 = (int4)(i, position, j, 0);
        image_dims = (int2)(volume_dims.s0, volume_dims.s2);
    }
    else if (axis == 2) {
        v_index4 = (int4)(i, j, position, 0);
        image_dims = (int2)(volume_dims.s0, volume_dims.s1);
    }
    size_t v_index = getVIndex(volume_dims, v_index4.s0, v_index4.s1, v_index4.s2);
    float4 v = volume[v_index];

    size_t i_index = getImageIndex(image_dims, (int2)(i,j));
    result[i_index] = v;
}


__kernel void KernelExtractVolumeSliceFloat4Length(
    __global float4 *volume,
    __global float *result,
    const int4 volume_dims,
    const int axis,
    const int position
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    int4 v_index4;
    int2 image_dims;
    if (axis == 0) {
        v_index4 = (int4)(position, i, j, 0);
        image_dims = (int2)(volume_dims.s1, volume_dims.s2);
    }
    else if (axis == 1) {
        v_index4 = (int4)(i, position, j, 0);
        image_dims = (int2)(volume_dims.s0, volume_dims.s2);
    }
    else if (axis == 2) {
        v_index4 = (int4)(i, j, position, 0);
        image_dims = (int2)(volume_dims.s0, volume_dims.s1);
    }
    size_t v_index = getVIndex(volume_dims, v_index4.s0, v_index4.s1, v_index4.s2);
    float4 v = volume[v_index];

    size_t i_index = getImageIndex(image_dims, (int2)(i,j));
    result[i_index] = length(v);
}



__kernel void KernelExtractVolumeFloat(
	__global float *volume,
	__global float *result,
	const int4 volume_dims,
	const int4 voxel
	)
{
	size_t v_index = getVIndex(volume_dims, voxel.s0, voxel.s1, voxel.s2);
	float v = volume[v_index];

	result[0] = v;
}



__kernel void KernelAddFrameTo2Means(
	__global float *mean_1,
	__global float *count_1,
	__global float *mean_2,
	__global float *count_2,
	__global float *depth_image,
	__global int *segments,
	const int which_segment,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose,
	const float2 focal_lengths,
	const float2 camera_centers,
    const int2 image_dims,
    const float min_truncation_distance
	)
{
	// should consider making this consistent part a function, obviously
	/////// UNCHANGED FROM addFrame
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);

	float4 voxel_sizes = (float4)(voxel_size);

	// get world coordinates
	float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
	float4 world_point = applyPoseToPoint(pose, voxel_world_point);
	if (world_point.s2 < 0) return;
	float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
	int2 pixel_int = convert_int2_rte(pixel);
	// make sure they fall in the image
	bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 && 
		pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
	if (!on_image) return;
	int pixel_index = getImageIndex(image_dims, pixel_int);
	float depth = depth_image[pixel_index];
	if (! (depth > 0)) return;
	// new, check segment:
	if (which_segment > 0) {
		if (segments[pixel_index] != which_segment) {
			return;
		}
	}

#define USE_3D_DISTANCE 1
	// This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
	float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
	float depth_norm = length((float4) (xy, depth, 0));
	float voxel_norm = length((float4) (world_point.xyz, 0));
	float d = depth_norm - voxel_norm;
#else
	float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

	if (d < d_min) return; // voxel is past depth


	/////// END: UNCHANGED FROM addFrame

	////////////////////////////
	// Actually change voxel values
	size_t v_index = getVIndex(volume_dims, i, j, k);

	// empty space uses d_max...
	float d_new = min(d, d_max);

	// figure out which mean to update, and update it

	float m1 = mean_1[v_index];
	float m2 = mean_2[v_index];

	float distance_1 = fabs(d_new - m1);
	float distance_2 = fabs(d_new - m2);

	if (distance_1 <= distance_2) {
		float count = count_1[v_index];
		float mean_sum = d_new + count * m1;
		float new_count = count + 1;
		float new_mean = mean_sum / new_count;
		mean_1[v_index] = new_mean;
		count_1[v_index] = new_count;
	}
	else {
		float count = count_2[v_index];
		float mean_sum = d_new + count * m2;
		float new_count = count + 1;
		float new_mean = mean_sum / new_count;
		mean_2[v_index] = new_mean;
		count_2[v_index] = new_count;
	}
}

// try to somewhat robustly get a normal for a voxel (for multimodal add)
// should use local memory to avoid all these extra reads!
float4 getNormalForVoxel(__global float *volume, const int4 volume_dims, const int4 voxel)
{
    const float4 no_normal = (float4)(0,0,0,0);
    // return 0 vector if voxel not in volume
    if (!checkVoxelWithinVolume(volume_dims, voxel)) return no_normal;

	// assume one side out means other side in
    int4 x_offset = (int4)(1,0,0,0);
    int4 y_offset = (int4)(0,1,0,0);
    int4 z_offset = (int4)(0,0,1,0);

// this is probably wrong...gotta flip the meaning later as well
#if 0
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) x_offset = -x_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) y_offset = -y_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) z_offset = -z_offset;
#endif

    // instead just bomb out
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) return no_normal;

	float v = volume[getVIndexInt4(volume_dims, voxel)];
	float v_x = volume[getVIndexInt4(volume_dims, voxel + x_offset)];
	float v_y = volume[getVIndexInt4(volume_dims, voxel + y_offset)];
	float v_z = volume[getVIndexInt4(volume_dims, voxel + z_offset)];

	// totally ignoring weights at this point!
	float4 normal = (float4)(v_x - v, v_y - v, v_z - v, 0);

    // perhaps look at the length of this vector, and if < epsilon, set to zero?
    const float epsilon = 1e-4;
    if (length(normal) < epsilon) {
        return no_normal;
    }
    else {
        return normalize(normal);
    }
}


// try to somewhat robustly get a normal for a voxel (for multimodal add)
// should use local memory to avoid all these extra reads!
float4 getNormalForVoxelWithWeights(__global float *volume, __global float * weights, const int4 volume_dims, const int4 voxel)
{
    const float4 no_normal = (float4)(0,0,0,0);
    // return 0 vector if voxel not in volume
    if (!checkVoxelWithinVolume(volume_dims, voxel)) return no_normal;

    // assume one side out means other side in
    int4 x_offset = (int4)(1,0,0,0);
    int4 y_offset = (int4)(0,1,0,0);
    int4 z_offset = (int4)(0,0,1,0);

// this is probably wrong...gotta flip the meaning later as well
#if 0
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) x_offset = -x_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) y_offset = -y_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) z_offset = -z_offset;
#endif

    // instead just bomb out
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) return no_normal;

    // also bomb out if any don't have weights..
    float w = weights[getVIndexInt4(volume_dims, voxel)];
    float w_x = weights[getVIndexInt4(volume_dims, voxel + x_offset)];
    float w_y = weights[getVIndexInt4(volume_dims, voxel + y_offset)];
    float w_z = weights[getVIndexInt4(volume_dims, voxel + z_offset)];
    if (w <= 0 || w_x <= 0 || w_y <= 0 || w_z <= 0) return no_normal;


    float v = volume[getVIndexInt4(volume_dims, voxel)];
    float v_x = volume[getVIndexInt4(volume_dims, voxel + x_offset)];
    float v_y = volume[getVIndexInt4(volume_dims, voxel + y_offset)];
    float v_z = volume[getVIndexInt4(volume_dims, voxel + z_offset)];

    float4 normal = (float4)(v_x - v, v_y - v, v_z - v, 0);

    // perhaps look at the length of this vector, and if < epsilon, set to zero?
    const float epsilon = 1e-5f;
    if (length(normal) < epsilon) {
        return no_normal;
    }
    else {
        return normalize(normal);
    }
}


// THIS IS OBVIOUS CRAP FOR DEBUGGING
// try to somewhat robustly get a normal for a voxel (for multimodal add)
// should use local memory to avoid all these extra reads!
float4 getNormalForVoxelWithWeightsUnnormalized(__global float *volume, __global float * weights, const int4 volume_dims, const int4 voxel)
{
    const float4 no_normal = (float4)(0,0,0,0);
    // return 0 vector if voxel not in volume
    if (!checkVoxelWithinVolume(volume_dims, voxel)) return no_normal;

    // assume one side out means other side in
    int4 x_offset = (int4)(1,0,0,0);
    int4 y_offset = (int4)(0,1,0,0);
    int4 z_offset = (int4)(0,0,1,0);

// this is probably wrong...gotta flip the meaning later as well
#if 0
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) x_offset = -x_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) y_offset = -y_offset;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) z_offset = -z_offset;
#endif

    // instead just bomb out
    if (!checkVoxelWithinVolume(volume_dims, voxel + x_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + y_offset)) return no_normal;
    if (!checkVoxelWithinVolume(volume_dims, voxel + z_offset)) return no_normal;

    // also bomb out if any don't have weights..
    float w = weights[getVIndexInt4(volume_dims, voxel)];
    float w_x = weights[getVIndexInt4(volume_dims, voxel + x_offset)];
    float w_y = weights[getVIndexInt4(volume_dims, voxel + y_offset)];
    float w_z = weights[getVIndexInt4(volume_dims, voxel + z_offset)];
    if (w <= 0 || w_x <= 0 || w_y <= 0 || w_z <= 0) return no_normal;


    float v = volume[getVIndexInt4(volume_dims, voxel)];
    float v_x = volume[getVIndexInt4(volume_dims, voxel + x_offset)];
    float v_y = volume[getVIndexInt4(volume_dims, voxel + y_offset)];
    float v_z = volume[getVIndexInt4(volume_dims, voxel + z_offset)];

    float4 normal = (float4)(v_x - v, v_y - v, v_z - v, 0);

    // NOTE UNNORMALIZED
    return normal;
}


__kernel void KernelAddFrameTo2MeansUsingNormals(
        __global float *mean_1,
        __global float *count_1,
        __global float *mean_2,
        __global float *count_2,
        __global float *depth_image,
		__global float4 *normals_image,
        __global int *segments,
        const int which_segment,
        const int4 volume_dims,
        const float voxel_size,
        const float16 pose,
        const float2 focal_lengths,
        const float2 camera_centers,
        const int2 image_dims,
        const float min_truncation_distance
        )
{
        // should consider making this consistent part a function, obviously
        /////// UNCHANGED FROM addFrame
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        float4 voxel_sizes = (float4)(voxel_size);

        // get world coordinates
        float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
        float4 world_point = applyPoseToPoint(pose, voxel_world_point);
        if (world_point.s2 < 0) return;
        float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
        int2 pixel_int = convert_int2_rte(pixel);
        // make sure they fall in the image
        bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 &&
                pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
        if (!on_image) return;
        int pixel_index = getImageIndex(image_dims, pixel_int);
        float depth = depth_image[pixel_index];
        if (! (depth > 0)) return;
        // new, check segment:
        if (which_segment > 0) {
                if (segments[pixel_index] != which_segment) {
                        return;
                }
        }

#define USE_3D_DISTANCE 1
        // This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
        float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
        float depth_norm = length((float4) (xy, depth, 0));
        float voxel_norm = length((float4) (world_point.xyz, 0));
        float d = depth_norm - voxel_norm;
#else
        float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

        if (d < d_min) return; // voxel is past depth


        /////// END: UNCHANGED FROM addFrame

        ////////////////////////////
        // Actually change voxel values
        size_t v_index = getVIndex(volume_dims, i, j, k);

        // empty space uses d_max...
        float d_new = min(d, d_max);

        ///////////////////////////////////////


        //////////////// yeah this is still not working well

#if 0
        // assuming good frame normals, this is actually the right thing
		float4 normal_frame = normals_image[pixel_index];
		// can have invalid normal for valid depth:
		if (!isfinite(normal_frame.s2)) return;
#endif

// instead try normal always to camera?
        float4 normal_frame = (float4)(0,0,-1,0);


        // instead, compute the normal based on surrounding voxels
        float4 normal_1 = getNormalForVoxel(mean_1, volume_dims, ijk4);
		float4 normal_1_in_camera = applyPoseToNormal(pose, normal_1);
        float4 normal_2 = getNormalForVoxel(mean_2, volume_dims, ijk4);
		float4 normal_2_in_camera = applyPoseToNormal(pose, normal_2);

		float normal_dot_1 = dot(normal_frame, normal_1_in_camera);
		float normal_dot_2 = dot(normal_frame, normal_2_in_camera);

		// simplest, stupidest, probably wrong thing
        //if (normal_dot_1 >= normal_dot_2) {
        if (normal_dot_1 >= 0) {
			float mean = mean_1[v_index];
			float count = count_1[v_index];
			float mean_sum = d_new + count * mean;
			float new_count = count + 1;
			float new_mean = mean_sum / new_count;
			mean_1[v_index] = new_mean;
			count_1[v_index] = new_count;
		}
		else {
			float mean = mean_2[v_index];
			float count = count_2[v_index];
			float mean_sum = d_new + count * mean;
			float new_count = count + 1;
			float new_mean = mean_sum / new_count;
			mean_2[v_index] = new_mean;
			count_2[v_index] = new_count;
		}
}


__kernel void KernelAddFrameTo2MeansUsingStoredNormals(
        __global float *mean_1,
        __global float *count_1,
        __global float4 *normals_1,
        __global float *mean_2,
        __global float *count_2,
        __global float4 *normals_2,
        __global float *depth_image,
        __global float4 *normals_image,
        __global int *segments,
        const int which_segment,
        const int4 volume_dims,
        const float voxel_size,
        const float16 model_pose,
        const float16 model_pose_inverse,
        const float2 focal_lengths,
        const float2 camera_centers,
        const int2 image_dims,
        const float min_truncation_distance
        )
{
        // should consider making this consistent part a function, obviously
        /////// UNCHANGED FROM addFrame
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        float4 voxel_sizes = (float4)(voxel_size);

        // get world coordinates
        float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
        float4 world_point = applyPoseToPoint(model_pose, voxel_world_point);
        if (world_point.s2 < 0) return;
        float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
        int2 pixel_int = convert_int2_rte(pixel);
        // make sure they fall in the image
        bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 &&
                pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
        if (!on_image) return;
        int pixel_index = getImageIndex(image_dims, pixel_int);
        float depth = depth_image[pixel_index];
        if (! (depth > 0)) return;
        // new, check segment:
        if (which_segment > 0) {
                if (segments[pixel_index] != which_segment) {
                        return;
                }
        }

#define USE_3D_DISTANCE 1
        // This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
        float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
        float depth_norm = length((float4) (xy, depth, 0));
        float voxel_norm = length((float4) (world_point.xyz, 0));
        float d = depth_norm - voxel_norm;
#else
        float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

        if (d < d_min) return; // voxel is past depth


        /////// END: UNCHANGED FROM addFrame

        ////////////////////////////
        // Actually change voxel values
        size_t v_index = getVIndex(volume_dims, i, j, k);

        // empty space uses d_max...
        float d_new = min(d, d_max);

        ///////////////////////////////////////



#if 0
        // could skip empties?
        if (d_new >= d_max) return; // skip empties
#endif


#if 1
        // assuming good frame normals, this is actually the right thing
        float4 normal_frame = normals_image[pixel_index];
        // can have invalid normal for valid depth:
        if (!isfinite(normal_frame.s2)) return;
#else
        // instead try normal always to camera?
        float4 normal_frame = (float4)(0,0,-1,0);
#endif

        float4 normal_frame_in_model = applyPoseToNormal(model_pose_inverse, normal_frame);

#if 0
        // instead, compute the normal based on surrounding voxels
        float4 normal_1 = getNormalForVoxel(mean_1, volume_dims, ijk4);
        float4 normal_1_in_camera = applyPoseToNormal(pose, normal_1);
        float4 normal_2 = getNormalForVoxel(mean_2, volume_dims, ijk4);
        float4 normal_2_in_camera = applyPoseToNormal(pose, normal_2);
#endif

//#define UPDATE_OPPOSITE

        // used stored normals instead
        float4 normal_1 = normals_1[v_index];
        float4 normal_2 = normals_2[v_index];

        float normal_dot_1 = dot(normal_frame_in_model, normal_1);
        float normal_dot_2 = dot(normal_frame_in_model, normal_2);

        // simplest, stupidest, probably wrong thing
        //if (normal_dot_1 >= normal_dot_2) {
        if (normal_dot_1 >= 0) {
            float mean = mean_1[v_index];
            float count = count_1[v_index];
            float mean_sum = d_new + count * mean;
            float new_count = count + 1;
            float new_mean = mean_sum / new_count;
            mean_1[v_index] = new_mean;
            count_1[v_index] = new_count;

            // also normal
            float4 new_normal = count * normal_1 + normal_frame_in_model;
            normals_1[v_index] = normalize(new_normal);

#ifdef UPDATE_OPPOSITE
            float other_count = count_2[v_index];
            if (other_count <= 0) {
                normals_2[v_index] = -new_normal;
            }
#endif
        }
        else {
            float mean = mean_2[v_index];
            float count = count_2[v_index];
            float mean_sum = d_new + count * mean;
            float new_count = count + 1;
            float new_mean = mean_sum / new_count;
            mean_2[v_index] = new_mean;
            count_2[v_index] = new_count;

            // also normal
            float4 new_normal = count * normal_2 + normal_frame_in_model;
            normals_2[v_index] = normalize(new_normal);

#ifdef UPDATE_OPPOSITE
            float other_count = count_1[v_index];
            if (other_count <= 0) {
                normals_1[v_index] = -new_normal;
            }
#endif
        }
}





__kernel void KernelAddFrame(
        __global float *means,
        __global float *counts,
        __global float *depth_image,
        __global int *segments,
        const int which_segment,
        const int4 volume_dims,
        const float voxel_size,
        const float16 model_pose,
        const float16 model_pose_inverse,
        const float2 focal_lengths,
        const float2 camera_centers,
        const int2 image_dims,
        const float min_truncation_distance
        )
{
        // should consider making this consistent part a function, obviously
        /////// UNCHANGED FROM addFrame
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        float4 voxel_sizes = (float4)(voxel_size);

        // get world coordinates
        float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
        float4 world_point = applyPoseToPoint(model_pose, voxel_world_point);
        if (world_point.s2 < 0) return;
        float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
        int2 pixel_int = convert_int2_rte(pixel);
        // make sure they fall in the image
        bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 &&
                pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
        if (!on_image) return;
        int pixel_index = getImageIndex(image_dims, pixel_int);
        float depth = depth_image[pixel_index];
        if (! (depth > 0)) return;
        // new, check segment:
        if (which_segment > 0) {
                if (segments[pixel_index] != which_segment) {
                        return;
                }
        }

#define USE_3D_DISTANCE 1
        // This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
        float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
        float depth_norm = length((float4) (xy, depth, 0));
        float voxel_norm = length((float4) (world_point.xyz, 0));
        float d = depth_norm - voxel_norm;
#else
        float d = depth - world_point.s2;
#endif

    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

        if (d < d_min) return; // voxel is past depth


        /////// END: UNCHANGED FROM addFrame

        ////////////////////////////
        // Actually change voxel values
        size_t v_index = getVIndex(volume_dims, i, j, k);

        // empty space uses d_max...
        float d_new = min(d, d_max);

        ///////////////////////////////////////



#if 0
        // could skip empties?
        if (d_new >= d_max) return; // skip empties
#endif



		// just always this:
		float mean = means[v_index];
		float count = counts[v_index];
		float mean_sum = d_new + count * mean;
		float new_count = count + 1;
		float new_mean = mean_sum / new_count;
		means[v_index] = new_mean;
		counts[v_index] = new_count;
}


__kernel void KernelAddFrameIfCompatible(
        __global float *means,
        __global float *counts,
        __global float *depth_image,
        __global float4 *normals_image,
        __global int *segments,
        const int which_segment,
        const int4 volume_dims,
        const float voxel_size,
        const float16 model_pose,
        const float16 model_pose_inverse,
        const float2 focal_lengths,
        const float2 camera_centers,
        const int2 image_dims,
        const float4 this_volume_normal,
        const float min_dot_product,
		const int empty_always_included,
        const int cos_weight,
        const float min_truncation_distance
        )
{
        // should consider making this consistent part a function, obviously
        /////// UNCHANGED FROM addFrame
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);
        size_t k = get_global_id(2);
        int4 ijk4 = (int4)(i,j,k,0);

        float4 voxel_sizes = (float4)(voxel_size);

        // get world coordinates
        float4 voxel_world_point = voxelToWorldFloat(voxel_sizes, ijk4);
        float4 world_point = applyPoseToPoint(model_pose, voxel_world_point);
        if (world_point.s2 < 0) return;
        float2 pixel = projectPoint(focal_lengths, camera_centers, world_point);
        int2 pixel_int = convert_int2_rte(pixel);
        // make sure they fall in the image
        bool on_image = (pixel_int.s0 >= 0 && pixel_int.s0 < image_dims.s0 &&
                pixel_int.s1 >= 0 && pixel_int.s1 < image_dims.s1);
        if (!on_image) return;
        int pixel_index = getImageIndex(image_dims, pixel_int);
        float depth = depth_image[pixel_index];
        if (! (depth > 0)) return;
        // new, check segment:
        if (which_segment > 0) {
                if (segments[pixel_index] != which_segment) {
                        return;
                }
        }

#define USE_3D_DISTANCE 1
        // This makes it take like another 10 ms over "float d = depth - world_point.s2;"
#ifdef USE_3D_DISTANCE
        float2 xy = (depth / focal_lengths) * (convert_float2(pixel_int) - camera_centers);
        float depth_norm = length((float4) (xy, depth, 0));
        float voxel_norm = length((float4) (world_point.xyz, 0));
        float d = depth_norm - voxel_norm;
#else
        float d = depth - world_point.s2;
#endif


    // this is getting weirder and weirder:
    float d_min = -MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    float d_max = MIN_VOXELS_IN_TSDF_RAMP * voxel_size;
    d_min = min(d_min, -min_truncation_distance);
    d_max = max(d_max, min_truncation_distance);
#ifdef USE_SIGMA
    float sigma_z = simpleAxial(depth);
    d_min = min(d_min, -SIGMA_COUNT * sigma_z);
    d_max = max(d_max, SIGMA_COUNT * sigma_z);
#endif

        if (d < d_min) return; // voxel is past depth


        /////// END: UNCHANGED FROM addFrame

        ////////////////////////////
        // Actually change voxel values
        size_t v_index = getVIndex(volume_dims, i, j, k);

        // empty space uses d_max...
        float d_new = min(d, d_max);

        ///////////////////////////////////////



#if 0
        // could skip empties?
        if (d_new >= d_max) return; // skip empties
#endif


		/////////// REALLY SHOULD USE SURFACE NORMAL NOT VIEW NORMAL
#if 1
        // assuming good frame normals, this is actually the right thing
        float4 normal_frame = normals_image[pixel_index];
        // can have invalid normal for valid depth:
        if (!isfinite(normal_frame.s2)) return;
#else
        // instead try normal always to camera?
        float4 normal_frame = (float4)(0,0,-1,0);
#endif

        float4 normal_frame_in_model = applyPoseToNormal(model_pose_inverse, normal_frame);

        float normal_dot = dot(normal_frame_in_model, this_volume_normal);

		// can add to empties in all volumes safely (fwiw)
        if (normal_dot >= min_dot_product || (empty_always_included && d_new >= d_max)) {
            float mean = means[v_index];
            float count = counts[v_index];
            float mean_sum = d_new + count * mean;
            float new_count = count + 1;
			if (cos_weight) new_count = count + fabs(normal_dot); // fabs only for weird negative case (not normal usage)
            float new_mean = mean_sum / new_count;
            means[v_index] = new_mean;
            counts[v_index] = new_count;
        }
}



__kernel void KernelDotVolumeNormal(
    __global float *volume,
    __global float *result_dot,
    const int4 volume_dims,
    const float4 vector)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);
    float4 normal = getNormalForVoxel(volume, volume_dims, ijk4);
    result_dot[v_index] = dot(normal, vector);
}


__kernel void KernelComputeNormalVolume(
	__global float *volume,
	__global float4 *result_normal_volume,
	const int4 volume_dims
	)
{
	size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);
    float4 normal = getNormalForVoxel(volume, volume_dims, ijk4);
    result_normal_volume[v_index] = normal;
}

__kernel void KernelComputeNormalVolumeWithWeights(
    __global float *volume,
    __global float *weights,
    __global float4 *result_normal_volume,
    const int4 volume_dims
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);
    float4 normal = getNormalForVoxelWithWeights(volume, weights, volume_dims, ijk4);
    result_normal_volume[v_index] = normal;
}

__kernel void KernelComputeNormalVolumeWithWeightsUnnormalized(
    __global float *volume,
    __global float *weights,
    __global float4 *result_normal_volume,
    const int4 volume_dims
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);
    float4 normal = getNormalForVoxelWithWeightsUnnormalized(volume, weights, volume_dims, ijk4);
    result_normal_volume[v_index] = normal;
}


__kernel void KernelMinAbsVolume(
    __global float *volume_1,
    __global float *volume_counts_1,
    __global float *volume_2,
    __global float *volume_counts_2,
    __global float *result_volume,
    __global float *result_counts,
    const int4 volume_dims,
	float minimum_relative_count
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);

    float count_1 = volume_counts_1[v_index];
    float count_2 = volume_counts_2[v_index];
    float value_1 = volume_1[v_index];
    float value_2 = volume_2[v_index];

    if (count_1 > 0 && count_2 > 0) {
		float count_1_over_count_2 = count_1 / count_2;

        if (fabs(value_1) < fabs(value_2) && count_1_over_count_2 >= minimum_relative_count) {
            result_volume[v_index] = value_1;
            result_counts[v_index] = count_1;
        }
        else {
            result_volume[v_index] = value_2;
            result_counts[v_index] = count_2;
        }
    }
    else if (count_1 > 0) {
        result_volume[v_index] = value_1;
        result_counts[v_index] = count_1;
    }
    else if (count_2 > 0) {
        result_volume[v_index] = value_2;
        result_counts[v_index] = count_2;
    }
    else {
        result_volume[v_index] = -1; // empty value?? This is ugly
        result_counts[v_index] = 0;
    }
}


__kernel void KernelBetterNormal(
    __global float *volume_1,
    __global float *volume_counts_1,
    __global float4 *volume_normals_1,
    __global float *volume_2,
    __global float *volume_counts_2,
    __global float4 *volume_normals_2,
    __global float *result_volume,
    __global float *result_counts,
    // also normal result?
    const int4 volume_dims,
    const float16 pose,
	const float minimum_relative_count
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);

    float count_1 = volume_counts_1[v_index];
    float count_2 = volume_counts_2[v_index];
    float value_1 = volume_1[v_index];
    float value_2 = volume_2[v_index];
    float4 normal_1 = volume_normals_1[v_index];
    float4 normal_2 = volume_normals_2[v_index];
    float4 normal_1_in_camera = applyPoseToNormal(pose, normal_1);
    float4 normal_2_in_camera = applyPoseToNormal(pose, normal_2);

    if (count_1 > 0 && count_2 > 0) {
		float count_1_over_count_2 = count_1 / count_2;

        const float4 to_camera = (float4)(0,0,-1,0);
        float normal_1_dot = dot(to_camera, normal_1_in_camera);
        float normal_2_dot = dot(to_camera, normal_2_in_camera);
        if (normal_1_dot >= normal_2_dot && count_1_over_count_2 >= minimum_relative_count) {
            result_volume[v_index] = value_1;
            result_counts[v_index] = count_1;
        }
        else {
            result_volume[v_index] = value_2;
            result_counts[v_index] = count_2;
        }
    }
    else if (count_1 > 0) {
        result_volume[v_index] = value_1;
        result_counts[v_index] = count_1;
    }
    else if (count_2 > 0) {
        result_volume[v_index] = value_2;
        result_counts[v_index] = count_2;
    }
    else {
        result_volume[v_index] = -1; // empty value?? This is ugly
        result_counts[v_index] = 0;
    }
}



__kernel void KernelAddVolumes(
    __global float *volume_1,
    __global float *volume_counts_1,
    __global float *volume_2,
    __global float *volume_counts_2,
    __global float *result_volume,
    __global float *result_counts,
    const int4 volume_dims
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t k = get_global_id(2);
    int4 ijk4 = (int4)(i,j,k,0);
    size_t v_index = getVIndex(volume_dims, i, j, k);

    float count_1 = volume_counts_1[v_index];
    float count_2 = volume_counts_2[v_index];
    float value_1 = volume_1[v_index];
    float value_2 = volume_2[v_index];

    float result_count = count_1 + count_2;
    float result_value = result_count > 0 ? (value_1 * count_1 + value_2 * count_2) / result_count : -1;
    result_volume[v_index] = result_value;
    result_counts[v_index] = result_count;
}


__kernel void KernelRenderPointsAndNormals(
	__global float *D,
	__global float *DW,
	__global int *rendered_mask,
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	const int4 volume_dims,
	const float voxel_size,
    const float16 volume_pose,
    const float16 volume_pose_inverse,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int2 image_dims,
	const float min_render_depth,
	const float max_render_depth,
	const int replace_only_if_nearer,
	const int mask_value
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	float4 voxel_sizes = (float4)(voxel_size);

	int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

    float4 camera_center = applyPoseToPoint(volume_pose_inverse, (float4)(0));
    float4 pixel_on_focal = pixelOnFocal(focal_lengths, camera_centers, pixel);
    float4 pixel_transformed = applyPoseToPoint(volume_pose_inverse, pixel_on_focal);
	float4 ray_unit_v = (pixel_transformed - camera_center);
	ray_unit_v.s3 = 0;
	ray_unit_v = normalize(ray_unit_v);
        bool found_surface_voxel = false;
        const float reset_d = 1.0;
        float mat_d = reset_d;
        float mat_d_previous = reset_d;
        float t_previous = 0; // will be set if surface is found
        float t = 0; // initialized in loop

        // replace if nearer check
        float previous_distance = 0;
        if (replace_only_if_nearer) {
                // read previous
                float4 previous_point = rendered_points[image_index];
                if (isfinite(previous_point.s2)) {
                        // from later: final_point = applyPoseToPoint(pose, t_star_point);
                        // was:
                        // previous_depth = previous_point.s2;
                        // to think about...points behind camera?
                        previous_distance = distance(camera_center, applyPoseToPoint(volume_pose_inverse, previous_point));
                        //printf("previous distance:%f\n", previous_distance);
                }
        }

        // new initial box check code
        // box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
#define SLABTEST 1
#ifdef SLABTEST
        float4 box_min = voxelToWorldFloat(voxel_sizes, (int4)(-1,-1,-1,0)); // -1 to be "safe"
        float4 box_max = voxelToWorldFloat(voxel_sizes, volume_dims); // volume dims is already "+1"
        float2 t_bounds = box_intersect(camera_center, ray_unit_v, box_min, box_max, min_render_depth, max_render_depth);
        if (!isfinite(t_bounds.s0) || !isfinite(t_bounds.s1)) {
                // only really need to check the first one
                return;
        }
#else
        float2 t_bounds = (float2)(min_render_depth, max_render_depth);
#endif

        const float step_size = voxel_size;
	//for (t = min_render_depth; t <= max_render_depth; t += step_size) {
	for (t = t_bounds.s0; t <= t_bounds.s1; t += step_size) {
		float4 volume_point = camera_center + t * ray_unit_v;

		// only if replace_only_if_nearer
		if (replace_only_if_nearer && previous_distance > 0) {
			if (t > previous_distance) {
				break;
			}
		}

		float4 id_f = worldToVoxelFloat(voxel_sizes, volume_point);
		int4 id_floor = convert_int4(id_f);

		// could "check for never intersecting volume (just an optimization):"
		// This can be commented out without affecting result
#ifndef SLABTEST
		if (rayWillNeverHitVolume(volume_dims, id_f, ray_unit_v)) break;
#endif

		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, id_floor)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		if (!checkAllHaveWeights(volume_dims, id_floor, DW)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		mat_d = trilinearInterpolate(volume_dims, id_f, D);

		// backside:
		if (mat_d > 0 && mat_d_previous < 0) break;

		// frontside:
		if (mat_d < 0 && mat_d_previous > 0 && mat_d_previous < reset_d) {
			found_surface_voxel = true;
			break;
		}

		// these 2 go together:
		mat_d_previous = mat_d;
		t_previous = t;
	}

	if (found_surface_voxel) {
		// DEBUG
		// "paper":
		//float t_star =  t_previous - (t - t_previous) * mat_d_previous / (mat_d - mat_d_previous);
		// stupid:
		//float t_star = t;
		// peter algebra (matches paper I believe)
		float t_star = t - mat_d * (t - t_previous) / (mat_d - mat_d_previous);
		float4 t_star_point = camera_center + t_star * ray_unit_v;
		float4 id_t_star = worldToVoxelFloat(voxel_sizes, t_star_point);

        // to see what lack of normals is costing us, set point always!
        float4 t_star_point_relative_to_camera = applyPoseToPoint(volume_pose, t_star_point);
        // THIS IS TOTALLY WASTEFUL...fix once you debug
        rendered_mask[image_index] = -1;
        //rendered_points[image_index] = t_star_point_relative_to_camera;
        // also idea: can set mask codes depending on various early returns below


		const float normals_volume_delta = 1.f;
		const float4 normal_delta_v = (float4)(normals_volume_delta, normals_volume_delta, normals_volume_delta, 0);
		const float4 normal_delta_v_x = (float4)(normals_volume_delta, 0, 0, 0);
		const float4 normal_delta_v_y = (float4)(0, normals_volume_delta, 0, 0);
		const float4 normal_delta_v_z = (float4)(0, 0, normals_volume_delta, 0);

		float4 id_t_star_check_pos = id_t_star + normal_delta_v;
		float4 id_t_star_x_pos = id_t_star + normal_delta_v_x;
		float4 id_t_star_y_pos = id_t_star + normal_delta_v_y;
		float4 id_t_star_z_pos = id_t_star + normal_delta_v_z;

		// bomb out if normal check exits volume
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star))) return;
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star_check_pos))) return;

		// bomb out if any of the normal  of the normal checks don't have weights
		// this is the new check that solves the rendering bug
		// note there are redundant checks in here
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_x_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_y_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_z_pos), DW)) return;

		// THIS IS TOTALLY WASTEFUL...fix once you debug
        rendered_mask[image_index] = -2;

		// could also assume this d_t_star is 0 (which is of course the goal)
		float d_t_star = trilinearInterpolate(volume_dims, id_t_star, D);
		float d_t_star_x_pos = trilinearInterpolate(volume_dims, id_t_star_x_pos, D);
		float d_t_star_y_pos = trilinearInterpolate(volume_dims, id_t_star_y_pos, D);
		float d_t_star_z_pos = trilinearInterpolate(volume_dims, id_t_star_z_pos, D);

		float4 normal = (float4)
			(
			(d_t_star_x_pos - d_t_star),
			(d_t_star_y_pos - d_t_star),
			(d_t_star_z_pos - d_t_star),
			0
			);
		normal = normalize(normal);
		// got normal

		// whew!  Can finally set some output values
        float4 normal_relative_to_camera = applyPoseToNormal(volume_pose, normal);

		// check that normal points towards camera...if not, we've got a "back" surface (no point flipping it)
		if (normal_relative_to_camera.s2 > 0) return;

		// HERE WE ACTUALLY RENDER SOMETHING
		rendered_mask[image_index] = mask_value;
		rendered_points[image_index] = t_star_point_relative_to_camera;
		rendered_normals[image_index] = normal_relative_to_camera;
	}
}

__kernel void KernelRenderPoints(
	__global float *D,
	__global float *DW,
	__global int *rendered_mask,
	__global float4 *rendered_points,
	const int4 volume_dims,
	const float voxel_size,
    const float16 volume_pose,
    const float16 volume_pose_inverse,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int2 image_dims,
	const float min_render_depth,
	const float max_render_depth,
	const int replace_only_if_nearer,
	const int mask_value
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	float4 voxel_sizes = (float4)(voxel_size);

	int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

    float4 camera_center = applyPoseToPoint(volume_pose_inverse, (float4)(0));
    float4 pixel_on_focal = pixelOnFocal(focal_lengths, camera_centers, pixel);
    float4 pixel_transformed = applyPoseToPoint(volume_pose_inverse, pixel_on_focal);
	float4 ray_unit_v = (pixel_transformed - camera_center);
	ray_unit_v.s3 = 0;
	ray_unit_v = normalize(ray_unit_v);
        bool found_surface_voxel = false;
        const float reset_d = 1.0;
        float mat_d = reset_d;
        float mat_d_previous = reset_d;
        float t_previous = 0; // will be set if surface is found
        float t = 0; // initialized in loop

        // replace if nearer check
        float previous_distance = 0;
        if (replace_only_if_nearer) {
                // read previous
                float4 previous_point = rendered_points[image_index];
                if (isfinite(previous_point.s2)) {
                        // from later: final_point = applyPoseToPoint(pose, t_star_point);
                        // was:
                        // previous_depth = previous_point.s2;
                        // to think about...points behind camera?
                        previous_distance = distance(camera_center, applyPoseToPoint(volume_pose_inverse, previous_point));
                        //printf("previous distance:%f\n", previous_distance);
                }
        }

        // new initial box check code
        // box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
#define SLABTEST 1
#ifdef SLABTEST
        float4 box_min = voxelToWorldFloat(voxel_sizes, (int4)(-1,-1,-1,0)); // -1 to be "safe"
        float4 box_max = voxelToWorldFloat(voxel_sizes, volume_dims); // volume dims is already "+1"
        float2 t_bounds = box_intersect(camera_center, ray_unit_v, box_min, box_max, min_render_depth, max_render_depth);
        if (!isfinite(t_bounds.s0) || !isfinite(t_bounds.s1)) {
                // only really need to check the first one
                return;
        }
#else
        float2 t_bounds = (float2)(min_render_depth, max_render_depth);
#endif

        const float step_size = voxel_size;
	//for (t = min_render_depth; t <= max_render_depth; t += step_size) {
	for (t = t_bounds.s0; t <= t_bounds.s1; t += step_size) {
		float4 volume_point = camera_center + t * ray_unit_v;

		// only if replace_only_if_nearer
		if (replace_only_if_nearer && previous_distance > 0) {
			if (t > previous_distance) {
				break;
			}
		}

		float4 id_f = worldToVoxelFloat(voxel_sizes, volume_point);
		int4 id_floor = convert_int4(id_f);

		// could "check for never intersecting volume (just an optimization):"
		// This can be commented out without affecting result
#ifndef SLABTEST
		if (rayWillNeverHitVolume(volume_dims, id_f, ray_unit_v)) break;
#endif

		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, id_floor)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		if (!checkAllHaveWeights(volume_dims, id_floor, DW)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}

		mat_d = trilinearInterpolate(volume_dims, id_f, D);

		// backside:
		if (mat_d > 0 && mat_d_previous < 0) break;

		// frontside:
		if (mat_d < 0 && mat_d_previous > 0 && mat_d_previous < reset_d) {
			found_surface_voxel = true;
			break;
		}

		// these 2 go together:
		mat_d_previous = mat_d;
		t_previous = t;
	}

	// could move this into "backside" and "frontside"
	// in particular, backside should set something to avoid leaking through backs of surfaces in one volume to fronts in other
	if (found_surface_voxel) {
		// DEBUG
		// "paper":
		//float t_star =  t_previous - (t - t_previous) * mat_d_previous / (mat_d - mat_d_previous);
		// stupid:
		//float t_star = t;
		// peter algebra (matches paper I believe)
		float t_star = t - mat_d * (t - t_previous) / (mat_d - mat_d_previous);
		float4 t_star_point = camera_center + t_star * ray_unit_v;
		float4 t_star_point_relative_to_camera = applyPoseToPoint(volume_pose, t_star_point);

		rendered_mask[image_index] = mask_value;
		rendered_points[image_index] = t_star_point_relative_to_camera;
	}
}

__kernel void KernelRenderNormalForPoints(
	__global float *D,
	__global float *DW,
	__global int *rendered_mask,
    __global float4 *rendered_points,
    __global float4 *rendered_normals, // filled in by this
    const int4 volume_dims,
    const float voxel_size,
    const float16 volume_pose,
    const float16 volume_pose_inverse,
    const int2 image_dims,
	const int mask_value
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    float4 voxel_sizes = (float4)(voxel_size);

    int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

	int rendered_mask_value = rendered_mask[image_index];
	if (rendered_mask_value != mask_value) return;

    float4 rendered_point = rendered_points[image_index];
    if (!isfinite(rendered_point.s2)) return;

    float4 point_in_volume_frame = applyPoseToPoint(volume_pose_inverse, rendered_point);
    float4 voxel_f = worldToVoxelFloat(voxel_sizes, point_in_volume_frame);

	// oh wait...this is going to look up ALL points, even those from other volumes...
	// now that I'm doing the mask check, this shouldn't be necessary, but it is safe...
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f))) return;

	////////
	// actually compute and store the normal
	// function for alternatives?

	const float normals_volume_delta = 1.f;
	// this was for "fast" checking:
	//const float4 normal_delta_v = (float4)(normals_volume_delta, normals_volume_delta, normals_volume_delta, 0);
	const float4 normal_delta_v_x = (float4)(normals_volume_delta, 0, 0, 0);
	const float4 normal_delta_v_y = (float4)(0, normals_volume_delta, 0, 0);
	const float4 normal_delta_v_z = (float4)(0, 0, normals_volume_delta, 0);

	float4 voxel_f_x = voxel_f + normal_delta_v_x;
	float4 voxel_f_y = voxel_f + normal_delta_v_y;
	float4 voxel_f_z = voxel_f + normal_delta_v_z;

	// definitely check against volume dims (some redundant checks here, but fast and simple)
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f))) return;
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f_x))) return;
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f_y))) return;
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f_z))) return;

	// bomb out if any of the normal  of the normal checks don't have weights
	// this is the new check that solves the rendering bug (but also causes many to not have normals)
	// note there are redundant checks in here (which are SLOW global lookups)
	if (!checkAllHaveWeights(volume_dims, convert_int4(voxel_f), DW)) return;
	if (!checkAllHaveWeights(volume_dims, convert_int4(voxel_f_x), DW)) return;
	if (!checkAllHaveWeights(volume_dims, convert_int4(voxel_f_y), DW)) return;
	if (!checkAllHaveWeights(volume_dims, convert_int4(voxel_f_z), DW)) return;

	float d_t_star = trilinearInterpolate(volume_dims, voxel_f, D); // should be 0 of course!
	float d_t_star_x_pos = trilinearInterpolate(volume_dims, voxel_f_x, D);
	float d_t_star_y_pos = trilinearInterpolate(volume_dims, voxel_f_y, D);
	float d_t_star_z_pos = trilinearInterpolate(volume_dims, voxel_f_z, D);

	float4 normal = (float4) ( (d_t_star_x_pos - d_t_star), (d_t_star_y_pos - d_t_star), (d_t_star_z_pos - d_t_star), 0 );
	normal = normalize(normal);

	float4 normal_relative_to_camera = applyPoseToNormal(volume_pose, normal);
	rendered_normals[image_index] = normal_relative_to_camera;
}


__kernel void KernelRenderColorForPoints(
    __global uchar4 *C,
	__global int *rendered_mask,
    __global float4 *rendered_points,
    __global uchar4 *rendered_colors, // filled in by this
    const int4 volume_dims,
    const float voxel_size,
    const float16 volume_pose,
    const float16 volume_pose_inverse,
    const int2 image_dims,
	const int mask_value
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    float4 voxel_sizes = (float4)(voxel_size);

    int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

	int rendered_mask_value = rendered_mask[image_index];
	if (rendered_mask_value != mask_value) return;

    float4 rendered_point = rendered_points[image_index];
    if (!isfinite(rendered_point.s2)) return;

    float4 point_in_volume_frame = applyPoseToPoint(volume_pose_inverse, rendered_point);
    float4 voxel_f = worldToVoxelFloat(voxel_sizes, point_in_volume_frame);

	// oh wait...this is going to look up ALL points, even those from other volumes...
	// now that I'm doing the mask check, this shouldn't be necessary, but it is safe...
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(voxel_f))) return;

    uchar4 color = trilinearInterpolateColor(volume_dims, voxel_f, C);
    rendered_colors[image_index] = color;
}


//// not finished or working:
__kernel void KernelRenderMax(
        __global float *volume,
        __global float *volume_weights,
        __global int *rendered_mask,
        __global float4 *rendered_points,
        __global float4 *rendered_normals,
        const int4 volume_dims,
        const float voxel_size,
        const float16 pose,
        const float16 inverse_pose,
        const float2 focal_lengths,
        const float2 camera_centers,
        const int2 image_dims,
        const float min_render_depth,
        const float max_render_depth,
        const int replace_only_if_nearer,
        const int mask_value
        )
{
        size_t i = get_global_id(0);
        size_t j = get_global_id(1);

        float4 voxel_sizes = (float4)(voxel_size);

        int2 pixel = (int2)(i,j);
        int image_index = getImageIndex(image_dims, pixel);

        float4 camera_center = applyPoseToPoint(inverse_pose, (float4)(0));
        float4 pixel_on_focal = pixelOnFocal(focal_lengths, camera_centers, pixel);
        float4 pixel_transformed = applyPoseToPoint(inverse_pose, pixel_on_focal);
        float4 ray_unit_v = (pixel_transformed - camera_center);
        ray_unit_v.s3 = 0;
        ray_unit_v = normalize(ray_unit_v);

        // replace if nearer check
        float previous_distance = 0;
        if (replace_only_if_nearer) {
                // read previous
                float4 previous_point = rendered_points[image_index];
                if (isfinite(previous_point.s2)) {
                        // from later: final_point = applyPoseToPoint(pose, t_star_point);
                        // was:
                        // previous_depth = previous_point.s2;
                        // to think about...points behind camera?
                        previous_distance = distance(camera_center, applyPoseToPoint(inverse_pose, previous_point));
                        //printf("previous distance:%f\n", previous_distance);
                }
        }

        // new initial box check code
        // box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
#define SLABTEST 1
#ifdef SLABTEST
        float4 box_min = voxelToWorldFloat(voxel_sizes, (int4)(-1,-1,-1,0)); // -1 to be "safe"
        float4 box_max = voxelToWorldFloat(voxel_sizes, volume_dims); // volume dims is already "+1"
        float2 t_bounds = box_intersect(camera_center, ray_unit_v, box_min, box_max, min_render_depth, max_render_depth);
        if (!isfinite(t_bounds.s0) || !isfinite(t_bounds.s1)) {
                // only really need to check the first one
                return;
        }
#else
        float2 t_bounds = (float2)(min_render_depth, max_render_depth);
#endif

        ////////////////////
        // ok, now start stepping
        bool found_surface_voxel = false;
        const float reset_d = 1.0;
        float mat_d = reset_d;
        float mat_d_previous = reset_d;
        float t_previous = 0; // will be set if surface is found
        float t = 0; // initialized in loop

        // max finding stuff
        #define MAX_WINDOW 5
        float max_window[MAX_WINDOW];
        for (size_t i = 0; i < MAX_WINDOW; ++i) max_window[i] = 0;



        const float step_size = voxel_size;
        for (t = t_bounds.s0; t <= t_bounds.s1; t += step_size) {
                float4 volume_point = camera_center + t * ray_unit_v;

                // only if replace_only_if_nearer
                if (replace_only_if_nearer && previous_distance > 0) {
                        if (t > previous_distance) {
                                break;
                        }
                }

                float4 id_f = worldToVoxelFloat(voxel_sizes, volume_point);
                int4 id_floor = convert_int4(id_f);

                // this is the original code:
                #if 0
                if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, id_floor)) {
                        mat_d_previous = reset_d;
                        t_previous = t;
                        continue;
                }

                if (!checkAllHaveWeights(volume_dims, id_floor, volume_weights)) {
                        mat_d_previous = reset_d;
                        t_previous = t;
                        continue;
                }



                mat_d = trilinearInterpolate(volume_dims, id_f, D);

                // backside:
                if (mat_d > 0 && mat_d_previous < 0) break;

                // frontside:
                if (mat_d < 0 && mat_d_previous > 0 && mat_d_previous < reset_d) {
                        found_surface_voxel = true;
                        break;
                }
                #endif

                // instead, just do nearest voxel for now
                int4 id_nearest = convert_int4_rte(id_f);
                if (!checkVoxelWithinVolume(volume_dims, id_nearest)) {
                    mat_d_previous = reset_d;
                    t_previous = t;
                    continue;
                }

                int i = id_nearest.s0;
                int j = id_nearest.s1;
                int k = id_nearest.s2;
                size_t v_index = getVIndex(volume_dims, i, j, k);
                mat_d = volume[v_index];
                // weight for something?
                float mat_dw = volume_weights[v_index];

                // i dunno..do something with these



                // these 2 go together:
                mat_d_previous = mat_d;
                t_previous = t;
        }

        // old found surface foxel stuff:
#if 0
        if (found_surface_voxel) {
                // DEBUG
                // "paper":
                //float t_star =  t_previous - (t - t_previous) * mat_d_previous / (mat_d - mat_d_previous);
                // stupid:
                //float t_star = t;
                // peter algebra (matches paper I believe)
                float t_star = t - mat_d * (t - t_previous) / (mat_d - mat_d_previous);
                float4 t_star_point = camera_center + t_star * ray_unit_v;
                float4 id_t_star = worldToVoxelFloat(voxel_sizes, t_star_point);

                const float normals_volume_delta = 1.f;
                const float4 normal_delta_v = (float4)(normals_volume_delta, normals_volume_delta, normals_volume_delta, 0);
                const float4 normal_delta_v_x = (float4)(normals_volume_delta, 0, 0, 0);
                const float4 normal_delta_v_y = (float4)(0, normals_volume_delta, 0, 0);
                const float4 normal_delta_v_z = (float4)(0, 0, normals_volume_delta, 0);

                float4 id_t_star_check_pos = id_t_star + normal_delta_v;
                float4 id_t_star_x_pos = id_t_star + normal_delta_v_x;
                float4 id_t_star_y_pos = id_t_star + normal_delta_v_y;
                float4 id_t_star_z_pos = id_t_star + normal_delta_v_z;

                // bomb out if normal check exits volume
                if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star))) return;
                if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star_check_pos))) return;

                // bomb out if any of the normal  of the normal checks don't have weights
                // this is the new check that solves the rendering bug
                // note there are redundant checks in here
                if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star), DW)) return;
                if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_x_pos), DW)) return;
                if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_y_pos), DW)) return;
                if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_z_pos), DW)) return;

                // could also assume this d_t_star is 0 (which is of course the goal)
                float d_t_star = trilinearInterpolate(volume_dims, id_t_star, D);
                float d_t_star_x_pos = trilinearInterpolate(volume_dims, id_t_star_x_pos, D);
                float d_t_star_y_pos = trilinearInterpolate(volume_dims, id_t_star_y_pos, D);
                float d_t_star_z_pos = trilinearInterpolate(volume_dims, id_t_star_z_pos, D);

                float4 normal = (float4)
                        (
                        (d_t_star_x_pos - d_t_star),
                        (d_t_star_y_pos - d_t_star),
                        (d_t_star_z_pos - d_t_star),
                        0
                        );
                normal = normalize(normal);
                // got normal

                // whew!  Can finally set some output values
                float4 t_star_point_relative_to_camera = applyPoseToPoint(pose, t_star_point);
                float4 normal_relative_to_camera = applyPoseToNormal(pose, normal);

                // check that normal points towards camera...if not, we've got a "back" surface (no point flipping it)
                if (normal_relative_to_camera.s2 > 0) return;

                // HERE WE ACTUALLY RENDER SOMETHING
                rendered_mask[image_index] = mask_value;
                rendered_points[image_index] = t_star_point_relative_to_camera;
                rendered_normals[image_index] = normal_relative_to_camera;
        }
        #endif
}

bool getLowerAbsValue(
	__global float *D_1,
	__global float *DW_1,
	__global float *D_2,
	__global float *DW_2,
	const int4 volume_dims,
	const float4 id_f,
	const int4 id_floor,
	float* result
	)
{
	// I'm sure this could be made more efficient:
	bool dw_1_all = checkAllHaveWeights(volume_dims, id_floor, DW_1);
	bool dw_2_all = checkAllHaveWeights(volume_dims, id_floor, DW_2);

	if (dw_1_all && dw_2_all) {
		float d_1_interp = trilinearInterpolate(volume_dims, id_f, D_1);
		float d_2_interp = trilinearInterpolate(volume_dims, id_f, D_2);
		float d_1_abs = fabs(d_1_interp);
		float d_2_abs = fabs(d_2_interp);

		if (d_1_abs <= d_2_abs) {
			*result = d_1_interp;
		}
		else {
			*result = d_2_interp;
		}
	}
	else if (dw_1_all) {
		float d_1_interp = trilinearInterpolate(volume_dims, id_f, D_1);
		*result = d_1_interp;
	}
	else if (dw_2_all) {
		float d_2_interp = trilinearInterpolate(volume_dims, id_f, D_2);
		*result = d_2_interp;
	}
	else {
		// not enough weights in either
		return false;
	}

	return true;
}


__kernel void KernelRender2MeansAbs(
	__global float *D_1,
	__global float *DW_1,
	__global float *D_2,
	__global float *DW_2,
	__global int *rendered_mask,
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	const int4 volume_dims,
	const float voxel_size,
	const float16 pose,
	const float16 inverse_pose,
	const float2 focal_lengths,
	const float2 camera_centers,
	const int2 image_dims,
	const float min_render_depth,
	const float max_render_depth,
	const int replace_only_if_nearer,
	const int mask_value
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	float4 voxel_sizes = (float4)(voxel_size);

	int2 pixel = (int2)(i,j);
        int image_index = getImageIndex(image_dims, pixel);

	float4 camera_center = applyPoseToPoint(inverse_pose, (float4)(0));
        float4 pixel_on_focal = pixelOnFocal(focal_lengths, camera_centers, pixel);
	float4 pixel_transformed = applyPoseToPoint(inverse_pose, pixel_on_focal);
	float4 ray_unit_v = (pixel_transformed - camera_center);
	ray_unit_v.s3 = 0;
	ray_unit_v = normalize(ray_unit_v);
        bool found_surface_voxel = false;
        const float reset_d = 1.0;
        float mat_d = reset_d;
        float mat_d_previous = reset_d;
        float t_previous = 0; // will be set if surface is found
        float t = 0; // initialized in loop

        // replace if nearer check
        float previous_distance = 0;
        if (replace_only_if_nearer) {
                // read previous
                float4 previous_point = rendered_points[image_index];
                if (isfinite(previous_point.s2)) {
                        // from later: final_point = applyPoseToPoint(pose, t_star_point);
                        // was:
                        // previous_depth = previous_point.s2;
                        // to think about...points behind camera?
                        previous_distance = distance(camera_center, applyPoseToPoint(inverse_pose, previous_point));
                        //printf("previous distance:%f\n", previous_distance);
                }
        }

        // new initial box check code
        // box_intersect(float4 camera_center, float4 ray_unit_v, float4 box_min, float4 box_max, float min_t, float max_t)
#define SLABTEST 1
#ifdef SLABTEST
        float4 box_min = voxelToWorldFloat(voxel_sizes, (int4)(-1,-1,-1,0)); // -1 to be "safe"
        float4 box_max = voxelToWorldFloat(voxel_sizes, volume_dims); // volume dims is already "+1"
        float2 t_bounds = box_intersect(camera_center, ray_unit_v, box_min, box_max, min_render_depth, max_render_depth);
        if (!isfinite(t_bounds.s0) || !isfinite(t_bounds.s1)) {
                // only really need to check the first one
                return;
        }
#else
        float2 t_bounds = (float2)(min_render_depth, max_render_depth);
#endif

        const float step_size = voxel_size;
	//for (t = min_render_depth; t <= max_render_depth; t += step_size) {
	for (t = t_bounds.s0; t <= t_bounds.s1; t += step_size) {
		float4 volume_point = camera_center + t * ray_unit_v;

		// only if replace_only_if_nearer
		if (replace_only_if_nearer && previous_distance > 0) {
			if (t > previous_distance) {
				break;
			}
		}

		float4 id_f = worldToVoxelFloat(voxel_sizes, volume_point);
		int4 id_floor = convert_int4(id_f);

		// could "check for never intersecting volume (just an optimization):"
		// This can be commented out without affecting result
#ifndef SLABTEST
		if (rayWillNeverHitVolume(volume_dims, id_f, ray_unit_v)) break;
#endif

		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, id_floor)) {
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}


		//////////////////
		// gotta change from here down:


		float value = 0;
		bool got_value = getLowerAbsValue(D_1, DW_1, D_2, DW_2, volume_dims, id_f, id_floor, &value);
		if (got_value) {
			mat_d = value;
		}
		else {
			// not enough weights in either
			mat_d_previous = reset_d;
			t_previous = t;
			continue;
		}


		// back to standard logic:

		// backside:
		if (mat_d > 0 && mat_d_previous < 0) break;

		// frontside:
		if (mat_d < 0 && mat_d_previous > 0 && mat_d_previous < reset_d) {
			found_surface_voxel = true;
			break;
		}

		// these 2 go together:
		mat_d_previous = mat_d;
		t_previous = t;
	}

	if (found_surface_voxel) {
		// DEBUG
		// "paper":
		//float t_star =  t_previous - (t - t_previous) * mat_d_previous / (mat_d - mat_d_previous);
		// stupid:
		//float t_star = t;
		// peter algebra (matches paper I believe)
		float t_star = t - mat_d * (t - t_previous) / (mat_d - mat_d_previous);
		float4 t_star_point = camera_center + t_star * ray_unit_v;
		float4 id_t_star = worldToVoxelFloat(voxel_sizes, t_star_point);

		const float normals_volume_delta = 1.f;
		const float4 normal_delta_v = (float4)(normals_volume_delta, normals_volume_delta, normals_volume_delta, 0);
		const float4 normal_delta_v_x = (float4)(normals_volume_delta, 0, 0, 0);
		const float4 normal_delta_v_y = (float4)(0, normals_volume_delta, 0, 0);
		const float4 normal_delta_v_z = (float4)(0, 0, normals_volume_delta, 0);

		float4 id_t_star_check_pos = id_t_star + normal_delta_v;
		float4 id_t_star_x_pos = id_t_star + normal_delta_v_x;
		float4 id_t_star_y_pos = id_t_star + normal_delta_v_y;
		float4 id_t_star_z_pos = id_t_star + normal_delta_v_z;

		// bomb out if normal check exits volume
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star))) return;
		if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, convert_int4(id_t_star_check_pos))) return;

		/////////////////
		// new stuff for 2 means

		// as Richard said, just increase the truncation distance...
#if 0
		// bomb out if any of the normal  of the normal checks don't have weights
		// this is the new check that solves the rendering bug
		// note there are redundant checks in here
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_x_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_y_pos), DW)) return;
		if (!checkAllHaveWeights(volume_dims, convert_int4(id_t_star_z_pos), DW)) return;

		// could also assume this d_t_star is 0 (which is of course the goal)
		float d_t_star = trilinearInterpolate(volume_dims, id_t_star, D);
		float d_t_star_x_pos = trilinearInterpolate(volume_dims, id_t_star_x_pos, D);
		float d_t_star_y_pos = trilinearInterpolate(volume_dims, id_t_star_y_pos, D);
		float d_t_star_z_pos = trilinearInterpolate(volume_dims, id_t_star_z_pos, D);
#endif

		// we will ASSUME d_t_star is 0 now!  Yay...
		float d_t_star_x_pos;
		bool got_d_t_star_x_pos = getLowerAbsValue(D_1, DW_1, D_2, DW_2, volume_dims, id_t_star_x_pos, convert_int4(id_t_star_x_pos), &d_t_star_x_pos);
		float d_t_star_y_pos;
		bool got_d_t_star_y_pos = getLowerAbsValue(D_1, DW_1, D_2, DW_2, volume_dims, id_t_star_y_pos, convert_int4(id_t_star_y_pos), &d_t_star_y_pos);
		float d_t_star_z_pos;
		bool got_d_t_star_z_pos = getLowerAbsValue(D_1, DW_1, D_2, DW_2, volume_dims, id_t_star_z_pos, convert_int4(id_t_star_z_pos), &d_t_star_x_pos);

		if (! (got_d_t_star_x_pos && got_d_t_star_y_pos && got_d_t_star_z_pos) ) return; // didn't get all values for normal

		// just assume this was 0.  Could actually get it like everything else...
		float d_t_star = 0;

		//////////
		// the rest should be the same...

		float4 normal = (float4)
			(
			(d_t_star_x_pos - d_t_star),
			(d_t_star_y_pos - d_t_star),
			(d_t_star_z_pos - d_t_star),
			0
			);
		normal = normalize(normal);
		// got normal

		// whew!  Can finally set some output values
		float4 t_star_point_relative_to_camera = applyPoseToPoint(pose, t_star_point);
		float4 normal_relative_to_camera = applyPoseToNormal(pose, normal);

		// check that normal points towards camera...if not, we've got a "back" surface (no point flipping it)
		if (normal_relative_to_camera.s2 > 0) return;

		// HERE WE ACTUALLY RENDER SOMETHING
		rendered_mask[image_index] = mask_value;
		rendered_points[image_index] = t_star_point_relative_to_camera;
		rendered_normals[image_index] = normal_relative_to_camera;
	}
}


// from normals.cl
__kernel void KernelNormalsToShadedImage(
        __global float4 *normals,
        __global uchar4 *image,
        const int2 image_dims,
        const float4 vector_to_light
        )
{
        int2 pixel;
        pixel.s0 = get_global_id(0);
        pixel.s1 = get_global_id(1);

        int image_index = getImageIndex(image_dims, pixel);
        float4 normal = normals[image_index];

        uchar4 result = (uchar4)(0);
        if (isfinite(normal.s2)) {
                const float ambient = 0.2;
                const float diffuse = 1.0 - ambient;
                float dot_prod = dot(normal, vector_to_light);
                float intensity = ambient + diffuse * max(0.f, dot_prod);
                uchar intensity_char = convert_uchar(intensity * 255);
                result = (uchar4)(intensity_char, intensity_char, intensity_char, 255);
        }

        image[image_index] = result;
}

// from normals.cl
__kernel void KernelNormalsToColorImage(
        __global float4 *normals,
        __global uchar4 *image,
        const int2 image_dims
        )
{
        int2 pixel;
        pixel.s0 = get_global_id(0);
        pixel.s1 = get_global_id(1);

        int image_index = getImageIndex(image_dims, pixel);
        float4 normal = normals[image_index];

        uchar4 result = (uchar4)(0);
        if (isfinite(normal.s2)) {
                // bgr
                result.s2 = (uchar) clamp ( (normal.s0 + 1.f) * .5f * 255.f, 0.f, 255.f);
                result.s1 = (uchar) clamp ( (normal.s1 + 1.f) * .5f * 255.f, 0.f, 255.f);
                result.s0 = (uchar) clamp ( max(-normal.s2 , 0.f) * 255.f, 0.f, 255.f);
                result.s3 = 255;
        }

        image[image_index] = result;
}


__kernel void KernelPointsToDepthImage(
        __global float4 *points,
        __global float *depth_image,
        const int2 image_dims
        )
{
        int2 pixel;
        pixel.s0 = get_global_id(0);
        pixel.s1 = get_global_id(1);

        int image_index = getImageIndex(image_dims, pixel);
        float z = points[image_index].z;

        float depth = 0;
        if (isfinite(z)) depth = z;

        depth_image[image_index] = depth;
}


__kernel void KernelExtractFloat4ForPointImage(
	__global float4 *volume,
	__global float4 *points_image,
	__global float4 *result_image,
	const float16 model_pose,
	const float16 model_pose_inverse,
	const int4 volume_dims,
	const float voxel_size,
	const int2 image_dims
	)
{
	const float4 empty_value = nan((uint4)0); // not sure what (if any) this should be.  zero?

	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t i_index = getImageIndex(image_dims, (int2)(i,j));
	float4 point = points_image[i_index];
	if (!isfinite(point.s2)) {
		result_image[i_index] = empty_value;
		return;
	}

	float4 point_in_volume = applyPoseToPoint(model_pose_inverse, point);
	float4 voxel_f = point_in_volume / (float4)(voxel_size);
	int4 voxel = convert_int4_rte(voxel_f);
	if (!checkVoxelWithinVolume(volume_dims, voxel)) {
		result_image[i_index] = empty_value;
		return;
	}

	size_t v_index = getVIndexInt4(volume_dims, voxel);
	result_image[i_index] = volume[v_index];
}


__kernel void KernelExtractIntForPointImage(
	__global int *volume,
	__global float4 *points_image,
	__global int *result_image,
	const float16 model_pose,
	const float16 model_pose_inverse,
	const int4 volume_dims,
	const float voxel_size,
	const int2 image_dims
	)
{
	const int empty_value = -1;

	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t i_index = getImageIndex(image_dims, (int2)(i,j));
	float4 point = points_image[i_index];
	if (!isfinite(point.s2)) {
		result_image[i_index] = empty_value;
		return;
	}

	float4 point_in_volume = applyPoseToPoint(model_pose_inverse, point);
	float4 voxel_f = point_in_volume / (float4)(voxel_size);
	int4 voxel = convert_int4_rte(voxel_f);
	if (!checkVoxelWithinVolume(volume_dims, voxel)) {
		result_image[i_index] = empty_value;
		return;
	}

	size_t v_index = getVIndexInt4(volume_dims, voxel);
	result_image[i_index] = volume[v_index];
}


// in place
__kernel void KernelApplyPoseToPoints(
	__global float4 *buffer,
	const float16 pose
	)
{
	size_t i = get_global_id(0);
	buffer[i] = applyPoseToPoint(pose, buffer[i]);
}

// in place
__kernel void KernelApplyPoseToNormals(
	__global float4 *buffer,
	const float16 pose
	)
{
	size_t i = get_global_id(0);
	buffer[i] = applyPoseToNormal(pose, buffer[i]);
}

__kernel void KernelSetVolumeSDFBox(
	__global float * volume,
	const int4 volume_dims,
	const float voxel_size,
	const float16 volume_pose,
	const int2 camera_size,
	const float2 camera_focal,
	const float2 camera_center,
	const float4 box_corner_from_origin,
	const float16 box_pose,
	const float16 box_pose_inverse
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	size_t k = get_global_id(2);
	int4 ijk4 = (int4)(i,j,k,0);
	float4 voxel_sizes = (float4)(voxel_size);
	size_t v_index = getVIndexInt4(volume_dims, ijk4);

	float4 voxel_point = voxelToWorldFloat(voxel_sizes, ijk4);
	float4 point_in_world = applyPoseToPoint(volume_pose, voxel_point);
	float4 point_in_box = applyPoseToPoint(box_pose_inverse, point_in_world);

#if 0
	http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
	float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) +
         length(max(d,0.0));
}
#endif

	// careful...last values both either 1 or 0!!
	float4 d = fabs(point_in_box) - box_corner_from_origin;
	
	float result = min(max(d.s0, max(d.s1, d.s2)), 0.f) + length(max(d, (float4)(0)));
	volume[v_index] = result;
}



__kernel void KernelRaytraceBox(
	__global int *rendered_mask,
	__global float4 *rendered_points,
	__global float4 *rendered_normals,
	const float16 model_pose,
	const float16 model_pose_inverse,
	const float16 box_pose,
	const float16 box_pose_inverse,
	const float4 box_corner_from_origin,
	const float2 camera_focal,
	const float2 camera_center,
	const int2 image_dims,
	const float min_render_depth,
	const float max_render_depth,
	const int mask_value
	)
{
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);
	int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

	// this is sort of borrowed from other renders...
	// should probably get the ray first, then start messing with poses...
	float4 camera_cop = applyPoseToPoint(model_pose_inverse, (float4)(0,0,0,1));
	float4 pixel_on_focal = pixelOnFocal(camera_focal, camera_center, pixel);
	float4 pixel_transformed = applyPoseToPoint(model_pose_inverse, pixel_on_focal);
	float4 ray_unit_v = (pixel_transformed - camera_cop);
	ray_unit_v.s3 = 0;
	ray_unit_v = normalize(ray_unit_v);

	// put points from WORLD into BOX
	float4 camera_cop_box = applyPoseToPoint(box_pose_inverse, camera_cop);
	float4 ray_unit_v_box = applyPoseToNormal(box_pose_inverse, ray_unit_v);

	float4 box_min = -box_corner_from_origin;
	box_min.s3 = 1; // sigh
	float4 box_max = box_corner_from_origin;
	box_max.s3 = 1; // probably already true

	float2 t_bounds = box_intersect(camera_cop_box, ray_unit_v_box, box_min, box_max, min_render_depth, max_render_depth);

	// actual code:
	if (!isfinite(t_bounds.s0)) {
		return;
	}

	float4 t_point_box = camera_cop_box + t_bounds.s0 * ray_unit_v_box;
	float4 t_point_world = applyPoseToPoint(box_pose, t_point_box);

	// this is awkward:
	float4 t_point_camera = applyPoseToPoint(model_pose, t_point_world);

	// just point towards camera for now...doesn't matter for depth generation
	float4 t_normal = (float4)(0,0,-1,0);

	// do a "zbuffer" test
	float4 previous_point = rendered_points[image_index];
	if (!isfinite(previous_point.s2) || t_point_camera.s2 < previous_point.s2) {
		rendered_mask[image_index] = mask_value; // mask_value
		rendered_points[image_index] = t_point_camera;
		rendered_normals[image_index] = t_normal;
	}
}

float2 intersectCylinder(const float4 ray_origin, const float4 ray_direction, const float radius)
{
	float x_d = ray_direction.s0;
	float y_d = ray_direction.s1;
	float x_e = ray_origin.s0;
	float y_e = ray_origin.s1;
	float a = x_d * x_d + y_d * y_d;
	float b = 2 * x_e * x_d + 2 * y_e * y_d;
	float c = x_e * x_e + y_e * y_e - radius * radius;

	float discrim = b * b - 4 * a * c;
	float sqrt_discrim = sqrt(discrim);
	float2 result_t;
	result_t.s0 = (-b - sqrt_discrim) / (2 * a);
	result_t.s1 = (-b + sqrt_discrim) / (2 * a);
	return result_t;
}

// shouldn't really need this:
float2 intersectCylinderXZ(const float4 ray_origin, const float4 ray_direction, const float radius)
{
	float x_d = ray_direction.s0;
	float z_d = ray_direction.s2;
	float x_e = ray_origin.s0;
	float z_e = ray_origin.s2;
	float a = x_d * x_d + z_d * z_d;
	float b = 2 * x_e * x_d + 2 * z_e * z_d;
	float c = x_e * x_e + z_e * z_e - radius * radius;

	float discrim = b * b - 4 * a * c;
	float sqrt_discrim = sqrt(discrim);
	float2 result_t;
	result_t.s0 = (-b - sqrt_discrim) / (2 * a);
	result_t.s1 = (-b + sqrt_discrim) / (2 * a);
	return result_t;
}

// get min valid intersection of intervals (t)
bool isValidInterval(const float2 interval)
{
	return (isfinite(interval.s0) && isfinite(interval.s1));
}

float2 intersectInterval(const float2 A, const float2 B)
{
	const float2 EMPTY = nan((uint2)0);
	if (!isValidInterval(A) || !isValidInterval(B)) return EMPTY;
	// check if no intersection
	if (A.s1 < B.s0 || B.s1 < A.s0) return EMPTY;
	// now some intersection, pick mins and maxes
	float2 result;
	result.s0 = max(A.s0, B.s0);
	result.s1 = min(A.s1, B.s1);
	return result;
}

#if 0
// A - B in set sense
float2 differenceInterval(const float2 A, const float2 B)
{

}
#endif

__kernel void KernelRaytraceSpecial(
    __global int *rendered_mask,
    __global float4 *rendered_points,
    __global float4 *rendered_normals,
    const float16 model_pose,
    const float16 model_pose_inverse,
    const float16 object_pose,
    const float16 object_pose_inverse,
    const float2 camera_focal,
    const float2 camera_center,
    const int2 image_dims,
    const float min_render_depth,
    const float max_render_depth,
    const int mask_value
    )
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    int2 pixel = (int2)(i,j);
    int image_index = getImageIndex(image_dims, pixel);

    // this is sort of borrowed from other renders...
    // should probably get the ray first, then start messing with poses...
    float4 camera_cop = applyPoseToPoint(model_pose_inverse, (float4)(0,0,0,1));
    float4 pixel_on_focal = pixelOnFocal(camera_focal, camera_center, pixel);
    float4 pixel_transformed = applyPoseToPoint(model_pose_inverse, pixel_on_focal);
    float4 ray_unit_v = (pixel_transformed - camera_cop);
    ray_unit_v.s3 = 0;
    ray_unit_v = normalize(ray_unit_v);

    // put points from WORLD into object frame
    float4 camera_cop_object = applyPoseToPoint(object_pose_inverse, camera_cop);
    float4 ray_unit_v_object = applyPoseToNormal(object_pose_inverse, ray_unit_v);

    // now do stuff...
	const float outer_radius = 0.1;
	const float thickness = 0.01;
	const float height = 0.2;

	const float2 min_max_render = (float2)(min_render_depth, max_render_depth);

	// outer cylinder
	float2 outer_cylinder_interval = intersectCylinderXZ(camera_cop_object, ray_unit_v_object, outer_radius);

	float2 intersect_with_render = intersectInterval(min_max_render, outer_cylinder_interval);
	float t = intersect_with_render.s0;

	// todo:
	// intersect using intervals with other stuff...
	


	if (isfinite(t)) {
		float4 t_point_object = camera_cop_object + t * ray_unit_v_object;
		float4 t_point_world = applyPoseToPoint(object_pose, t_point_object);
		float4 t_point_camera = applyPoseToPoint(model_pose, t_point_world);

		// just point towards camera for now...doesn't matter for depth generation
		float4 t_normal = (float4)(0,0,-1,0);

		float4 previous_point = rendered_points[image_index];
		if (!isfinite(previous_point.s2) || t_point_camera.s2 < previous_point.s2) {
			rendered_mask[image_index] = mask_value;
			rendered_points[image_index] = t_point_camera;
			rendered_normals[image_index] = t_normal;
		}
	}
}



// mark as true those points which violate empty space
// leave result AS IS for those which don't
// could also obviously mark false...
__kernel void KernelMarkPointsViolateEmpty(
	__global float * volume,
	__global float * weights,
	__global float4 * points,
	__global uchar * result_violates_empty,
	const float voxel_size,
	const int4 volume_dims,
	const float16 volume_pose,
    const float16 volume_pose_inverse,
    const float min_value_invalid
	)
{
	size_t i = get_global_id(0);
	float4 voxel_sizes = (float4)(voxel_size);

	float4 point = points[i];

	float4 point_in_volume = applyPoseToPoint(volume_pose_inverse, point);
	float4 voxel_f = point_in_volume / (float4)(voxel_size);

#if 0
	int4 voxel = convert_int4_rte(voxel_f);
	if (!checkVoxelWithinVolume(volume_dims, voxel)) {
		// nothing to say about this point
		return;
	}
	size_t v_index = getVIndexInt4(volume_dims, voxel);

	// can now look up volume value (or weight if you like)
	float v = volume[v_index];
#endif
	// instead check all surrounding voxels?
	int4 voxel_floor = convert_int4(voxel_f);
	if (!checkSurroundingVoxelsAreWithinVolume(volume_dims, voxel_floor)) {
		return;
	}
	float min_value_around = getMinValueSurroundingVoxels(volume_dims, voxel_floor, volume);

	if (min_value_around >= min_value_invalid) {
		result_violates_empty[i] = true;
	}
	// could mark false as well...
}


__kernel void KernelDepthImageToPoints(
	__global float *depth_image,
	__global float4 *points,
	const int2 image_dims,
	const float2 camera_f,
	const float2 camera_c
	)
{
	int2 pixel;
	pixel.s0 = get_global_id(0);
	pixel.s1 = get_global_id(1);

	int image_index = getImageIndex(image_dims, pixel);
	float depth = depth_image[image_index];

	float4 result = depthToPoint(pixel, depth, camera_c, camera_f);
	points[image_index] = result;
}


__kernel void KernelTransformPoints(
	__global float4 *input_points,
	__global float4 *output_points,
	const int2 image_dims,
	const float16 pose
	)
{
	int2 pixel;
	pixel.s0 = get_global_id(0);
	pixel.s1 = get_global_id(1);

	int image_index = getImageIndex(image_dims, pixel);
	float4 point = input_points[image_index];
	float4 result = isfinite(point.s2) ? applyPoseToPoint(pose, point) : nan((uint4)0);
	output_points[image_index] = result;
}

// could probably be written better...
__kernel void KernelSetInvalidPointsTrue(
	__global float * depth_image,
	__global uchar * inside_image,
	const int2 image_dims
	)
{
	int2 pixel;
	pixel.s0 = get_global_id(0);
	pixel.s1 = get_global_id(1);
	int image_index = getImageIndex(image_dims, pixel);
	float d = depth_image[image_index];

	bool is_invalid = !(d > 0);

	if (is_invalid) {
		inside_image[image_index] = 0xFF;
	}
}



////////////////////////////////////////////////
// from optimize.cl
// try to make optimize kernels that you can work with and understand...

#define DEBUG_CODE_RENDER_INVALID 1;
#define DEBUG_CODE_PROJECT_OOB 2;
#define DEBUG_CODE_FRAME_INVALID 3;
#define DEBUG_CODE_OUTLIER_DISTANCE 4;
#define DEBUG_CODE_OUTLIER_ANGLE 5;
#define DEBUG_CODE_SUCCESS 6;

__kernel void KernelOptimizeErrorAndJacobianICP(
    __global float4 *render_points,
    __global float4 *render_normals,
    __global float4 *frame_points,
    __global float4 *frame_normals,
    __global float *result_weighted_error, // (0 1 2 3 ...)
    __global float *result_weighted_jacobian, // (a0 a1 .. a6, b0 b1... b6, ...)
    __global int *result_debug_code,
    __global float *result_debug_huber_weight,
    __global float *result_debug_distance_weight,
    const int2 image_dims_render,
    const float2 camera_focal_render,
    const float2 camera_center_render,
    const int2 image_dims_frame,
    const float2 camera_focal_frame,
    const float2 camera_center_frame,
    const float16 x_transform,
    const float outlier_max_distance, // really?
    const float outlier_min_normal_dot_product, // really?
    const float huber_param_icp
)
{
    // switch to image-based indexing
    int2 pixel;
    pixel.s0 = get_global_id(0);
    pixel.s1 = get_global_id(1);
    int image_index_render = getImageIndex(image_dims_render, pixel);

    //////// associate
    float4 render_point = render_points[image_index_render];
    float4 render_normal = render_normals[image_index_render];
    if (!isfinite(render_point.z) || !isfinite(render_normal.z)) {
        result_debug_code[image_index_render] = DEBUG_CODE_RENDER_INVALID;
        return;
    }
    float4 p = applyPoseToPoint(x_transform, render_point);
    float4 p_normal = applyPoseToNormal(x_transform, render_normal);
    float2 p_proj = projectPoint(camera_focal_frame, camera_center_frame, p);

    // check that this falls within the target frame image
    int2 p_proj_floor = convert_int2_rtn(p_proj); // to negative infinity (always down) (floor)
    if (any(p_proj_floor < (int2)(0)) || any(p_proj_floor + (int2)(1) >= image_dims_frame)) {
        result_debug_code[image_index_render] = DEBUG_CODE_PROJECT_OOB;
        return;
    }

    // for depth association, get the nearest pixel (known to be valid now)
    int2 p_proj_int = convert_int2_rte(p_proj);
    int image_index_frame = getImageIndex(image_dims_frame, p_proj_int);

    float4 frame_point = frame_points[image_index_frame];
    float4 frame_normal = frame_normals[image_index_frame];
    if (!isfinite(frame_point.z) || !isfinite(frame_normal.z)) {
        result_debug_code[image_index_render] = DEBUG_CODE_FRAME_INVALID;
        return;
    }

    // now have associated frame point and normal (as well as float p_proj for image stuff)


    #define ICP_OUTLIER_BY_DISTANCE false // was false
    #define ICP_OUTLIER_BY_DISTANCE_SIGMAS 12 // was 6 (only relevant if OUTLIER_BY_DISTANCE)
    #define ICP_WEIGHT_BY_DISTANCE true

    float max_distance_to_use = ICP_OUTLIER_BY_DISTANCE ? ICP_OUTLIER_BY_DISTANCE_SIGMAS * simpleAxial(frame_point.s2) : outlier_max_distance;

    float4 icp_vector = frame_point - p;
    if (length(icp_vector) > max_distance_to_use) {
        result_debug_code[image_index_render] = DEBUG_CODE_OUTLIER_DISTANCE;
        return;
    }

    float dot_prod = dot(frame_normal, p_normal);
    if (dot_prod < outlier_min_normal_dot_product) {
        result_debug_code[image_index_render] = DEBUG_CODE_OUTLIER_ANGLE;
        return;
    }

    // note that, as before, ICP error is being computed against the plane of the RENDERED points, even though those are the moving ones
    float icp_error = dot(icp_vector, p_normal);
    float squared_icp_error = icp_error * icp_error;

    float sqrt_distance_weight_icp = 1.f;
    if (ICP_WEIGHT_BY_DISTANCE) {
        // DOUBLE CHECK THIS...I THINK FOR THE NORMAL EQUATIONS, we want sqrt(weight)
        // added another layer of sqrt distance weighting...just recently
        sqrt_distance_weight_icp = sqrt(simpleAxial(Z_MIN) / simpleAxial(frame_point.s2));
    }
    result_debug_distance_weight[image_index_render] = sqrt_distance_weight_icp;

    float sqrt_huber_weight_icp = 1.f;
    if (huber_param_icp > 0) {
        float huber_weight_icp = 1.f;
        if (squared_icp_error > huber_param_icp * huber_param_icp) {
            huber_weight_icp = huber_param_icp / fabs(icp_error); // yes, this is right because we want k / sqrt(loss)
        }

        // because we apply this to both the error and the jacobian row, need to sqrt so it comes out right for the normal eq
        sqrt_huber_weight_icp = sqrt(huber_weight_icp); // really?
    }
    result_debug_huber_weight[image_index_render] = sqrt_huber_weight_icp;

    float weight = sqrt_distance_weight_icp * sqrt_huber_weight_icp;

    float weighted_error = weight * icp_error;

    // need these for gradient:
    float x_2 = 2*p.x;
    float y_2 = 2*p.y;
    float z_2 = 2*p.z;
    // 3x6
    float j_3d_x[18] = {1,0,0, 0   , z_2,-y_2,
                        0,1,0, -z_2, 0  , x_2,
                        0,0,1,  y_2,-x_2, 0};

    float j_rot_x[18] = {0,0,0,  0  , z_2,-y_2,
                         0,0,0, -z_2, 0  , x_2,
                         0,0,0,  y_2,-x_2, 0};

    // Eigen::MatrixXf jacobian_row_icp_unweighted = - ( n_3d.transpose() * j_3d_x ) + ( (p_frame - p_3d).transpose() * j_rot_x );
    float jacobian_row[6];
    for (size_t col = 0; col < 6; ++col) {
        float to_store = - ( p_normal.s0 * j_3d_x[col+0] + p_normal.s1 * j_3d_x[col+6] + p_normal.s2 * j_3d_x[col+12] );
        to_store += icp_vector.s0 * j_rot_x[col+0] + icp_vector.s1 * j_rot_x[col+6] + icp_vector.s2 * j_rot_x[col+12];

        // weight here?
        to_store *= weight;

        jacobian_row[col] = to_store;
    }

    // for simplicity, just store the weighted error and weighted jacobian row...
    // in image style indexing
    // note that faster might be col * image_size + image_index?
    result_weighted_error[image_index_render] = weighted_error;
    for (int col = 0; col < 6; ++col) {
        result_weighted_jacobian[6 * image_index_render + col] = jacobian_row[col];
    }

    result_debug_code[image_index_render] = DEBUG_CODE_SUCCESS;
}

// based on optimize.cl, but simplified because no channels needed
float interpolateImage(__global float *image, int2 image_dims, float2 p_proj)
{
    int2 p_proj_image_floor = convert_int2(p_proj);
    int i = p_proj_image_floor.s0;
    int j = p_proj_image_floor.s1;

    float i00 = image[getImageIndex(image_dims, (int2)(i, j))];
    float i01 = image[getImageIndex(image_dims, (int2)(i, j+1))];
    float i10 = image[getImageIndex(image_dims, (int2)(i+1, j))];
    float i11 = image[getImageIndex(image_dims, (int2)(i+1, j+1))];
    float2 offset = p_proj - convert_float2(p_proj_image_floor);
    float off_x = offset.s0;
    float off_y = offset.s1;
    float result =		  i00 * (1 - off_x) * (1 - off_y)
                        + i01 * (1 - off_x) * (off_y)
                        + i10 * (off_x) * (1 - off_y)
                        + i11 * (off_x) * (off_y);
    return result;
}


__kernel void KernelOptimizeErrorAndJacobianImage(
    __global float4 *render_points,
    __global float4 *render_normals,
    __global float *render_image,
    __global float4 *frame_points,
    __global float4 *frame_normals,
    __global float *frame_image,
    __global float *frame_image_gradient_x,
    __global float *frame_image_gradient_y,
    __global float *result_weighted_error, // (0 1 2 3 ...)
    __global float *result_weighted_jacobian, // (a0 a1 .. a6, b0 b1... b6, ...)
    __global int *result_debug_code,
    __global float *result_debug_huber_weight,
    __global float *result_debug_distance_weight,
    const int2 image_dims_render,
    const float2 camera_focal_render,
    const float2 camera_center_render,
    const int2 image_dims_frame,
    const float2 camera_focal_frame,
    const float2 camera_center_frame,
    const float16 x_transform,
    const float outlier_max_distance, // really?
    const float outlier_min_normal_dot_product, // really?
    const float huber_param_icp,
    const float huber_param_image
)
{
    // switch to image-based indexing
    int2 pixel;
    pixel.s0 = get_global_id(0);
    pixel.s1 = get_global_id(1);
    int image_index_render = getImageIndex(image_dims_render, pixel);

    //////// associate
    float4 render_point = render_points[image_index_render];
    float4 render_normal = render_normals[image_index_render];
    if (!isfinite(render_point.z) || !isfinite(render_normal.z)) {
        result_debug_code[image_index_render] = DEBUG_CODE_RENDER_INVALID;
        return;
    }
    float4 p = applyPoseToPoint(x_transform, render_point);
    float4 p_normal = applyPoseToNormal(x_transform, render_normal);
    float2 p_proj = projectPoint(camera_focal_frame, camera_center_frame, p);

    // check that this falls within the target frame image
    int2 p_proj_floor = convert_int2_rtn(p_proj); // to negative infinity (always down) (floor)
    if (any(p_proj_floor < (int2)(0)) || any(p_proj_floor + (int2)(1) >= image_dims_frame)) {
        result_debug_code[image_index_render] = DEBUG_CODE_PROJECT_OOB;
        return;
    }

    // for depth association, get the nearest pixel (known to be valid now)
    int2 p_proj_int = convert_int2_rte(p_proj);
    int image_index_frame = getImageIndex(image_dims_frame, p_proj_int);

    float4 frame_point = frame_points[image_index_frame];
    float4 frame_normal = frame_normals[image_index_frame];
    if (!isfinite(frame_point.z) || !isfinite(frame_normal.z)) {
        result_debug_code[image_index_render] = DEBUG_CODE_FRAME_INVALID;
        return;
    }

    // now have associated frame point and normal (as well as float p_proj for image stuff)


    #define ICP_OUTLIER_BY_DISTANCE false // was false
    #define ICP_OUTLIER_BY_DISTANCE_SIGMAS 12 // was 6 (only relevant if OUTLIER_BY_DISTANCE)
    #define ICP_WEIGHT_BY_DISTANCE true

    float max_distance_to_use = ICP_OUTLIER_BY_DISTANCE ? ICP_OUTLIER_BY_DISTANCE_SIGMAS * simpleAxial(frame_point.s2) : outlier_max_distance;

    float4 icp_vector = frame_point - p;
    if (length(icp_vector) > max_distance_to_use) {
        result_debug_code[image_index_render] = DEBUG_CODE_OUTLIER_DISTANCE;
        return;
    }

    float dot_prod = dot(frame_normal, p_normal);
    if (dot_prod < outlier_min_normal_dot_product) {
        result_debug_code[image_index_render] = DEBUG_CODE_OUTLIER_ANGLE;
        return;
    }

    // note that, as before, ICP error is being computed against the plane of the RENDERED points, even though those are the moving ones
    float icp_error = dot(icp_vector, p_normal);
    float squared_icp_error = icp_error * icp_error;

    float sqrt_distance_weight_icp = 1.f;
    if (ICP_WEIGHT_BY_DISTANCE) {
        // DOUBLE CHECK THIS...I THINK FOR THE NORMAL EQUATIONS, we want sqrt(weight)
        // added another layer of sqrt distance weighting...just recently
        sqrt_distance_weight_icp = sqrt(simpleAxial(Z_MIN) / simpleAxial(frame_point.s2));
    }
    result_debug_distance_weight[image_index_render] = sqrt_distance_weight_icp;

    float sqrt_huber_weight_icp = 1.f;
    if (huber_param_icp > 0) {
        float huber_weight_icp = 1.f;
        if (squared_icp_error > huber_param_icp * huber_param_icp) {
            huber_weight_icp = huber_param_icp / fabs(icp_error); // yes, this is right because we want k / sqrt(loss)
        }

        // because we apply this to both the error and the jacobian row, need to sqrt so it comes out right for the normal eq
        sqrt_huber_weight_icp = sqrt(huber_weight_icp); // really?
    }
    // sigh...do I want to see this when doing color?
    //result_debug_huber_weight[image_index_render] = sqrt_huber_weight_icp;


    //////////////////
    // only this stuff is different for image error and gradient
    // you probably want the distance weight and huber weight for the image as well
    float rendered_image_value = render_image[image_index_render];
    float frame_image_value = interpolateImage(frame_image, image_dims_frame, p_proj);
    float image_error = (frame_image_value - rendered_image_value);

    float sqrt_huber_weight_image = 1.f;
    if (huber_param_image > 0) {
        float huber_weight_image = 1.f;
        float squared_image_error = image_error * image_error;
        if (squared_image_error > huber_param_image * huber_param_image) {
            huber_weight_image = huber_param_image / fabs(image_error);
        }
            // because we apply this to both the error and the jacobian row, need to sqrt so it comes out right for the normal eq
        sqrt_huber_weight_image = sqrt(huber_weight_image);
    }
    result_debug_huber_weight[image_index_render] = sqrt_huber_weight_image; // also include huber_weight_icp???

    float weight = sqrt_distance_weight_icp * sqrt_huber_weight_icp * sqrt_huber_weight_image; // distance for color?

    float weighted_error = weight * image_error;

    // need these for gradient:
    float x_2 = 2*p.x;
    float y_2 = 2*p.y;
    float z_2 = 2*p.z;
    // 3x6
    float j_3d_x[18] = {1,0,0, 0   , z_2,-y_2,
                        0,1,0, -z_2, 0  , x_2,
                        0,0,1,  y_2,-x_2, 0};

    const float inverse_depth = 1.0f / p.z;
    // 2x3
    float4 j_proj_3d_r0 = (float4) ( (camera_focal_frame.s0 * inverse_depth), (0), (-p.x * camera_focal_frame.s0 * inverse_depth * inverse_depth), 0 );
    float4 j_proj_3d_r1 = (float4) ( (0), (camera_focal_frame.s1 * inverse_depth), (-p.y * camera_focal_frame.s1 * inverse_depth * inverse_depth), 0 );

    // 1x2
    float2 j_error_proj =
            (float2) ( interpolateImage(frame_image_gradient_x, image_dims_frame, p_proj),
                       interpolateImage(frame_image_gradient_y, image_dims_frame, p_proj) );

    // Eigen::MatrixXf jacobian_row_color = weight_color * j_error_proj * j_proj_3d * j_3d_x;
    // 1x3 (j_error_proj * j_proj_3d)
    float4 j_error_3d = (float4) ( (j_error_proj.s0 * j_proj_3d_r0.s0 + j_error_proj.s1 * j_proj_3d_r1.s0),
                                   (j_error_proj.s0 * j_proj_3d_r0.s1 + j_error_proj.s1 * j_proj_3d_r1.s1),
                                   (j_error_proj.s0 * j_proj_3d_r0.s2 + j_error_proj.s1 * j_proj_3d_r1.s2),
                                   0 );

    float jacobian_row[6];
    for (size_t col = 0; col < 6; col++) {
        float to_store = j_error_3d.s0 * j_3d_x[col+0] + j_error_3d.s1 * j_3d_x[col+6] + j_error_3d.s2 * j_3d_x[col+12];
        to_store *= weight;
        jacobian_row[col] = to_store;
    }

    // for simplicity, just store the weighted error and weighted jacobian row...
    // in image style indexing
    // note that faster might be col * image_size + image_index?
    result_weighted_error[image_index_render] = weighted_error;
    for (int col = 0; col < 6; ++col) {
        result_weighted_jacobian[6 * image_index_render + col] = jacobian_row[col];
    }

    result_debug_code[image_index_render] = DEBUG_CODE_SUCCESS;
}



__kernel void KernelOptimizeNormalEquationTerms(
    __global float *weighted_error, // (0 1 2 3 ...)
    __global float *weighted_jacobian, // (a0 a1 .. a6, b0 b1... b6, ...)
	__global float *LHS_terms, // I think it's all 0 terms, then all 1 terms, etc... (21)
	__global float *RHS_terms, // I think it's all 0 terms, then all 1 terms, etc... (6)
	const int total_count,
	const int offset,
	const float weight
	)
{
	// do stuff
	const int index = get_global_id(0);

	int LHS_counter = 0; // index to pack in the upper triangle...must end at 21
    for (int row = 0; row < 6; ++row) {
        for (int col = row; col < 6; ++col) {
			float this_term = weighted_jacobian[6 * index + row] * weighted_jacobian[6 * index + col];
			LHS_terms[LHS_counter * total_count + offset + index] = this_term * weight;
            LHS_counter++;
        }
    }

	for (int col = 0; col < 6; ++col) {
		float this_term = weighted_jacobian[6 * index + col] * weighted_error[index];
		RHS_terms[col * total_count + offset + index] = this_term * weight;
	}
}


__kernel void KernelVignetteApplyModelPolynomial3Uchar4(
    __global uchar4 * input_image,
    __global uchar4 * output_image,
    const int2 image_dims,
    const float2 camera_center,
    const float4 vignette_model)
{
    int2 pixel;
    pixel.s0 = get_global_id(0);
    pixel.s1 = get_global_id(1);
    int image_index = getImageIndex(image_dims, pixel);

    float radius_scale = length((float2)(0,0) - camera_center);
    float radius = length(convert_float2(pixel) - camera_center) / radius_scale;
    float r_2 = radius * radius;
    float r_4 = r_2 * r_2;
    float r_6 = r_2 * r_2 * r_2;
    float vignette_factor = 1 + vignette_model.s0 * r_2 + vignette_model.s1 * r_4 + vignette_model.s2 * r_6;

    // depends on type:
    uchar4 input_value = input_image[image_index];
    uchar4 output_value = convert_uchar4_sat_rte(convert_float4(input_value) / vignette_factor);
    output_image[image_index] = output_value;
}

__kernel void KernelVignetteApplyModelPolynomial3Float(
    __global float * input_image,
    __global float * output_image,
    const int2 image_dims,
    const float2 camera_center,
    const float4 vignette_model)
{
    int2 pixel;
    pixel.s0 = get_global_id(0);
    pixel.s1 = get_global_id(1);
    int image_index = getImageIndex(image_dims, pixel);

    float radius_scale = length((float2)(0,0) - camera_center);
    float radius = length(convert_float2(pixel) - camera_center) / radius_scale;
    float r_2 = radius * radius;
    float r_4 = r_2 * r_2;
    float r_6 = r_2 * r_2 * r_2;
    float vignette_factor = 1 + vignette_model.s0 * r_2 + vignette_model.s1 * r_4 + vignette_model.s2 * r_6;

    // depends on type:
    float input_value = input_image[image_index];
    float output_value = input_value / vignette_factor;
    output_image[image_index] = output_value;
}


#if 0

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
#endif

