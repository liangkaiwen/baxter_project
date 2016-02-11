#include "stdafx.h"

#include "OpenCLTSDF.h"
#include "RunLengthEncoding.hpp"
#include "MarchingCubes.h"
#include "MeshUtilities.h"

#include "util.h"

#include <boost/assign.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/timer.hpp>
#include <boost/foreach.hpp>

#include "KernelRenderPointsAndNormals.h"
#include "KernelRenderPoints.h"
#include "KernelRenderNormalForPoints.h"
#include "KernelRenderColorForPoints.h"
#include "KernelExtractVolumeSlice.h"
#include "KernelSetFloat.h"

#undef min
#undef max

using std::cout;
using std::endl;

uint64_t OpenCLTSDF::deallocation_counter = 0;
uint64_t OpenCLTSDF::reallocation_counter = 0;
uint64_t OpenCLTSDF::save_to_disk_counter = 0;
uint64_t OpenCLTSDF::load_from_disk_counter = 0;



OpenCLTSDF::OpenCLTSDF(boost::shared_ptr<OpenCLAllKernels> all_kernels,
	float camera_focal_x, float camera_focal_y,
	float camera_center_x, float camera_center_y,
	int camera_size_x, int camera_size_y,
	int volume_cell_count_x, int volume_cell_count_y, int volume_cell_count_z, float volume_cell_size,
	float volume_max_weight_icp, float volume_max_weight_color, bool use_most_recent_color, float min_truncation_distance,
    fs::path temp_base_path)
	: cl(all_kernels->getCL()),
	all_kernels_(all_kernels),
	add_frame_kernel(all_kernels->getKernel("addFrame")),
	add_volume_kernel(all_kernels->getKernel("addVolume")),
	set_volume_to_sphere_kernel(all_kernels->getKernel("setVolumeToSphere")),
	set_max_weight_in_volume_kernel(all_kernels->getKernel("setMaxWeightInVolume")),
	set_value_in_sphere_kernel(all_kernels->getKernel("setValueInSphere")),
	set_value_in_box_kernel(all_kernels->getKernel("setValueInBox")),
	set_points_inside_box_true_kernel(all_kernels->getKernel("setPointsInsideBoxTrue")),
	does_box_contain_surface_kernel(all_kernels->getKernel("doesBoxContainSurface")),
	camera_focal(camera_focal_x, camera_focal_y),
	camera_center(camera_center_x, camera_center_y),
	camera_size(camera_size_x, camera_size_y),
	volume_cell_counts(volume_cell_count_x, volume_cell_count_y, volume_cell_count_z),
	volume_cell_size(volume_cell_size),
	volume_max_weight_icp(volume_max_weight_icp),
	volume_max_weight_color(volume_max_weight_color),
	use_most_recent_color(use_most_recent_color),
	min_truncation_distance(min_truncation_distance),
	min_expand_locked(0,0,0),
	max_expand_locked(0,0,0),
	temp_base_path(temp_base_path),
	mesh_cache_is_valid(false),
	volume_buffer_d_(all_kernels),
	volume_buffer_dw_(all_kernels),
	volume_buffer_c_(all_kernels),
	volume_buffer_cw_(all_kernels),
    tsdf_state(TSDF_STATE_EMPTY)
{
	constructorBody();
}

OpenCLTSDF::OpenCLTSDF(OpenCLTSDF const& other, int volume_cell_count_x, int volume_cell_count_y, int volume_cell_count_z)
	: cl(other.cl),
	all_kernels_(other.all_kernels_),
	add_frame_kernel(other.add_frame_kernel),
	add_volume_kernel(other.add_volume_kernel),
	set_volume_to_sphere_kernel(other.set_volume_to_sphere_kernel),
	set_max_weight_in_volume_kernel(other.set_max_weight_in_volume_kernel),
	set_value_in_sphere_kernel(other.set_value_in_sphere_kernel),
	set_value_in_box_kernel(other.set_value_in_box_kernel),
	set_points_inside_box_true_kernel(other.set_points_inside_box_true_kernel),
	does_box_contain_surface_kernel(other.does_box_contain_surface_kernel),
	camera_focal(other.camera_focal),
	camera_center(other.camera_center),
	camera_size(other.camera_size),
	volume_cell_counts(volume_cell_count_x, volume_cell_count_y, volume_cell_count_z),
	volume_cell_size(other.volume_cell_size),
	volume_max_weight_icp(other.volume_max_weight_icp),
	volume_max_weight_color(other.volume_max_weight_color),
	use_most_recent_color(other.use_most_recent_color),
	min_truncation_distance(other.min_truncation_distance),
	min_expand_locked(0,0,0),
	max_expand_locked(0,0,0),
	temp_base_path(other.temp_base_path),
	mesh_cache_is_valid(false),
	volume_buffer_d_(other.volume_buffer_d_),
	volume_buffer_dw_(other.volume_buffer_dw_),
	volume_buffer_c_(other.volume_buffer_c_),
	volume_buffer_cw_(other.volume_buffer_cw_),
    tsdf_state(TSDF_STATE_EMPTY)
{
	constructorBody();
}

void OpenCLTSDF::generateTempPath()
{
	temp_generated = fs::unique_path(temp_base_path / "%%%%-%%%%-%%%%-%%%%");
}

void OpenCLTSDF::constructorBody()
{
	generateTempPath();
    // assuming no resize, we can generate center and sphere once
	generateCenterAndRadius(sphere_center, sphere_radius);
}

// new not broken version? (well, currently broken by choice)
OpenCLTSDF* OpenCLTSDF::clone()
{
    throw std::runtime_error("you shouldn't call this yet");
    setTSDFState(TSDF_STATE_RAM);
	OpenCLTSDF* result = new OpenCLTSDF(*this);
	result->generateTempPath();
	result->makeBuffersNull(); // added this so that result and this don't share the same (empty) buffers (probably not necessary?)
	return result;
}

void OpenCLTSDF::allocateEmptyVolumeBuffers()
{
	clearMeshCache();

	volume_buffer_d_.resize(volume_cell_counts, sizeof(float));
	volume_buffer_dw_.resize(volume_cell_counts, sizeof(float));
	volume_buffer_c_.resize(volume_cell_counts, 4 * sizeof(unsigned char));
	volume_buffer_cw_.resize(volume_cell_counts, sizeof(float));

	volume_buffer_d_.setFloat(getEmptyDValue());
	volume_buffer_dw_.setFloat(0);
	volume_buffer_c_.setUChar(0);
	volume_buffer_cw_.setFloat(0);

    tsdf_state = TSDF_STATE_GPU;
}

void OpenCLTSDF::makeEmptyInPlace()
{
	clearMeshCache();

    // could make this faster...idea for this function is that TSDF_STATE_GPU already
    setTSDFState(TSDF_STATE_GPU);

	volume_buffer_d_.setFloat(getEmptyDValue());
	volume_buffer_dw_.setFloat(0);
	volume_buffer_c_.setUChar(0);
	volume_buffer_cw_.setFloat(0);
}

void OpenCLTSDF::allocateFloatBufferFast(size_t number_of_floats, float float_value, cl::Buffer & result_buffer, size_t & result_buffer_size_bytes)
{
	cl_int error_code; // look at this?
	result_buffer_size_bytes = number_of_floats * sizeof(float);
	result_buffer = cl::Buffer(cl.context, CL_MEM_READ_WRITE, result_buffer_size_bytes, NULL, &error_code);

	KernelSetFloat _KernelSetFloat(*all_kernels_);
	_KernelSetFloat.runKernel(result_buffer, number_of_floats, float_value);
}


void OpenCLTSDF::allocateVolumeBuffersWithValues(std::vector<float> const& d, std::vector<float> const& dw, std::vector<unsigned char> const& c, std::vector<float> const& cw)
{
	clearMeshCache();

	// check input
	size_t cells = getCellCount();
	if (d.size() != cells) throw std::runtime_error ( (boost::format("d.size() = %d but cells = %d") % d.size() % cells).str() );
	if (dw.size() != cells) throw std::runtime_error ("dw.size() != cells");
	if (c.size() != cells * 4) throw std::runtime_error ("c.size() != cells * 4");
	if (cw.size() != cells) throw std::runtime_error ("cw.size() != cells");

	volume_buffer_d_.resize(volume_cell_counts, sizeof(float));
	volume_buffer_dw_.resize(volume_cell_counts, sizeof(float));
	volume_buffer_c_.resize(volume_cell_counts, 4 * sizeof(unsigned char));
	volume_buffer_cw_.resize(volume_cell_counts, sizeof(float));

	volume_buffer_d_.getBufferWrapper().writeFromFloatVector(d);
	volume_buffer_dw_.getBufferWrapper().writeFromFloatVector(dw);
	volume_buffer_c_.getBufferWrapper().writeFromByteVector(c);
	volume_buffer_cw_.getBufferWrapper().writeFromFloatVector(cw);

    tsdf_state = TSDF_STATE_GPU;
}

cl::Buffer OpenCLTSDF::allocateBufferCheckError(size_t size, void* data)
{
	cl_int error_code;
	cl::Buffer result = cl::Buffer(cl.context, CL_MEM_COPY_HOST_PTR, size, data, &error_code);
	if (error_code != CL_SUCCESS) {
		std::string error_string = std::string("OpenCL Error: ") + oclErrorString(error_code);
		cout << error_string << endl;
		cout << "EXIT(1)" << endl;
		exit(1);
	}
	return result;
}

TSDFState OpenCLTSDF::getTSDFState() const
{
    return tsdf_state;
}

void OpenCLTSDF::setTSDFState(TSDFState new_state)
{
    if (tsdf_state == new_state) return;

    if (new_state == TSDF_STATE_EMPTY) {
        changeStateEmpty();
    }
    else if (new_state == TSDF_STATE_GPU) {
        if (tsdf_state == TSDF_STATE_EMPTY) {
            allocateEmptyVolumeBuffers();
        }
        else if (tsdf_state == TSDF_STATE_DISK) {
            changeStateDiskToRAM();
            changeStateRAMToGPU();
        }
        else if (tsdf_state == TSDF_STATE_RAM) {
            changeStateRAMToGPU();
        }
        else {
            throw std::runtime_error("nope");
        }
    }
    else if (new_state == TSDF_STATE_RAM) {
        if (tsdf_state == TSDF_STATE_EMPTY) {
            cout << "Unexpected inefficent TSDF_STATE_EMPTY to TSDF_STATE_RAM" << endl;
            allocateEmptyVolumeBuffers();
            changeStateGPUToRAM();
        }
        else if (tsdf_state == TSDF_STATE_DISK) {
            changeStateDiskToRAM();
        }
        else if (tsdf_state == TSDF_STATE_GPU) {
            changeStateGPUToRAM();
        }
        else {
            throw std::runtime_error("nope");
        }
    }
    else if (new_state == TSDF_STATE_DISK) {
        if (tsdf_state == TSDF_STATE_EMPTY) {
            cout << "Unexpected inefficent TSDF_STATE_EMPTY to TSDF_STATE_DISK" << endl;
            allocateEmptyVolumeBuffers();
            changeStateGPUToRAM();
            changeStateRAMToDisk();
        }
        else if (tsdf_state == TSDF_STATE_GPU) {
            changeStateGPUToRAM();
            changeStateRAMToDisk();
        }
        else if (tsdf_state == TSDF_STATE_RAM) {
            changeStateRAMToDisk();
        }
        else {
            throw std::runtime_error("nope");
        }
    }
    else {
        throw std::runtime_error("nope");
    }
}

void OpenCLTSDF::ensureAllocated()
{
    setTSDFState(TSDF_STATE_GPU);
}

void OpenCLTSDF::freeRLEMemory()
{
	std::vector<std::pair<size_t, float> >().swap(vector_d_rle);
	std::vector<std::pair<size_t, float> >().swap(vector_dw_rle);
	std::vector<std::pair<size_t, unsigned char> >().swap(vector_c_rle);
	std::vector<std::pair<size_t, float> >().swap(vector_cw_rle);
}

void OpenCLTSDF::clearMeshCache()
{
    mesh_cache_is_valid = false;
    MeshVertexVector().swap(mesh_cache_all_vertices);
	TriangleVector().swap(mesh_cache_all_triangles);
	std::vector<bool>().swap(mesh_cache_vertex_validity);
}

size_t OpenCLTSDF::getBytesMeshCache() const
{
    size_t result = 0;
    result += mesh_cache_all_vertices.capacity() * sizeof(MeshVertex);
    result += mesh_cache_all_triangles.capacity() * sizeof(Triangle);
    result += mesh_cache_vertex_validity.capacity() * sizeof(bool);
    return result;
}


void OpenCLTSDF::save(fs::path const& folder)
{
	if (!fs::exists(folder) && !fs::create_directories(folder)) {
		throw std::runtime_error("bad folder: " + folder.string());
	}

    // must make sure state is RAM first (from disk or GPU!)
    setTSDFState(TSDF_STATE_RAM);

	// write the four RLE buffers to disk
	fs::path filename = folder / "vector_d_rle.bin";
	EigenUtilities::writeVector(vector_d_rle, filename);
	filename = folder / "vector_dw_rle.bin";
	EigenUtilities::writeVector(vector_dw_rle, filename);
	filename = folder / "vector_c_rle.bin";
	EigenUtilities::writeVector(vector_c_rle, filename);
	filename = folder / "vector_cw_rle.bin";
	EigenUtilities::writeVector(vector_cw_rle, filename);

	// also "variable" params
	{
		fs::path filename = folder / "volume_cell_counts.txt";
		std::ofstream file (filename.string().c_str());
		file << volume_cell_counts[0] << endl;
		file << volume_cell_counts[1] << endl;
		file << volume_cell_counts[2] << endl;
	}
}

void OpenCLTSDF::load(fs::path const& folder)
{
	if (!fs::exists(folder)) {
		throw std::runtime_error("bad folder: " + folder.string());
	}

    setTSDFState(TSDF_STATE_EMPTY); // clears mesh cache

	// read into the four RLE buffers
	fs::path filename = folder / "vector_d_rle.bin";
	EigenUtilities::readVector(vector_d_rle, filename);
	filename = folder / "vector_dw_rle.bin";
	EigenUtilities::readVector(vector_dw_rle, filename);
	filename = folder / "vector_c_rle.bin";
	EigenUtilities::readVector(vector_c_rle, filename);
	filename = folder / "vector_cw_rle.bin";
	EigenUtilities::readVector(vector_cw_rle, filename);

	// also "variable" params
	{
		fs::path filename = folder / "volume_cell_counts.txt";
		std::ifstream file (filename.string().c_str());
		file >> volume_cell_counts[0];
		file >> volume_cell_counts[1];
		file >> volume_cell_counts[2];
	}

    tsdf_state = TSDF_STATE_RAM;
}

void OpenCLTSDF::loadLazy(fs::path const& folder)
{
	if (!fs::exists(folder)) {
		throw std::runtime_error("bad folder: " + folder.string());
	}

    // just keep a reference to the folder so that if needed later, it will be loaded from there
    load_lazy_path = folder;
    tsdf_state = TSDF_STATE_DISK;
}

void OpenCLTSDF::makeBuffersNull()
{
	volume_buffer_d_.deallocate();
	volume_buffer_dw_.deallocate();
	volume_buffer_c_.deallocate();
	volume_buffer_cw_.deallocate();
}

void OpenCLTSDF::changeStateGPUToRAM()
{
	if (tsdf_state != TSDF_STATE_GPU) {
        cout << "changeStateGPUToRAM" << endl;
        throw std::runtime_error("changeStateGPUToRAM");
    }

	std::vector<float> vector_d;
	std::vector<float> vector_dw;
	std::vector<unsigned char> vector_c;
	std::vector<float> vector_cw;

	getAllBuffers(vector_d, vector_dw, vector_c, vector_cw);
	makeBuffersNull();

	// run length encode 
	runLengthEncode(vector_d, vector_d_rle);
	runLengthEncode(vector_dw, vector_dw_rle);
	runLengthEncode(vector_c, vector_c_rle);
	runLengthEncode(vector_cw, vector_cw_rle);

    tsdf_state = TSDF_STATE_RAM;

	++deallocation_counter;
}

void OpenCLTSDF::changeStateRAMToGPU()
{
    if (tsdf_state != TSDF_STATE_RAM) {
        cout << "changeStateRAMToGPU" << endl;
        throw std::runtime_error("changeStateRAMToGPU");
    }

	std::vector<float> vector_d;
	std::vector<float> vector_dw;
	std::vector<unsigned char> vector_c;
	std::vector<float> vector_cw;

	runLengthDecode(vector_d_rle, vector_d);
	runLengthDecode(vector_dw_rle, vector_dw);
	runLengthDecode(vector_c_rle, vector_c);
	runLengthDecode(vector_cw_rle, vector_cw);
	freeRLEMemory();

	allocateVolumeBuffersWithValues(vector_d, vector_dw, vector_c, vector_cw);

    tsdf_state = TSDF_STATE_GPU;

	++reallocation_counter;
}

void OpenCLTSDF::changeStateRAMToDisk()
{
    if (tsdf_state != TSDF_STATE_RAM) {
        cout << "changeStateRAMToDisk" << endl;
        throw std::runtime_error("changeStateRAMToDisk");
    }

	save(temp_generated);

	freeRLEMemory();

    tsdf_state = TSDF_STATE_DISK;

    save_to_disk_counter++;
}

void OpenCLTSDF::changeStateDiskToRAM()
{
    if (tsdf_state != TSDF_STATE_DISK) {
        cout << "changeStateDiskToRAM" << endl;
        throw std::runtime_error("changeStateDiskToRAM");
    }

    if (!load_lazy_path.empty()) {
        load(load_lazy_path);
        load_lazy_path.clear();
    }
    else {
    	load(temp_generated); 
    }

    load_from_disk_counter++;
}

void OpenCLTSDF::changeStateEmpty()
{
    // regardless of current state, clear ram and gpu, and set state back to empty
    // note that this gets called by load
    freeRLEMemory();
    makeBuffersNull();
    clearMeshCache();

    tsdf_state = TSDF_STATE_EMPTY;
}


void OpenCLTSDF::addFrame(const Eigen::Affine3f& pose,
	ImageBuffer const& buffer_depth_image,
	ImageBuffer const& buffer_color_image,
	ImageBuffer const& buffer_segments,
	int which_segment)
{
	cout << "deprecated version of addFrame" << endl;

	ImageBuffer buffer_trivial_weights (cl);
	std::vector<float> trivial_weights (camera_size[0] * camera_size[1], 1.f);
	buffer_trivial_weights.resize(camera_size[1], camera_size[0], 1, CV_32F);
	buffer_trivial_weights.getBufferWrapper().writeFromFloatVector(trivial_weights);
	addFrame(pose, buffer_depth_image, buffer_color_image, buffer_segments, buffer_trivial_weights, buffer_trivial_weights, which_segment);
}

void OpenCLTSDF::addFrame(const Eigen::Affine3f& pose,
	ImageBuffer const& buffer_depth_image,
	ImageBuffer const& buffer_color_image,
	ImageBuffer const& buffer_segments,
	ImageBuffer const& buffer_depth_weights,
	ImageBuffer const& buffer_color_weights,
	int which_segment)
{
	ensureAllocated();
	clearMeshCache();


	// todo: separate kernel
	try {
		cl_float16 cl_pose = getCLPose(pose);

		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
		cl_int2 cl_camera_size = {camera_size[0], camera_size[1]};

		// not a problem on nvidia, but CPU complains
		cl::Buffer local_segment_buffer;
		if (which_segment == 0) {
			local_segment_buffer = cl::Buffer(cl.context, 0, 1);
		}
		else {
			local_segment_buffer = buffer_segments.getBuffer();
		}

		//////////////////////////
		int kernel_arg = 0;
		// buffers first
		add_frame_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, buffer_depth_image.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, buffer_color_image.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, local_segment_buffer);
		add_frame_kernel.setArg(kernel_arg++, buffer_depth_weights.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, buffer_color_weights.getBuffer());
		add_frame_kernel.setArg(kernel_arg++, (cl_int)which_segment);
		add_frame_kernel.setArg(kernel_arg++, cl_volume_dims);
		add_frame_kernel.setArg(kernel_arg++, (cl_float)volume_cell_size);
		add_frame_kernel.setArg(kernel_arg++, (cl_float)volume_max_weight_icp);
		add_frame_kernel.setArg(kernel_arg++, (cl_float)volume_max_weight_color);
		add_frame_kernel.setArg(kernel_arg++, cl_pose);
		add_frame_kernel.setArg(kernel_arg++, cl_camera_focal);
		add_frame_kernel.setArg(kernel_arg++, cl_camera_center);
		add_frame_kernel.setArg(kernel_arg++, cl_camera_size);
		add_frame_kernel.setArg(kernel_arg++, (cl_int)use_most_recent_color);
		add_frame_kernel.setArg(kernel_arg++, (cl_float)min_truncation_distance);
		////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(add_frame_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("add_frame_kernel\n");
		throw er;
	}

}

void OpenCLTSDF::renderPoints(
	const Eigen::Affine3f& model_pose,
	const Eigen::Array2f & camera_focal,
	const Eigen::Array2f & camera_center,
	const Eigen::Array2f & min_max_depth,
	bool replace_only_if_nearer,
	int mask_value,
	RenderBuffers& render_buffers)
{
	ensureAllocated();

	KernelRenderPoints _KernelRenderPoints(*all_kernels_);
	_KernelRenderPoints.runKernel(volume_buffer_d_, volume_buffer_dw_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(),
		volume_cell_size, model_pose, camera_focal, camera_center, min_max_depth[0], min_max_depth[1], mask_value);
}

void OpenCLTSDF::renderColors(
	const Eigen::Affine3f& model_pose,
	const Eigen::Array2f & camera_focal,
	const Eigen::Array2f & camera_center,
	const Eigen::Array2f & min_max_depth,
	bool replace_only_if_nearer,
	int mask_value,
	RenderBuffers& render_buffers)
{
	ensureAllocated();

	KernelRenderColorForPoints _KernelRenderColorForPoints(*all_kernels_);
	_KernelRenderColorForPoints.runKernel(volume_buffer_c_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(), render_buffers.getImageBufferColorImage(),
		volume_cell_size, model_pose, mask_value);
}


void OpenCLTSDF::renderFrame(
	const Eigen::Affine3f& model_pose,
	const Eigen::Array2f & camera_focal,
	const Eigen::Array2f & camera_center,
	const Eigen::Array2f & min_max_depth,
	bool replace_only_if_nearer,
	int mask_value,
	RenderBuffers& render_buffers)
{
	ensureAllocated();

	KernelRenderPointsAndNormals _KernelRenderPointsAndNormals(*all_kernels_);
	KernelRenderPoints _KernelRenderPoints(*all_kernels_);
	KernelRenderNormalForPoints _KernelRenderNormalForPoints(*all_kernels_);
	KernelRenderColorForPoints _KernelRenderColorForPoints(*all_kernels_);

#if 1
	// points and normals
	_KernelRenderPointsAndNormals.runKernel(volume_buffer_d_, volume_buffer_dw_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(), render_buffers.getImageBufferNormals(),
		volume_cell_size, model_pose, camera_focal, camera_center, min_max_depth[0], min_max_depth[1], mask_value);
#else
	// new separated:
	_KernelRenderPoints.runKernel(volume_buffer_d_, volume_buffer_dw_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(),
		volume_cell_size, model_pose, camera_focal, camera_center, min_max_depth[0], min_max_depth[1], mask_value);
	_KernelRenderNormalForPoints.runKernel(volume_buffer_d_, volume_buffer_dw_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(), render_buffers.getImageBufferNormals(),
		volume_cell_size, model_pose, mask_value);
#endif

	_KernelRenderColorForPoints.runKernel(volume_buffer_c_,
		render_buffers.getImageBufferMask(), render_buffers.getImageBufferPoints(), render_buffers.getImageBufferColorImage(),
		volume_cell_size, model_pose, mask_value);

	// original render code:
#if 0
	try {
		cl_float16 cl_pose = getCLPose(pose);
		cl_float16 cl_pose_inverse = getCLPose(pose.inverse());

		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};

		cl_float2 focal_lengths_render = {focal_lengths_render_x, focal_lengths_render_y};
		cl_float2 centers_render = {centers_render_x, centers_render_y};
		cl_int4 render_rect = {range_rect_x, range_rect_y, range_rect_width, range_rect_height};

		//////////////////////////
		int kernel_arg = 0;

		render_frame_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		render_frame_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		render_frame_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		render_frame_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		render_frame_kernel.setArg(kernel_arg++, render_buffers.getImageBufferMask().getBuffer());
		render_frame_kernel.setArg(kernel_arg++, render_buffers.getImageBufferColorImage().getBuffer());
		render_frame_kernel.setArg(kernel_arg++, render_buffers.getImageBufferPoints().getBuffer());
		render_frame_kernel.setArg(kernel_arg++, render_buffers.getImageBufferNormals().getBuffer());
		render_frame_kernel.setArg(kernel_arg++, cl_volume_dims);
		render_frame_kernel.setArg(kernel_arg++, (cl_float)volume_cell_size);
		render_frame_kernel.setArg(kernel_arg++, cl_pose);
		render_frame_kernel.setArg(kernel_arg++, cl_pose_inverse);
		render_frame_kernel.setArg(kernel_arg++, focal_lengths_render);
		render_frame_kernel.setArg(kernel_arg++, centers_render);
		render_frame_kernel.setArg(kernel_arg++, render_rect);
		render_frame_kernel.setArg(kernel_arg++, (cl_float)volume_cell_size /*step_size*/);
		render_frame_kernel.setArg(kernel_arg++, (cl_float)render_min_depth);
		render_frame_kernel.setArg(kernel_arg++, (cl_float)render_max_depth);
		render_frame_kernel.setArg(kernel_arg++, (cl_int) replace_only_if_nearer);
		render_frame_kernel.setArg(kernel_arg++, (cl_int) mask_value);
		//////////////////////////////////

		cl::NDRange global(render_rect.s[2], render_rect.s[3]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(render_frame_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("render_frame_kernel\n");
		throw er;
	}
#endif
}

void OpenCLTSDF::generateMesh(MeshVertexVector & vertices, TriangleVector & triangles)
{
	MeshVertexVector all_vertices;
	TriangleVector all_triangles;
	std::vector<bool> vertex_validity;
	generateMeshAndValidity(all_vertices, all_triangles, vertex_validity);
	MeshUtilities::extractValidVerticesAndTriangles(all_vertices, all_triangles, vertex_validity, vertices, triangles);
}


void OpenCLTSDF::generateMeshAndValidity(MeshVertexVector & all_vertices, TriangleVector & all_triangles, std::vector<bool> & vertex_validity)
{
	if (mesh_cache_is_valid) {
		all_vertices = mesh_cache_all_vertices;
		all_triangles = mesh_cache_all_triangles;
		vertex_validity = mesh_cache_vertex_validity;
		return;
	}

	std::vector<float> bufferD;
	std::vector<float> bufferDW;
	std::vector<unsigned char> bufferC;
	getBufferD(bufferD);
	getBufferDW(bufferDW);
	getBufferC(bufferC);

	MeshUtilities::generateMeshAndValidity(bufferD, bufferDW, bufferC, volume_cell_counts, volume_cell_size, all_vertices, all_triangles, vertex_validity);

	////////////////
	// copy into cache
	mesh_cache_all_vertices = all_vertices;
	mesh_cache_all_triangles = all_triangles;
	mesh_cache_vertex_validity = vertex_validity;
	mesh_cache_is_valid = true;
}


std::vector<Eigen::Vector3f> OpenCLTSDF::getVolumeCorners(const Eigen::Affine3f& pose) const
{
	return ::getVolumeCorners(volume_cell_counts, volume_cell_size, pose);
}

// perhaps a better version with explicit d_min and d_max
void OpenCLTSDF::getNonzeroFilteredVoxelCenters(const Eigen::Affine3f& pose, float d_min, float d_max, std::vector<std::pair<Eigen::Vector3f, float> >& result)
{
	ensureAllocated();

	std::vector<float> bufferDVector(getCellCount());
	std::vector<float> bufferDWVector(getCellCount());
	volume_buffer_d_.getBufferWrapper().readToFloatVector(bufferDVector);
	volume_buffer_dw_.getBufferWrapper().readToFloatVector(bufferDWVector);

	result.clear();
	for (int v_x = 0; v_x < volume_cell_counts[0]; ++v_x) {
		for (int v_y = 0; v_y < volume_cell_counts[1]; ++v_y) {
			for (int v_z = 0; v_z < volume_cell_counts[2]; ++v_z) {
				Eigen::Array3i v(v_x, v_y, v_z);
				size_t buffer_index = getVolumeIndex(volume_cell_counts, v);
				if (bufferDWVector[buffer_index] > 0) {
					float d = bufferDVector[buffer_index];
					if (d >= d_min && d <= d_max) {
						result.push_back(std::make_pair(pose * (volume_cell_size * v.matrix().cast<float>()), d));
					}
				}
			}
		}
	}
}

void OpenCLTSDF::getVoxelDepthAndWeightValues(const Eigen::Affine3f& pose, std::vector<std::pair<Eigen::Vector3f, std::pair<float,float> > >& result)
{
	ensureAllocated();

	// first read the buffer(s)
	std::vector<float> bufferDVector(getCellCount());
	std::vector<float> bufferDWVector(getCellCount());
	volume_buffer_d_.getBufferWrapper().readToFloatVector(bufferDVector);
	volume_buffer_dw_.getBufferWrapper().readToFloatVector(bufferDWVector);

	result.clear();
	for (int v_x = 0; v_x < volume_cell_counts[0]; ++v_x) {
		for (int v_y = 0; v_y < volume_cell_counts[1]; ++v_y) {
			for (int v_z = 0; v_z < volume_cell_counts[2]; ++v_z) {
				Eigen::Array3i v(v_x, v_y, v_z);
				size_t buffer_index = getVolumeIndex(volume_cell_counts, v);
				float d = bufferDVector[buffer_index];
				float dw = bufferDWVector[buffer_index];
				Eigen::Vector3f p = pose * (volume_cell_size * v.matrix().cast<float>());
				result.push_back(std::make_pair(p, std::make_pair(d,dw)));
			}
		}
	}
}

// stupid function
void OpenCLTSDF::getPrettyVoxelCenters(const Eigen::Affine3f& pose, MeshVertexVector & result)
{
	std::vector<std::pair<Eigen::Vector3f, float> > points_and_d;
	const static float massive_value = std::numeric_limits<float>::max();
	getNonzeroFilteredVoxelCenters(pose, -massive_value, massive_value, points_and_d);
	convertPointsAndFloatsToMeshVertices(points_and_d, result);
}

// stupid function
void OpenCLTSDF::getPrettyVoxelCenters(const Eigen::Affine3f& pose, Eigen::Array3f const& box_min, Eigen::Array3f const& box_max, MeshVertexVector & result)
{
	result.clear();
	MeshVertexVector all_points;
	getPrettyVoxelCenters(pose, all_points);
	BOOST_FOREACH(MeshVertex const& v, all_points) {
		if ( (v.p.head<3>().array() >= box_min).all() && (v.p.head<3>().array() <= box_max).all()) {
			result.push_back(v);
		}
	}
}

void OpenCLTSDF::getBufferD(std::vector<float>& result)
{
    if (tsdf_state == TSDF_STATE_GPU) {
        result.resize(getCellCount());
	    volume_buffer_d_.getBufferWrapper().readToFloatVector(result);
    }
    else if (tsdf_state == TSDF_STATE_RAM) {
        result.reserve(getCellCount());
        runLengthDecode(vector_d_rle, result);
    }
    else if (tsdf_state == TSDF_STATE_DISK) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferD(result); // try again!
    }
    else if (tsdf_state == TSDF_STATE_EMPTY) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferD(result); // try again!
    }
    else {
        throw std::runtime_error("nope");
    }
}

void OpenCLTSDF::getBufferDW(std::vector<float>& result)
{
    if (tsdf_state == TSDF_STATE_GPU) {
        result.resize(getCellCount());
	    volume_buffer_dw_.getBufferWrapper().readToFloatVector(result);
    }
    else if (tsdf_state == TSDF_STATE_RAM) {
        result.reserve(getCellCount());
        runLengthDecode(vector_dw_rle, result);
    }
    else if (tsdf_state == TSDF_STATE_DISK) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferDW(result); // try again!
    }
    else if (tsdf_state == TSDF_STATE_EMPTY) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferDW(result); // try again!
    }
    else {
        throw std::runtime_error("nope");
    }
}

void OpenCLTSDF::getBufferC(std::vector<unsigned char>& result)
{
    if (tsdf_state == TSDF_STATE_GPU) {
        result.resize(getCellCount() * 4);
	    volume_buffer_c_.getBufferWrapper().readToByteVector(result);
    }
    else if (tsdf_state == TSDF_STATE_RAM) {
        result.reserve(getCellCount() * 4);
        runLengthDecode(vector_c_rle, result);
    }
    else if (tsdf_state == TSDF_STATE_DISK) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferC(result); // try again!
    }
    else if (tsdf_state == TSDF_STATE_EMPTY) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferC(result); // try again!
    }
    else {
        throw std::runtime_error("nope");
    }
}

void OpenCLTSDF::getBufferCW(std::vector<float>& result)
{
    if (tsdf_state == TSDF_STATE_GPU) {
        result.resize(getCellCount());
	    volume_buffer_cw_.getBufferWrapper().readToFloatVector(result);
    }
    else if (tsdf_state == TSDF_STATE_RAM) {
        result.reserve(getCellCount());
        runLengthDecode(vector_cw_rle, result);
    }
    else if (tsdf_state == TSDF_STATE_DISK) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferCW(result); // try again!
    }
    else if (tsdf_state == TSDF_STATE_EMPTY) {
        setTSDFState(TSDF_STATE_RAM);
        getBufferCW(result); // try again!
    }
    else {
        throw std::runtime_error("nope");
    }
}

void OpenCLTSDF::getAllBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector)
{
	getBufferD(bufferDVector);
	getBufferDW(bufferDWVector);
	getBufferC(bufferCVector);
	getBufferCW(bufferCWVector);
}


// newer version of copyVolumeAxisAligned
// copy all available values that fit from other[other_origin] to this[this_origin]
void OpenCLTSDF::copyVolumeAxisAligned(OpenCLTSDF & other, Eigen::Array3i origin_other, Eigen::Array3i origin_this)
{
	// apply any negatives in one origin as positives to other origin
	for (int i = 0; i < 3; ++i) {
		if (origin_other[i] < 0) {
			origin_this[i] -= origin_other[i];
			origin_other[i] = 0;
		}
		if (origin_this[i] < 0) {
			origin_other[i] -= origin_this[i];
			origin_this[i] = 0;
		}
	}

	// check basic args
	if ( (origin_other < 0).any() || (origin_other >= other.volume_cell_counts).any() ) {
		cout << "copyVolumeAxisAligned bad origin_other: " << origin_other.transpose() << endl;
		throw std::runtime_error("copyVolumeAxisAligned");
	}
	if ( (origin_this < 0).any() || (origin_this >= volume_cell_counts).any() ) {
		cout << "copyVolumeAxisAligned bad origin_this: " << origin_this.transpose() << endl;
		throw std::runtime_error("copyVolumeAxisAligned");
	}

	ensureAllocated();
	other.ensureAllocated();

	size_t src_row_pitch = other.volume_cell_counts[0] * sizeof(float);
	size_t src_slice_pitch = other.volume_cell_counts[0] * other.volume_cell_counts[1] * sizeof(float);
	size_t dst_row_pitch = volume_cell_counts[0] * sizeof(float);
	size_t dst_slice_pitch = volume_cell_counts[0] * volume_cell_counts[1] * sizeof(float);

	// size to copy
	Eigen::Array3i size_this = volume_cell_counts - origin_this;
	Eigen::Array3i size_other = other.volume_cell_counts - origin_other;
	Eigen::Array3i size = size_this.min(size_other);

	// could do some checking
	if ( (size <= 0).any() ) {
		cout << "bad copyVolumeAxisAligned size: " << size.transpose() << endl;
		// exception?
		return;
	}

	cl::size_t<3> src_origin;
	src_origin[0] = origin_other[0] * sizeof(float);
	src_origin[1] = origin_other[1];
	src_origin[2] = origin_other[2];
	cl::size_t<3> dst_origin;
	dst_origin[0] = origin_this[0] * sizeof(float);
	dst_origin[1] = origin_this[1];
	dst_origin[2] = origin_this[2];

	cl::size_t<3> region;
	region[0] = size[0] * sizeof(float);
	region[1] = size[1];
	region[2] = size[2];

	try {
		cl.queue.enqueueCopyBufferRect(other.volume_buffer_d_.getBuffer(), volume_buffer_d_.getBuffer(), src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch);
		cl.queue.enqueueCopyBufferRect(other.volume_buffer_dw_.getBuffer(), volume_buffer_dw_.getBuffer(), src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch);
		cl.queue.enqueueCopyBufferRect(other.volume_buffer_c_.getBuffer(), volume_buffer_c_.getBuffer(), src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch); // uses sizeof(float) == 4 * sizeof(uchar)
		cl.queue.enqueueCopyBufferRect(other.volume_buffer_cw_.getBuffer(), volume_buffer_cw_.getBuffer(), src_origin, dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("copyVolumeAxisAligned\n");
		throw er;
	}
}


void OpenCLTSDF::getAABB(Eigen::Array3f & min_point, Eigen::Array3f & max_point) const
{
	::getAABB(volume_cell_counts, volume_cell_size, min_point, max_point);
}

void OpenCLTSDF::getBBForPose(Eigen::Affine3f const& pose, Eigen::Array3f & min_point, Eigen::Array3f & max_point)
{
	std::vector<Eigen::Vector3f> corners = getVolumeCorners(pose);

	min_point = Eigen::Array3f::Constant(std::numeric_limits<float>::max());
	max_point = Eigen::Array3f::Constant(-std::numeric_limits<float>::max());

	for (std::vector<Eigen::Vector3f>::iterator iter = corners.begin(); iter != corners.end(); ++iter) {
		Eigen::Array3f p = (*iter).array();
		min_point = min_point.min(p);
		max_point = max_point.max(p);
	}
}



size_t OpenCLTSDF::getBytesGPUActual() const
{
	size_t result = 0;
	result += volume_buffer_d_.getSizeInBytes();
	result += volume_buffer_dw_.getSizeInBytes();
	result += volume_buffer_c_.getSizeInBytes();
	result += volume_buffer_cw_.getSizeInBytes();
	return result;
}

size_t OpenCLTSDF::getBytesGPUExpected() const
{
	size_t cells = getCellCount();
	// 3 float and one uchar4 
	return 3 * cells * sizeof(float) + cells * sizeof(unsigned char) * 4;
}

size_t OpenCLTSDF::getBytesRAM() const
{
    // actually compute RAM size (what a novel idea)
    // note capacity, not size!
    size_t result = 0;
    result += vector_d_rle.capacity() * sizeof(float);
    result += vector_dw_rle.capacity() * sizeof(float);
    result += vector_c_rle.capacity() * sizeof(unsigned char);
    result += vector_cw_rle.capacity() * sizeof(float);
    return result;
}



// copied from addVolumeSmart
// first apply pose to the other volume, check all vertices of other against this "cube"
// conservative test...
// ALSO MAYBE WRONG
bool OpenCLTSDF::couldOtherVolumeIntersect(OpenCLTSDF & other, const Eigen::Affine3f & pose)
{
	cout << "WARNING: I think this function couldOtherVolumeIntersect might somehow be wrong (consider epsilon of volume cell size?)" << endl;

	std::vector<Eigen::Vector3f> corners = getVolumeCorners(pose);
	Eigen::Array3f min_point, max_point;
	other.getAABB(min_point, max_point);

	for (int axis = 0; axis < 3; ++axis) {
		bool all_min = true;
		bool all_max = true;
		for (std::vector<Eigen::Vector3f>::iterator iter = corners.begin(); iter != corners.end(); ++iter) {
			Eigen::Array3f corner = iter->array(); // a bit wasteful
			if (corner[axis] >= min_point[axis]) all_min = false;
			if (corner[axis] <= max_point[axis]) all_max = false;
		}
		if (all_min || all_max) return false;
	}
	return true;
}

bool OpenCLTSDF::addVolumeSmart(OpenCLTSDF & other, const Eigen::Affine3f & pose)
{
	if (!couldOtherVolumeIntersect(other, pose)) return false;
	addVolume(other, pose);
	return true;
}

void OpenCLTSDF::getBoundingMesh(Eigen::Affine3f const& pose, Eigen::Vector4ub const& color, MeshVertexVector & vertices, TriangleVector & triangles)
{
	vertices.clear();
	triangles.clear();

	std::vector<Eigen::Vector3f> corners = getVolumeCorners(pose);

	// vertices are just corners
	vertices.resize(8);
	for (int i = 0; i < 8; ++i) {
		MeshVertex & v = vertices[i];
		v.p.head<3>() = corners[i];
		v.p[3] = 1.f;
		// 0 normal ok?
		v.n = Eigen::Vector4f::Zero();
		// color argument
		v.c = color;
	}

	triangles.resize(12);
	int t = 0;
	// left
	triangles[t++] = Eigen::Array3i(0,1,3);
	triangles[t++] = Eigen::Array3i(0,3,2);
	// right
	triangles[t++] = Eigen::Array3i(4,7,5);
	triangles[t++] = Eigen::Array3i(4,6,7);
	// front
	triangles[t++] = Eigen::Array3i(0,2,6);
	triangles[t++] = Eigen::Array3i(0,6,4);
	// back
	triangles[t++] = Eigen::Array3i(1,5,7);
	triangles[t++] = Eigen::Array3i(1,7,3);
	// top
	triangles[t++] = Eigen::Array3i(0,5,1);
	triangles[t++] = Eigen::Array3i(0,4,5);
	// bottom
	triangles[t++] = Eigen::Array3i(2,3,7);
	triangles[t++] = Eigen::Array3i(2,7,6);
}



void OpenCLTSDF::getBoundingLines(Eigen::Affine3f const& pose, Eigen::Vector4ub const& color, MeshVertexVector & vertices)
{
	std::vector<Eigen::Vector3f> corners = getVolumeCorners(pose);
	getLinesForBoxCorners(corners, color, vertices);
}

Eigen::Vector3f OpenCLTSDF::getSphereCenter() const
{
	return sphere_center;
}

float OpenCLTSDF::getSphereRadius() const
{
	return sphere_radius;
}

// now protected, called in constructor
void OpenCLTSDF::generateCenterAndRadius(Eigen::Vector3f & center, float & radius)
{
	Eigen::Array3f min_point, max_point;
	getAABB(min_point, max_point);
	center = (min_point + max_point) / 2;
	radius = (center - min_point.matrix()).norm();
}

bool OpenCLTSDF::sphereTest(OpenCLTSDF const& other, Eigen::Affine3f const& pose) const
{
	// NOTE: I don't believe this epsilon is necessarily required
	const static float epsilon = volume_cell_size;
	Eigen::Vector3f center_this_by_pose = pose * sphere_center;
	return ( (center_this_by_pose - other.sphere_center).norm() <= sphere_radius + other.sphere_radius + epsilon);
}

bool OpenCLTSDF::addVolumeSphereTest(OpenCLTSDF & other, const Eigen::Affine3f & pose)
{
	if (!sphereTest(other, pose)) return false;
	addVolume(other, pose);
	return true;
}

void OpenCLTSDF::addVolume(OpenCLTSDF & other, const Eigen::Affine3f & pose)
{
	ensureAllocated();
	other.ensureAllocated();
	clearMeshCache();

	// todo: separate kernel
	try {
		cl_float16 cl_pose = getCLPose(pose);

		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_int4 cl_volume_dims_other = {other.volume_cell_counts[0], other.volume_cell_counts[1], other.volume_cell_counts[2], 0};
		cl_float2 cl_camera_focal = {camera_focal[0], camera_focal[1]};
		cl_float2 cl_camera_center = {camera_center[0], camera_center[1]};
		cl_int2 cl_camera_size = {camera_size[0], camera_size[1]};


		//////////////////////////
		int kernel_arg = 0;
		add_volume_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, other.volume_buffer_d_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, other.volume_buffer_dw_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, other.volume_buffer_c_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, other.volume_buffer_cw_.getBuffer());
		add_volume_kernel.setArg(kernel_arg++, cl_volume_dims);
		add_volume_kernel.setArg(kernel_arg++, cl_volume_dims_other);
		add_volume_kernel.setArg(kernel_arg++, (cl_float)volume_cell_size);
		add_volume_kernel.setArg(kernel_arg++, (cl_float)other.volume_cell_size);
		add_volume_kernel.setArg(kernel_arg++, (cl_float)volume_max_weight_icp);
		add_volume_kernel.setArg(kernel_arg++, (cl_float)volume_max_weight_color);
		add_volume_kernel.setArg(kernel_arg++, cl_pose);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(add_volume_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("add_volume_kernel\n");
		throw er;
	}
}

void OpenCLTSDF::setVolumeToSphere(float radius)
{
	ensureAllocated();
	clearMeshCache();

	try {
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};

		//////////////////////////
		int kernel_arg = 0;
		set_volume_to_sphere_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		set_volume_to_sphere_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		set_volume_to_sphere_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		set_volume_to_sphere_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		set_volume_to_sphere_kernel.setArg(kernel_arg++, cl_volume_dims);
		set_volume_to_sphere_kernel.setArg(kernel_arg++, (cl_float) volume_cell_size);
		set_volume_to_sphere_kernel.setArg(kernel_arg++, (cl_float) radius);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(set_volume_to_sphere_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("set_volume_to_sphere_kernel\n");
		throw er;
	}
}


void OpenCLTSDF::setMaxWeightInVolume(float new_weight)
{
	ensureAllocated();
	clearMeshCache();

	try {
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};

		//////////////////////////
		int kernel_arg = 0;
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, cl_volume_dims);
		set_max_weight_in_volume_kernel.setArg(kernel_arg++, (cl_float) new_weight);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(set_max_weight_in_volume_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("set_max_weight_in_volume_kernel\n");
		throw er;
	}
}

void OpenCLTSDF::setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius)
{
	ensureAllocated();
	clearMeshCache();

	try {
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_float4 cl_circle_center = {center[0], center[1], center[2], 0};
		cl_uchar4 cl_c_value = {c_value[0], c_value[1], c_value[2], c_value[3]};

		//////////////////////////
		int kernel_arg = 0;
		set_value_in_sphere_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		set_value_in_sphere_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		set_value_in_sphere_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		set_value_in_sphere_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		set_value_in_sphere_kernel.setArg(kernel_arg++, cl_volume_dims);
		set_value_in_sphere_kernel.setArg(kernel_arg++, (cl_float) volume_cell_size);
		set_value_in_sphere_kernel.setArg(kernel_arg++, (cl_float) d_value);
		set_value_in_sphere_kernel.setArg(kernel_arg++, (cl_float) dw_value);
		set_value_in_box_kernel.setArg(kernel_arg++, cl_c_value);
		set_value_in_box_kernel.setArg(kernel_arg++, (cl_float) cw_value);
		set_value_in_sphere_kernel.setArg(kernel_arg++, cl_circle_center);
		set_value_in_sphere_kernel.setArg(kernel_arg++, (cl_float) radius);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(set_value_in_sphere_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("set_value_in_sphere_kernel\n");
		throw er;
	}
}


void OpenCLTSDF::setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose)
{
	ensureAllocated();
	clearMeshCache();

	try {
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_float16 cl_pose = getCLPose(pose);
		cl_float16 cl_pose_inverse = getCLPose(pose.inverse());
		cl_uchar4 cl_c_value = {c_value[0], c_value[1], c_value[2], c_value[3]};

		//////////////////////////
		int kernel_arg = 0;
		set_value_in_box_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		set_value_in_box_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		set_value_in_box_kernel.setArg(kernel_arg++, volume_buffer_c_.getBuffer());
		set_value_in_box_kernel.setArg(kernel_arg++, volume_buffer_cw_.getBuffer());
		set_value_in_box_kernel.setArg(kernel_arg++, cl_volume_dims);
		set_value_in_box_kernel.setArg(kernel_arg++, (cl_float) volume_cell_size);
		set_value_in_box_kernel.setArg(kernel_arg++, (cl_float) d_value);
		set_value_in_box_kernel.setArg(kernel_arg++, (cl_float) dw_value);
		set_value_in_box_kernel.setArg(kernel_arg++, cl_c_value);
		set_value_in_box_kernel.setArg(kernel_arg++, (cl_float) cw_value);
		set_value_in_box_kernel.setArg(kernel_arg++, cl_pose);
		set_value_in_box_kernel.setArg(kernel_arg++, cl_pose_inverse);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(set_value_in_box_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("set_value_in_box_kernel\n");
		throw er;
	}
}

// given an existing set of "points inside" values, set to true those which are inside this volume
void OpenCLTSDF::setPointsInsideBoxTrue(
	const Eigen::Affine3f& pose,
	ImageBuffer const& buffer_depth_image,
	ImageBuffer & buffer_inside_image)
{
	try {
		cl_int2 cl_image_dims = {buffer_depth_image.getCols(), buffer_depth_image.getRows()};
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_float2 cl_camera_c = {camera_center[0], camera_center[1]};
		cl_float2 cl_camera_f = {camera_focal[0], camera_focal[1]};
		cl_float16 cl_pose = getCLPose(pose);

		//////////////////////////
		int kernel_arg = 0;
		set_points_inside_box_true_kernel.setArg(kernel_arg++, buffer_depth_image.getBuffer());
		set_points_inside_box_true_kernel.setArg(kernel_arg++, buffer_inside_image.getBuffer());
		set_points_inside_box_true_kernel.setArg(kernel_arg++, cl_camera_c);
		set_points_inside_box_true_kernel.setArg(kernel_arg++, cl_camera_f);
		set_points_inside_box_true_kernel.setArg(kernel_arg++, cl_image_dims);
		set_points_inside_box_true_kernel.setArg(kernel_arg++, cl_volume_dims);
		set_points_inside_box_true_kernel.setArg(kernel_arg++, (cl_float) volume_cell_size);
		set_points_inside_box_true_kernel.setArg(kernel_arg++, cl_pose);
		//////////////////////////////////

		// set sizes and run
		cl::NDRange global(cl_image_dims.s[0], cl_image_dims.s[1]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(set_points_inside_box_true_kernel, cl::NullRange, global, local);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("set_points_inside_box_true_kernel\n");
		throw er;
	}
}


bool OpenCLTSDF::doesBoxContainSurface(
	Eigen::Array3i const& origin,
	Eigen::Array3i const& size)
{
	ensureAllocated();

	try {
		cl_int4 cl_volume_dims = {volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2], 0};
		cl_int4 cl_box_origin = {origin[0], origin[1], origin[2], 0};
		cl_int4 cl_box_size = {size[0], size[1], size[2], 0};
		BufferWrapper result_buffer(cl);
		std::vector<int> result_buffer_vec(1,0);
		result_buffer.writeFromIntVector(result_buffer_vec);

		//////////////////////////
		int kernel_arg = 0;
		does_box_contain_surface_kernel.setArg(kernel_arg++, volume_buffer_d_.getBuffer());
		does_box_contain_surface_kernel.setArg(kernel_arg++, volume_buffer_dw_.getBuffer());
		does_box_contain_surface_kernel.setArg(kernel_arg++, result_buffer.getBuffer());
		does_box_contain_surface_kernel.setArg(kernel_arg++, cl_volume_dims);
		does_box_contain_surface_kernel.setArg(kernel_arg++, cl_box_origin);
		does_box_contain_surface_kernel.setArg(kernel_arg++, cl_box_size);
		//////////////////////////////////

		// set sizes and run
		//cl::NDRange global(volume_cell_counts[0], volume_cell_counts[1], volume_cell_counts[2]);
		cl::NDRange global(size[0], size[1], size[2]);
		cl::NDRange local = cl::NullRange;
		cl.queue.enqueueNDRangeKernel(does_box_contain_surface_kernel, cl::NullRange, global, local);

		result_buffer.readToIntVector(result_buffer_vec);
		return (bool)(result_buffer_vec[0]);
	}
	catch (cl::Error er) {
		printf("cl::Error: %s\n", oclErrorString(er.err()));
		printf("does_box_contain_surface_kernel\n");
		throw er;
	}
}

bool OpenCLTSDF::doesVolumeContainSurface()
{
	return doesBoxContainSurface(Eigen::Array3i(0,0,0), volume_cell_counts);
}

// could extend to color if interesting...
void OpenCLTSDF::extractSlice(int axis, int position, ImageBuffer & result_d, ImageBuffer & result_dw)
{
	ensureAllocated();

	KernelExtractVolumeSlice _KernelExtractVolumeSlice(*all_kernels_);
	_KernelExtractVolumeSlice.runKernel(volume_buffer_d_, axis, position, result_d);
	_KernelExtractVolumeSlice.runKernel(volume_buffer_dw_, axis, position, result_dw);
}
