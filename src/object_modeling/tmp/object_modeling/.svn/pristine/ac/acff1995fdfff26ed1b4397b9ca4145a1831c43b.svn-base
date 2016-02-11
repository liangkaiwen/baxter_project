#pragma once

#include "cll.h"
#include "convertOpenCLEigen.h"
#include "OpenCLAllKernels.h"
#include "RenderBuffers.h"
#include "VolumeBuffer.h"

// for frame-style normals
#include "OpenCLNormals.h"

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <opencv2/opencv.hpp>

#include "EigenUtilities.h"

#include "MeshTypes.h"

enum TSDFState {
    TSDF_STATE_EMPTY,
    TSDF_STATE_GPU,
    TSDF_STATE_RAM,
    TSDF_STATE_DISK
};

class OpenCLTSDF
{
public:

	OpenCLTSDF(boost::shared_ptr<OpenCLAllKernels> all_kernels, 
		float camera_focal_x, float camera_focal_y,
		float camera_center_x, float camera_center_y,
		int camera_size_x, int camera_size_y,
		int volume_cell_count_x, int volume_cell_count_y, int volume_cell_count_z, float volume_cell_size,
		float volume_max_weight_icp, float volume_max_weight_color, bool use_most_recent_color, float min_truncation_distance,
        fs::path temp_base_path = fs::path());

	// base a bunch on "other".  Except create empty, based on additional args
	OpenCLTSDF(OpenCLTSDF const& other, int volume_cell_count_x, int volume_cell_count_y, int volume_cell_count_z);

	// new version
	OpenCLTSDF* clone();

    // will set state to GPU (if needed), then overwrite GPU with empty values
	void makeEmptyInPlace();

	void addFrame(const Eigen::Affine3f& pose,
		ImageBuffer const& buffer_depth_image,
		ImageBuffer const& buffer_color_image,
		ImageBuffer const& buffer_segments,
		int which_segment);

	void addFrame(const Eigen::Affine3f& pose,
		ImageBuffer const& buffer_depth_image,
		ImageBuffer const& buffer_color_image,
		ImageBuffer const& buffer_segments,
		ImageBuffer const& buffer_depth_weights,
		ImageBuffer const& buffer_color_weights,
		int which_segment);

	// points only
	void renderPoints(
		const Eigen::Affine3f& model_pose,
		const Eigen::Array2f & camera_focal,
		const Eigen::Array2f & camera_center,
		const Eigen::Array2f & min_max_depth,
		bool replace_only_if_nearer,
		int mask_value,
		RenderBuffers& render_buffers);

	// colors only (based on existing points)
	void renderColors(
		const Eigen::Affine3f& model_pose,
		const Eigen::Array2f & camera_focal,
		const Eigen::Array2f & camera_center,
		const Eigen::Array2f & min_max_depth,
		bool replace_only_if_nearer,
		int mask_value,
		RenderBuffers& render_buffers);


	void renderFrame(const Eigen::Affine3f& model_pose,
		const Eigen::Array2f &camera_focal,
		const Eigen::Array2f &camera_center,
		const Eigen::Array2f &min_max_depth,
		bool replace_only_if_nearer,
		int mask_value,
		RenderBuffers& render_buffers);

	// newer version of copyVolumeAxisAligned
	// copy all available values that fit from other[other_origin] to this[this_origin]
	void copyVolumeAxisAligned(OpenCLTSDF & other, Eigen::Array3i origin_other, Eigen::Array3i origin_this);

	// from addVolumeSmart, for external testing...
	bool couldOtherVolumeIntersect(OpenCLTSDF & other, const Eigen::Affine3f & pose);

	// bomb out quickly if no possible intersection
	// pose applied to this volume puts it in the other volume
	// for example, takes origin of this volume to wherever it is in the other volume.  I think.
	bool addVolumeSmart(OpenCLTSDF & other, const Eigen::Affine3f & pose);

	// an alternative smart add
	bool sphereTest(OpenCLTSDF const& other, Eigen::Affine3f const& pose) const;
	bool addVolumeSphereTest(OpenCLTSDF & other, const Eigen::Affine3f & pose);

	// call the addvolume kernel (may need to allocate "other" so not const)
	// pose applied to this volume puts it in the other volume
	void addVolume(OpenCLTSDF & other, const Eigen::Affine3f & pose);

	// this generates only the "valid" mesh (using the other generateMesh and extractValidVerticesAndTriangles)
	void generateMesh(MeshVertexVector & vertices, TriangleVector & triangles);

	// this generates the full mesh and vertex validity information
	void generateMeshAndValidity(MeshVertexVector & all_vertices, TriangleVector & all_triangles, std::vector<bool> & vertex_validity);

	void getBoundingMesh(Eigen::Affine3f const& pose, Eigen::Vector4ub const& color, MeshVertexVector & vertices, TriangleVector & triangles);

	void getBoundingLines(Eigen::Affine3f const& pose, Eigen::Vector4ub const& color, MeshVertexVector & vertices);

	void setVolumeToSphere(float radius);

	void setMaxWeightInVolume(float new_weight);

	void setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius);

	void setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose);


	float getEmptyDValue() const {return -1;}
	size_t getCellCountX() const {return volume_cell_counts[0];}
	size_t getCellCountY() const {return volume_cell_counts[1];}
	size_t getCellCountZ() const {return volume_cell_counts[2];}
	size_t getCellCount() const {return volume_cell_counts[0] * volume_cell_counts[1] * volume_cell_counts[2];}

	Eigen::Array3i getVolumeCellCounts() const {return volume_cell_counts;}
	float getVolumeCellSize() const {return volume_cell_size;}

    size_t getBytesGPUExpected() const;
	size_t getBytesGPUActual() const;
    size_t getBytesRAM() const;

    // todo: include this in getBytesRAM?  The problem is that we clear the mesh cache separately
    // could have a "free RAM" function to both put stuff to disk and clear this...
    size_t getBytesMeshCache() const; 
    void clearMeshCache();

	void getAABB(Eigen::Array3f & min_point, Eigen::Array3f & max_point) const;

	void getBBForPose(Eigen::Affine3f const& pose, Eigen::Array3f & min_point, Eigen::Array3f & max_point);

	std::vector<Eigen::Vector3f> getVolumeCorners(const Eigen::Affine3f& pose) const;

	void getVoxelDepthAndWeightValues(const Eigen::Affine3f& pose, std::vector<std::pair<Eigen::Vector3f, std::pair<float,float> > >& result);

	void getNonzeroFilteredVoxelCenters(const Eigen::Affine3f& pose, float d_min, float d_max, std::vector<std::pair<Eigen::Vector3f, float> >& result);

	// a prettier version of getNonzeroFilteredVoxelCenters, provides colored mesh vertices
	// box is after applying pose
	void getPrettyVoxelCenters(const Eigen::Affine3f& pose, MeshVertexVector & result);
	void getPrettyVoxelCenters(const Eigen::Affine3f& pose, Eigen::Array3f const& box_min, Eigen::Array3f const& box_max, MeshVertexVector & result);

	void getBufferD(std::vector<float>& result);
	void getBufferDW(std::vector<float>& result);
	void getBufferC(std::vector<unsigned char>& result);
	void getBufferCW(std::vector<float>& result);
	void getAllBuffers(std::vector<float>& bufferDVector, std::vector<float>& bufferDWVector, std::vector<unsigned char>& bufferCVector, std::vector<float>& bufferCWVector);

	// for user controlled saving and loading
    // these may change the state of the TSDF
	void save(fs::path const& folder);
	void load(fs::path const& folder);
    void loadLazy(fs::path const& folder);

	// could make accessors for these...
	// were for planar patch volumes
	Eigen::Array3i min_expand_locked;
	Eigen::Array3i max_expand_locked;

	static uint64_t getDeallocationCounter() {return deallocation_counter;}
	static uint64_t getReallocationCounter() {return reallocation_counter;}
    static uint64_t getSaveToDiskCounter() {return save_to_disk_counter;}
    static uint64_t getLoadFromDiskCounter() {return load_from_disk_counter;}

	Eigen::Vector3f getSphereCenter() const;
	float getSphereRadius() const;

	// given an existing set of "points inside" values, set to true those which are inside this volume
	void setPointsInsideBoxTrue(
		const Eigen::Affine3f& pose,
		ImageBuffer const& buffer_depth_image,
		ImageBuffer & buffer_inside_image);

	bool doesBoxContainSurface(
		Eigen::Array3i const& origin,
		Eigen::Array3i const& size);

	// convenience function for doesBoxContainSurface on whole volume
	bool doesVolumeContainSurface();

	void extractSlice(int axis, int position, ImageBuffer & result_d, ImageBuffer & result_dw);

	// just so you can get the silly cl when you need it
	CL& getCL() {return cl;}

    // the new way to change the state of TSDF
    TSDFState getTSDFState() const;
    void setTSDFState(TSDFState new_state);

protected:
	void constructorBody();

    // these assume you are in the right state to call them
    void changeStateGPUToRAM();
    void changeStateRAMToGPU();
    void changeStateRAMToDisk();
    void changeStateDiskToRAM();
    void changeStateEmpty(); // can be called from any state

    // this could also be called EMPTYToGPU...
    void allocateEmptyVolumeBuffers();

    // shorthand for setTSDFState(GPU);
    // I had a bunch of code that called this already
    void ensureAllocated();

	void allocateVolumeBuffersWithValues(std::vector<float> const& d, std::vector<float> const& dw, std::vector<unsigned char> const& c, std::vector<float> const& cw);

	cl::Buffer allocateBufferCheckError(size_t size, void* data);

	void makeBuffersNull();

	void freeRLEMemory();

	std::string getBoostArchiveName() {return "opencl_tsdf_boost_archive.bin";}

	void generateTempPath();

	void generateCenterAndRadius(Eigen::Vector3f & center, float & radius);

	void allocateFloatBufferFast(size_t number_of_floats, float float_value, cl::Buffer & result_buffer, size_t & result_buffer_size_bytes);

	//////////////////
	// members

	CL& cl;
	boost::shared_ptr<OpenCLAllKernels> all_kernels_;

    TSDFState tsdf_state;

	// for deallocating
	std::vector<std::pair<size_t, float> > vector_d_rle;
	std::vector<std::pair<size_t, float> > vector_dw_rle;
	std::vector<std::pair<size_t, unsigned char> > vector_c_rle;
	std::vector<std::pair<size_t, float> > vector_cw_rle;

	// for disk deallocating
	fs::path temp_base_path; // can be the same across TSDFs
	fs::path temp_generated; // must be unique to this TSDF (generated on construction and clone)
    fs::path load_lazy_path;

	// for caching mesh generation (don't need to save as long as you always start with mesh_cache_is_valid=false)
	bool mesh_cache_is_valid;
	MeshVertexVector mesh_cache_all_vertices;
	TriangleVector mesh_cache_all_triangles;
	std::vector<bool> mesh_cache_vertex_validity;

	// for tracking de- and re-allocations
	static uint64_t reallocation_counter;
	static uint64_t deallocation_counter;
    static uint64_t save_to_disk_counter;
    static uint64_t load_from_disk_counter;

	// for fast inside checking
	Eigen::Vector3f sphere_center;
	float sphere_radius;

	///// params
	Eigen::Array2f camera_focal;
	Eigen::Array2f camera_center;
	Eigen::Array2i camera_size;
	Eigen::Array3i volume_cell_counts;
	float volume_cell_size;
	float volume_max_weight_icp;
	float volume_max_weight_color;
	bool use_most_recent_color; // should be similar to volume_max_weight_color ~= 0
	float min_truncation_distance;

	////// buffers
	VolumeBuffer volume_buffer_d_;
	VolumeBuffer volume_buffer_dw_;
	VolumeBuffer volume_buffer_c_;
	VolumeBuffer volume_buffer_cw_;

	// kernels (stop this!)
	cl::Kernel add_frame_kernel;
	cl::Kernel add_volume_kernel;
	cl::Kernel set_volume_to_sphere_kernel;
	cl::Kernel set_max_weight_in_volume_kernel;
	cl::Kernel set_value_in_sphere_kernel;
	cl::Kernel set_value_in_box_kernel;
	cl::Kernel set_points_inside_box_true_kernel;
	cl::Kernel does_box_contain_surface_kernel;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

typedef boost::shared_ptr<OpenCLTSDF> OpenCLTSDFPtr;
