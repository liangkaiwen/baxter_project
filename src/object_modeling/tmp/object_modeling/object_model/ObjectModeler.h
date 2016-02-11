#pragma once

#include "typedefs.h"
#include "Keypoints.hpp"
#include "Frame.hpp"
#include "MaskObject.h"
#include "ToggleBase.hpp"
#include "ToggleCloud.hpp"
#include "ToggleNormals.hpp"
#include "ToggleCloudNormals.hpp"
#include "ToggleLineSet.hpp"
#include "ToggleMesh.hpp"
#include "cloudToImage.hpp"
#include "parameters.h"
#include "runningStatistics.h"
#include "CIsoSurface.h"
#include "LearnHistogram.h"
#include "histogramUtil.h"
#include "Frustum.h"

// opencl Lib
#include <RenderBuffers.h>
#include <OpenCLTSDF.h>
#include <OpenCLOptimize.h>
#include <OpenCLNormals.h>
#include <OpenCLImages.h>
#include <OpenCLAllKernels.h>

// g2o lib
#include <G2OPoseGraph.h>
#include <G2OStereoProjector.hpp>

// PCL
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

class ObjectModeler
{
public:
	typedef pcl::PointCloud<pcl::PointXYZRGBNormal> VertexCloudT;
	typedef std::vector<pcl::Vertices> TrianglesT;
	typedef boost::shared_ptr<TrianglesT> TrianglesPtrT;

private:
	ObjectModeler(const ObjectModeler& other);
	void operator=(const ObjectModeler& other);

public:
	struct CameraPoseStruct {
		int vertex_id;
		Eigen::Affine3f pose;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	typedef boost::shared_ptr<CameraPoseStruct> CameraPoseStructPtr;
	
	struct PVStruct {
		OpenCLTSDFPtr tsdf_ptr;
		Eigen::Affine3f pose;
		int vertex_id;
		int frame_last_in_frustum;
		Eigen::Vector3f normal; // starts as (0,0,-1) (is relative to "pose")
		int points_added; // counts the number of points added to this patch volume (to track weighted mean normal)
		Eigen::Affine3f original_pose; // set this once, to compare against loop closure result

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	typedef boost::shared_ptr<PVStruct> PVStructPtr;


	ObjectModeler(Parameters& params);

	void setActive(bool active);
	bool processFrame(FrameT& frame);
	void stop();
	bool wasStopped();
	bool isEmpty() const;
	bool isInLiveMode();
	void processWaitKey();
	void pauseAndShowImages();
	void enqueueShowImage(std::string name, cv::Mat image);
	void showQueuedImages();
	
	// whatever debug info you happen to be keeping
	void writeTablesOfValues(fs::path folder);

	// do these need to be public?
	void cloudViewerOnVisualizationThreadOnce (pcl::visualization::PCLVisualizer& viewer);
	void cloudViewerOnVisualizationThread (pcl::visualization::PCLVisualizer& viewer);
	void cloudViewerKeyboardCallback (const pcl::visualization::KeyboardEvent& keyboard_event, void* cookie);

	void generateMesh();

protected:

	bool is_active;
	bool is_tracking;
	bool is_stopping; // maybe wrap in mutex??
	Parameters& params;

	// sync keyboard input (both vis thread and main thread)
	boost::mutex mutex_process_key;
	boost::mutex mutex_volumes; // needed for patch volumes...
	boost::mutex mutex_pv_meshes_show; // needed for the list of meshes

	// for out-of-thread image showing:
	boost::mutex mutex_show_image;
	std::list<std::pair<std::string, cv::Mat> > queue_show_image;

	////// functions

	void prepareFrame(FrameT& frame);
	bool addFrameToCalibrate(FrameT& frame);
	bool learnHistogram(FrameT& frame);
	bool alignAndAddFrame(FrameT& frame);
	bool writeFrame(FrameT& frame, fs::path folder);
	void processKey(char key);
	void initializeLearnHistogram();
	cv::Mat getPrettyDepthImage(cv::Mat raw_depth_image);

	bool doObjectMasking(FrameT& frame);
	void copyNormalsFromImageBuffers(FrameT& frame); // call if you need them off the GPU in the frame
	void addNormalsToFrame(FrameT& frame);
	bool alignPreparedFrame(FrameT& frame);
	void addFrameToModel(FrameT& frame);
	void addKeyframe(FrameT const& frame, std::vector<int> const& updated_this_frame);

	bool readTransforms(fs::path const& file, std::vector<std::pair<bool, Eigen::Affine3f> >& vector);
	bool readWhichPVs(fs::path const& file, std::vector<std::vector<int> >& vector);

	void combinedDebugImagesNew(FrameT const& frame, Eigen::Affine3f const& transform, int image_channel_count,
		std::vector<float> const& error_vector, std::vector<float> const& weight_vector,
		cv::Rect const& render_rect,
		std::vector<cv::Mat> const& frame_images_full_size, cv::Rect const& object_rect,
		RenderBuffers const& render_buffers, Eigen::Vector2f const& render_proj_f, Eigen::Vector2f const& render_proj_c);

	size_t getImageChannelCount() const;
	std::vector<ImageBuffer> getImageChannelsList(ImageBuffer const& color_bgra_uchar, size_t width, size_t height);
	ImageBuffer packImageChannelsList(std::vector<ImageBuffer> const& image_buffer_list, size_t width, size_t height);

	void showAndSaveRenderDebugImages(FrameT const& frame,
		std::string name, 
		bool do_save,
		RenderBuffers & render_buffers,
		cv::Rect& render_rect);

	void renderAfterAlign(FrameT const& frame,
		Eigen::Affine3f const& pose);

	bool renderForAlign(FrameT const& frame,
		bool is_loop_closure,
		Eigen::Affine3f const& initial_pose,
		RenderBuffers & render_buffers,
		cv::Rect & render_rect);

	CloudICPTargetT::Ptr renderBuffersToCloud(RenderBuffers const& render_buffers, cv::Rect const& render_rect);

	bool alignWithCombinedOptimizationNew(
		FrameT const& frame,
		RenderBuffers const& render_buffers,
		cv::Rect const& render_rect,
		bool show_debug_images,
		Eigen::Affine3f const& initial_pose,
		Eigen::Affine3f const& initial_relative_pose, 
		Eigen::Affine3f& result_pose);

#if 0
	bool alignWithCombinedOptimization(
		const FrameT& frame,
		bool is_loop_closure,
		const Eigen::Affine3f& initial_pose,
		Eigen::Affine3f& result_pose,
		CloudICPTargetT::Ptr& last_render_cloud,
		Eigen::Affine3f& pose_for_last_render_cloud,
		cv::Mat& which_segment);
#endif

	void addObjectFeaturesToFrame(FrameT& frame);
	//bool alignWithFeaturesToModel(const Eigen::Affine3f& initial_pose, FrameT& frame, Eigen::Affine3f& result_pose, std::vector<cv::DMatch>& inlier_matches);
	bool alignWithFeaturesToModel(Eigen::Affine3f const& initial_pose, FrameT const& frame, KeypointsT const& model_keypoints, Eigen::Affine3f& result_pose, std::vector<cv::DMatch>& inlier_matches);
	void updateModelKeypoints(const Eigen::Affine3f& object_pose, const FrameT& frame, const std::vector<cv::DMatch>& inlier_matches);
	CloudT::Ptr getVolumePointsFromOpenCLTSDFForPose(OpenCLTSDF & tsdf, const Eigen::Affine3f& object_pose, bool show_max_points) const;
	CloudT::Ptr getRidOfNaNs(CloudT::ConstPtr cloud);
	KeypointCloudT::Ptr computeObjectKPProjectionCloud(const FrameT& frame);
	
	void renderVolumeWithPose(const Eigen::Affine3f& custom_object_pose, const Eigen::Vector3f& light_direction, float scale, cv::Mat& result_colors, cv::Mat& result_normals, cv::Mat& result_depth);

	void renderVolumeWithOpenCL(const Eigen::Affine3f& object_pose, const Eigen::Vector2f& proj_f, const Eigen::Vector2f& proj_c, float render_min_depth, float render_max_depth, const cv::Rect& render_rect, 
		RenderBuffers & render_buffers);

	void extractRenderBuffersToClouds(cv::Rect const& render_rect, RenderBuffers const& render_buffers,
		CloudT& point_cloud, pcl::PointCloud<pcl::Normal>& normal_cloud);

	CloudT::Ptr getLineSetEdges(const std::vector<Eigen::Vector3f>& corners, const cv::Vec3b& color) const;
	CloudT::Ptr getLineSetEdges(const std::vector<Eigen::Vector3f>& corners) const;
	boost::shared_ptr<std::vector<pcl::Vertices> > getMeshVerticesForCorners(int offset);
	void showVolumeEdges();
	bool allPossibleEdgeVerticesNonzero(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const;
	bool allSurroundingVerticesNonzero(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const;
	bool allSurroundingVerticesNonzeroOrOutsideVolume(OpenCLTSDF const& tsdf, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f) const;
	Eigen::Matrix<unsigned char, 4, 1> interpolateColorForMesh(OpenCLTSDF const& tsdf, const std::vector<unsigned char> & color_vector, const Eigen::Vector3f& voxel_coords_f);
	bool loadBothHistograms();
	fs::path prepareDumpPath();
	fs::path prepareDumpPath(fs::path subfolder);
	fs::path getPNGFilenameWithIndex(const std::string& filename_prefix, int index);
	void savePNGWithIndex(const std::string& filename_prefix, const cv::Mat& image, int index);
	void savePCDWithIndex(const std::string& filename_prefix, const CloudT& cloud, int index);

	bool isReadyToActivate();

	pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const CloudT::ConstPtr& cloud);

	// old CPU versions
	pcl::PointCloud<pcl::Normal>::Ptr computeNormalsCrossProduct(const CloudT::ConstPtr& cloud);
	pcl::PointCloud<pcl::Normal>::Ptr smoothNormals(const CloudT::ConstPtr& cloud, const pcl::PointCloud<pcl::Normal>::ConstPtr& input_normals);
	cv::Mat getMaskOfValidNormals(const pcl::PointCloud<pcl::Normal>& normal_cloud);

	void prepareSegmentColorMap(int max_number);
	void prepareSegmentColorMap(const cv::Mat& segment_image);
	cv::Mat randomlyColorSegments(const cv::Mat& segment_image, const cv::Mat& mask_image);

	void segmentByMerging(const FrameT& frame, 
		const cv::Mat& input_segmentation,
		cv::Mat& output_segmentation, 
		std::vector<int>& output_component_sizes, 
		std::vector<Eigen::Vector3f>& output_mean_normals);

	void globalStopWatchMark(const std::string& s);

	// patch grid
	void updateGrid(FrameT const& frame);

	// patch volumes
	void updatePatchVolumes(FrameT const& frame);

	void getInitialSegmentation(const FrameT& frame, cv::Mat & result_render_segments, cv::Mat & result_consistent_segments);

	CloudT::Ptr getCloudForSegment(FrameT const& frame, int c, cv::Mat const& segments) const;

	void renderPatchVolumesWithPose(const Eigen::Affine3f& custom_object_pose, const Eigen::Vector3f& light_direction, float scale,
		cv::Mat& result_colors, cv::Mat& result_normals, cv::Mat& result_depth, cv::Mat& result_segments);

	void renderPatchVolumes(const Eigen::Affine3f& object_pose, const Eigen::Vector2f& proj_f, const Eigen::Vector2f& proj_c, float render_min_depth, float render_max_depth, const cv::Rect& render_rect,
		int min_age, int max_age, float max_normal_angle, bool deallocate_after, bool update_frame_in_frustum, 
		RenderBuffers & render_buffers);


	CloudT::Ptr getLinesForAllPatchVolumes();
	CloudT::Ptr getLinesForPatchVolume(int c);

	void addPointsToPatchVolumeWithBuffers(const FrameT& frame, int c, cv::Mat const& segments, std::vector<int> const& segment_sizes, std::vector<Eigen::Vector3f> const& segment_normals, 
		ImageBuffer const& buffer_depth_image, ImageBuffer const& buffer_color_image, ImageBuffer const& buffer_segments, ObjectModeler::PVStruct & pv_struct) const;

	void expandPatchVolumeToContainPoints(FrameT const& frame, int c, cv::Mat const& segments, std::vector<Eigen::Vector3f> const& segment_normals, ObjectModeler::PVStruct & pv_struct);
	PVStructPtr createNewPatchVolumeFromCloud(const CloudT::Ptr & cloud_ptr, const Eigen::Vector3f & segment_normal);
	PVStructPtr createNewPatchVolume(Eigen::Array3i const& cell_counts, float cell_size, Eigen::Affine3f const& pose);
	
	void getBoundingBox(CloudT::Ptr const& cloud_ptr, Eigen::Array3f & bb_min_result, Eigen::Array3f & bb_max_result) const;

	void getConsistentRenderSegments(FrameT const& frame, CloudT const& render_cloud, pcl::PointCloud<pcl::Normal> const& render_normal_cloud, cv::Mat const& render_segments, cv::Mat& result_segments);
	bool updatePatchVolumeAndCameraPoseGraph(const FrameT& frame);

	bool checkForLoopClosure(FrameT const& frame, Eigen::Affine3f & result_object_pose, std::set<int> & result_old_pvs_used, CloudICPTargetT::Ptr & last_render_cloud, Eigen::Affine3f & pose_for_last_render_cloud);
	void optimizePoseGraph();

	cv::Vec3b getPVStatusColor(int c);
	void showPVGraphEdges(Eigen::Affine3f const& pose);
	void showPVVolumeMesh();
	void showPVVolumeNormals();
	void updatePatchVolumeVisualizer();
	size_t getAllocatedPatchVolumeSize();
	size_t getRequiredPatchVolumesSize();
	void printPatchVolumeMemorySummary();
	void deallocateUntilUnderMaxSize();

	void setVolumeToDebugSphere();

	// meshes
	void generateMeshForTSDF(OpenCLTSDF & tsdf, VertexCloudT::Ptr & result_vertex_cloud, TrianglesPtrT & result_triangles);
	void saveMesh(fs::path filename, VertexCloudT::Ptr const& vertex_cloud_ptr, TrianglesPtrT const& triangles_ptr);
	void appendToMesh(VertexCloudT::Ptr & main_vertices_ptr, TrianglesPtrT & main_triangles_ptr, VertexCloudT::Ptr const& vertices_to_add_ptr, TrianglesPtrT const& triangles_to_add_ptr);

	// currently set in constructor (and fixed):
	Eigen::Vector3f light_direction;
	enum FeatureTypeEnum {FEATURE_TYPE_FAST, FEATURE_TYPE_SURF, FEATURE_TYPE_ORB};
	FeatureTypeEnum feature_type;

	// don't have meaningful state:
	MaskObject mask_object;
	G2OStereoProjector<KeypointPointT, KeypointPointT> g2o_stereo_projector;
	LearnHistogram learn_histogram;
	std::ofstream ofs_object_pose;
	std::ofstream ofs_camera_pose;
	std::vector<std::pair<bool, Eigen::Affine3f> > input_object_poses;
	std::vector<std::pair<bool, Eigen::Affine3f> > input_loop_poses;
	std::vector<std::vector<int> > input_loop_which_pvs;
	bool paused_by_user;
	bool any_failures_save_exception;
	
	// do have meaningful state (need to serialize)
	KeypointsT model_kp;
	Eigen::Affine3f object_pose;
	Eigen::Affine3f previous_relative_transform;
	cv::Rect previous_object_rect;
	cv::Mat previous_object_pixels;
	int input_frame_counter; // incremented at beginning of addAndAlign
	int output_frame_counter; // only increments on success in addAndAlign (at end)
	float previous_error_color;
	// HERE...previous full render rect?
	// end serialize

	// opencl stuff:
	boost::scoped_ptr<CL> cl_ptr;
	boost::scoped_ptr<class OpenCLAllKernels> all_kernels_ptr;
	boost::scoped_ptr<OpenCLOptimize> opencl_optimize_ptr;
	boost::scoped_ptr<OpenCLNormals> opencl_normals_ptr;
	boost::scoped_ptr<OpenCLImages> opencl_images_ptr;
	boost::scoped_ptr<OpenCLTSDF> opencl_tsdf_ptr; // only use this when not using patch volumes

	// pcl visualizer
	boost::scoped_ptr<pcl::visualization::CloudViewer> cloud_viewer_ptr;
	ToggleCloud<PointT> tc_masked_points;
	ToggleCloud<PointT> tc_model_kp;
	ToggleLineSet<PointT> tls_volume_corners;
	ToggleLineSet<PointT> tls_patch_volume_debug;
	ToggleMesh<PointT> tm_patch_volume_debug;
	ToggleLineSet<PointT> tls_patch_volume_normals;
	ToggleLineSet<PointT> tls_graph_edges;
	ToggleLineSet<PointT> tls_frustum;
	std::vector<boost::shared_ptr<ToggleMesh<pcl::PointXYZRGBNormal> > > tm_pv_show_generated_list; // lock mutex_pv_meshes_show
	std::vector<boost::shared_ptr<ToggleCloud<PointT> > > tc_pv_compare_to_mesh_list; // lock mutex_pv_meshes_show
	bool render_on_next_viewer_update;
	int pv_test_image_index;
	bool save_screenshot_on_next_viewer_update;
	int save_screenshot_counter;
	bool generate_mesh_on_next_viewer_update;

	// running stats
	RunningStatistics rs_addAndAlign;
	RunningStatistics rs_gn_iterations;
	RunningStatistics rs_add_normals_to_frame_inside;
	RunningStatistics rs_add_normals_to_frame_outside;
	RunningStatistics rs_pvs_touched;
	RunningStatistics rs_pvs_rendered;
	RunningStatistics rs_optimize_pose_graph;

	// global timing
	pcl::StopWatch g_sw;

	// tracking of per frame numbers
	typedef std::map<std::string, std::vector<float> > TablesOfValuesT;
	TablesOfValuesT tables_of_values;
	float getLastDelta(std::string name);
	float getRecentMean(std::string name, size_t count);
	
	// for calibration mode
	std::vector<std::vector<cv::Point3f> > calibrate_object_points;
	std::vector<std::vector<cv::Point2f> > calibrate_image_points;

	// histogram learning
	bool initialize_learn_histogram_next_frame;
	bool learning_histogram_hand;
	bool learning_histogram_object;

	// segments && patch volumes
	std::map<int, cv::Vec3b> segment_color_map;
	boost::shared_ptr<G2OPoseGraph> pv_pose_graph_ptr;
	size_t loop_closure_count;
	
	// The patch volumes
	std::vector<PVStructPtr> pv_list;
	std::map<int, PVStructPtr> vertex_to_pv_map;

	// Camera poses
	std::vector<CameraPoseStructPtr> camera_list;
	std::map<int, CameraPoseStructPtr> vertex_to_camera_map;

	// Edge information
	std::set<std::pair<int, int> > edge_is_loop_set;

	// Keyframes
	typedef boost::shared_ptr<FrameT> FrameTPtr;
	struct KeyframeStruct {
		FrameTPtr frame_ptr;
		int vertex_id;
	};
	std::vector<KeyframeStruct> keyframes;
	std::map<int, std::set<int> > pv_to_keyframes_map;

	// Patch grid
	//typedef std::map<boost::tuple<int,int,int>, int> GRID_TO_LIST_MAP;
	typedef std::map<boost::tuple<int,int,int>, std::vector<int> > GRID_TO_LIST_MAP;
	GRID_TO_LIST_MAP grid_to_list_map;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};



struct MergeEdge {
	int a;
	int b;
	float w;
};
bool operator<(const MergeEdge &a, const MergeEdge &b);

