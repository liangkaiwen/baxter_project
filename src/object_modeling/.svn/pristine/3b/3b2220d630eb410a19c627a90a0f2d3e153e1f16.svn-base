#pragma once

#include "G2OPoseGraph.h"

#include "model_base.h"
class ModelSingleVolume;

#include "edge_struct.h"
#include "keyframe_struct.h"

#include "feature_matching.h"

// hmmm...how to avoid always needing to build this
#include "dbow_place_recognition.h"

#include "ordering_container.h"

class ModelGrid : public ModelBase
{
public:
	friend class ModelSingleVolume;

	// types:
	// maybe make a general multi-volume base class?
	// also, the segment id is currently (index+1)
	struct GridStruct {
		OpenCLTSDFPtr tsdf_ptr;
		Eigen::Affine3f pose_external;
		Eigen::Affine3f pose_original; // set this once, to compare against loop closure result of pose_external
		Eigen::Affine3f pose_tsdf;
		int age; // frames since last updated
		bool active;
		bool merged;

		inline Eigen::Affine3f getExternalTSDFPose() { return pose_external * pose_tsdf; }

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	typedef boost::shared_ptr<GridStruct> GridStructPtr;

	typedef std::map<boost::tuple<int,int,int>, std::vector<int> > GridToListMapT;
	typedef std::map<int, boost::tuple<int,int,int> > GridListToGridCellT; // for debugging only?

	/////////////////////////////////
	// methods:
	ModelGrid(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
	ModelGrid(ModelGrid const& other);

	virtual ModelGrid* clone();
	
	// takes care of base classes too
    virtual void reset();

    virtual void renderModel(
            const ParamsCamera & params_camera,
            const Eigen::Affine3f & model_pose,
            RenderBuffers & render_buffers);

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

	virtual void generateMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list);

	virtual void generateMeshAndValidity(
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list, 
		std::vector<bool> & vertex_validity, 
		std::vector<bool> & triangle_validity);

	virtual void generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes);

	virtual void getBuffersLists(
		std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors, 
		std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors, 
		std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
		std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list);

	virtual std::string getSummary();

	virtual void deallocateBuffers();

	virtual void save(fs::path const& folder);

	virtual void load(fs::path const& folder);

	virtual void setMaxWeightInVolume(float new_weight);

	virtual void setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius);

	virtual void setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose);

	virtual void getNonzeroVolumePointCloud(MeshVertexVector & vertex_list);

    virtual bool loopClosure(Frame & frame);

    virtual void prepareForRenderCurrent();

	virtual void refreshUpdateInterface();

	///////////////
	// These are new functions to this class:

    virtual void prepareForRenderLoopClosureAllOld();
    virtual void prepareForRenderLoopClosureTargetKeyframe(int keyframe_index);


	virtual void generateSingleMeshAndValidity(
		MeshPtr & mesh_result,
		boost::shared_ptr<std::vector<bool> > & vertex_validity);

	virtual void generateSingleMesh(MeshPtr & mesh_result);

	// these are old (get each grid separately)
	void generateAllMeshes(std::vector<MeshVertexVectorPtr> & vertex_list_list, std::vector<TriangleVectorPtr> & triangle_list_list);

	void generateAllMeshesAndValidity(std::vector<MeshVertexVectorPtr> & vertex_list_list, 
		std::vector<TriangleVectorPtr> & triangle_list_list,
		std::vector<boost::shared_ptr<std::vector<bool> > > & vertex_validity_list_list,
		std::vector<boost::shared_ptr<std::vector<bool> > > & triangle_validity_list_list);

	void generateMesh(
		std::vector<int> const& grid_index_list,
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list);

	void generateMeshAndValidity(
		std::vector<int> const& grid_index_list,
		MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list,
		std::vector<bool> & vertex_validity_list,
		std::vector<bool> & triangle_validity_list);

	void generateMeshList(
		std::vector<int> const& grid_index_list,
		std::vector<MeshVertexVectorPtr> & vertex_list_list, 
		std::vector<TriangleVectorPtr> & triangle_list_list,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);

	void generateMeshAndValidityList(
		std::vector<int> const& grid_index_list,
		std::vector<MeshVertexVectorPtr> & vertex_list_list, 
		std::vector<TriangleVectorPtr> & triangle_list_list,
		std::vector<boost::shared_ptr<std::vector<bool> > > & vertex_validity_list_list,
		std::vector<boost::shared_ptr<std::vector<bool> > > & triangle_validity_list_list,
		std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);

	// external pose list is for making a pose graph
	void getPoseExternalList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);

	// only internal TSDF poses (could do external * tsdf for example)
	void getPoseTSDFList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);

	// pose_original list (original value of pose_external)
	void getPoseOriginalList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);

	// pose_external * pose_tsdf
	void getPoseExternalTSDFList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list);


	void generateMeshForActiveVolumes(MeshVertexVector & vertex_list, TriangleVector & triangle_list);
	void generateMeshAndValidityForActiveVolumes(MeshVertexVector & vertex_list, 
		TriangleVector & triangle_list, 
		std::vector<bool> & vertex_validity_list,
		std::vector<bool> & triangle_validity_list);

	void generateAllBoundingMeshes(
		std::vector<MeshVertexVectorPtr> & vertex_list_list, 
		std::vector<TriangleVectorPtr> & triangle_list_list);

	void generateAllBoundingLines(std::vector<MeshVertexVectorPtr> & vertex_list_list);

	// in identity pose??  matching mesh?
	void generateBoundingLinesForGrids(std::vector<int> const& grid_indices, std::vector<MeshVertexVectorPtr> & vertex_list_list);

	// more useful to do this..consider for meshes...or single meshes
	void generateBoundingLinesForGrids(std::vector<int> const& grid_indices, MeshVertexVector & vertex_list);

	void generateAllGridMeshes(std::vector<MeshVertexVectorPtr> & vertex_list_list);

	////////////
	// activation functions:



	void activateVolumesBasedOnAge(int min_age, int max_age);

	void activateVolumesBasedOnMerged(bool merged_value);

	void setAllVolumesActive(bool value);

	void setAllNonMergedVolumesActive(bool value);

	void setVolumesActive(std::vector<int> const& volumes, bool value);

	void setNonMergedVolumesActive(std::vector<int> const& volumes, bool value);
	

	////////////////
	void getVolumesRenderedLastCall(std::vector<int> & volumes_rendered);

	void getVolumesUpdatedLastCall(std::vector<int> & volumes_updated);

	//////////////
	Eigen::Affine3f getVolumePoseExternal(int volume);

	void setVolumePoseExternal(int volume, Eigen::Affine3f const& pose);

	/////////////////////
	Eigen::Array3i getGridCell(Eigen::Array3f const& p) const;

	Eigen::Affine3f getPoseForGridCell(Eigen::Array3i const& grid_cell) const;


	// returns volumes created so pose graph may be updated
	std::vector<int> mergeVolumeIntoActive(int volume, bool set_merged);
	std::vector<int> mergeVolumeIntoActive(ModelGrid::GridStruct & grid_to_merge, bool set_merged);

	// merge all active volumes from other into this model_grid
	std::vector<int> mergeOtherModelGridActiveIntoActive(ModelGrid & other);

	//////////////
	int getVolumeCount();

	void getActiveStatus(std::vector<bool> & result);

	void getMergedStatus(std::vector<bool> & result);

	// goes with the above to get a list of indices
	// this should be a general function, fool!
	void boolsToIndices(std::vector<bool> const& include, std::vector<int> & result);


	// debug render artifacts
	std::vector<cv::Mat> getDebugRenderImages();

	cv::Vec3b getColorCV(int segment_id);

	Eigen::Vector4ub getColorEigen(int segment_id);

	cv::Mat getColorSegmentMat(const cv::Mat & mat_int);

	void debugCheckOverlap();


protected:
	//////////////
	// functions
	
	// from constructor (just this class)
	void resetThisClass();

	OpenCLTSDFPtr createTSDF( Eigen::Array3i const& cell_counts );
	void appendNewGridStruct(Eigen::Affine3f const& pose_external, Eigen::Affine3f const& post_tsdf, Eigen::Array3i const& cell_counts);
	void appendNewGridStruct(Eigen::Affine3f const& pose); // calls the other with param-based cell counts and identity pose_tsdf
	int appendNewGridCellIfNeeded(Eigen::Array3i const& grid_cell);
	void updateLastOperation(int index);

    uint64_t getBytesGPUExpected() const;
    uint64_t getBytesGPUActual() const;
    uint64_t getBytesRAM() const;
    uint64_t getBytesMeshCache() const;
    size_t getCountByState(TSDFState state) const;

	void deallocateAllVolumes();
	void saveToDiskAllVolumes();
	int deallocateUntilUnderMB(int max_mb);
	int saveToDiskUntilUnderMB(int max_mb);

	void getActiveGridsInFrustum(ParamsCamera const& params_camera, Eigen::Affine3f const& pose, std::vector<int> & result);

	fs::path getSaveFolderGrid(fs::path const& folder, int id);
	fs::path getSaveFolderKeyframe(fs::path const& folder, int id);

	void deallocateAsNeeded(int & sum_deallocated, int & sum_saved);
	void deallocateAsNeeded();

	void addNewGridCellsFromCloud(Frame const& frame, Eigen::Affine3f const& pose);
	void addNewGridCellsFromFrustum(Eigen::Affine3f const& pose);
	void addNewCellsGridFree(Frame const& frame, Eigen::Affine3f const& pose, std::vector<int> const& existing_grids_in_frustum);

	void getBBFromList(std::vector<Eigen::Vector3f> const& points, Eigen::Array3f & min_point, Eigen::Array3f & max_point);

    size_t getKeyframeMemoryBytes();

	/////
	// loop closure

	// this now just updates the keyframe (and keyframe_to_keyframe graph)
	// must call addEdgesToKeyframeGraphForVolumes before (and after if true) for grid style operation
	bool updateKeyframe(Frame& frame);

	// this uses grids_updated_last_frame_ to update keyframe_edges_ to volumes before (and after if new)
	bool UpdateKeyframeAndVolumeGraph(Frame & frame);

	void getCameraAndVolumeGraphDistances(int camera_index, std::map<int,int> & camera_distances, std::map<int,int> & volume_distances);
	static void getVertexDistancesSimpleGraph(std::vector<EdgeStruct> const& edge_struct_list, int index_start, std::map<int,int> & vertex_distances);
	static void getVerticesWithinDistanceSimpleGraph(std::vector<EdgeStruct> const& edge_struct_list, int index_start, int max_distance, std::map<int,int> & vertex_distances);

	void activateAge(bool loop_closure);
	void activateFullGraph(bool loop_closure);
    void activateKeyframeGraph(int target_keyframe, bool invert_selection);

	void addEdgesToKeyframeGraphForVolumes(int keyframe_index, std::vector<int> const& volumes);
	void addEdgesToKeyframeGraphForVolumes(int keyframe_index, Eigen::Affine3f const& keyframe_pose, std::vector<int> const& volumes);

	bool addEdgeToPoseGraph(G2OPoseGraph & pose_graph,
		std::map<int,int> const& camera_to_vertex_map, // or keyframe...it's just from edge indices to vertices
		std::map<int,int> const& volume_to_vertex_map,
		EdgeStruct const& edge);

	// virtual so that moving volume (or others?) can do additional stuff
	virtual void createG2OPoseGraphKeyframes(
		G2OPoseGraph & pose_graph,
		std::map<int, int> & keyframe_to_vertex_map,
		std::map<int, int> & volume_to_vertex_map);

	virtual void optimizeG2OPoseGraphKeyframes(
		G2OPoseGraph & pose_graph,
		std::map<int, int> const& keyframe_to_vertex_map,
		std::map<int, int> const& volume_to_vertex_map,
		int iterations);


	/////// end loop closure stuff?


    bool loopClosureOld(Frame & frame);
    bool loopClosureWithPlaceRecognition(Frame & frame);


    ///////////////
    // members

    // debug render
    std::vector<cv::Mat> debug_render_images_;

    // will be filled automatically as needed
    std::map<int, cv::Vec3b> segment_color_map_;

#if 0
    // need to regenerate
    // updated with updateDeallocateOrder():
	std::vector<std::pair<int, int> > deallocate_order_;
#endif

	boost::shared_ptr<FeatureMatching> features_ptr_;

    boost::shared_ptr<DBOWPlaceRecognition> dbow_place_recognition_ptr_;

	// need to save
	std::vector<GridStructPtr> grid_list_;
	GridToListMapT grid_to_list_map_;
	GridListToGridCellT grid_list_to_grid_cell_map_; // debug only?

	std::vector<int> grids_rendered_last_call_;
	std::vector<int> grids_updated_last_call_;

	// loop closure stuff (moved from volume_modeler)
	std::vector<KeyframeStructPtr> keyframe_list_;
	int current_keyframe_; // index into keyframe_list_ (so keyframe not always last keyframe)

	// index_0 is keyframe, index_1 is volume:
	std::vector<EdgeStruct> keyframe_edge_list_; // now holds keyframe index instead of camera index (finally!)

	// in this list, we record direct camera-camera edges
	std::vector<EdgeStruct> keyframe_to_keyframe_edges_;

	// record volume to keyframe links (needed by subclasses maybe?)
	std::map<int, std::set<int> > volume_to_keyframe_;

    OrderingContainer ordering_container_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

namespace boost {
	namespace serialization {
		template<class Archive>
		void serialize(Archive & ar, boost::tuple<int,int,int> & t, const unsigned int version)
		{
			ar & t.get<0>();
			ar & t.get<1>();
			ar & t.get<2>();
		}
	} 
} // namespaces
