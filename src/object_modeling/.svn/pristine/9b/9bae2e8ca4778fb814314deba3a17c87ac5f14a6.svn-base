#include "model_grid.h"

// for parts of load and save
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>

// for depth first graph search
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_utility.hpp>

#include "EigenUtilities.h"

#include "frustum.h"
#include "image_to_cloud.h"
#include "opencv_utilities.h"

#include "MarchingCubesManyVolumes.h"
#include "MeshUtilities.h"
#include "util.h"

#include "KernelNormalsToShadedImage.h"
#include "KernelNormalsToColorImage.h"

#include "KernelDepthImageToPoints.h"
#include "KernelTransformPoints.h"
#include "KernelSetInvalidPointsTrue.h"
#include "KernelSetUChar.h"


ModelGrid::ModelGrid(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers)
    : ModelBase(all_kernels, params, alignment_ptr, render_buffers)
{
    // make sure segment 0 is black?
    segment_color_map_[0] = cv::Vec3b(0,0,0);

    // could take this as argument
    // especially if constructor is expensive....or the class is statefull (neither true?)
    features_ptr_.reset(new FeatureMatching(params_.camera, params_.features));

    if (params_.loop_closure.use_dbow_place_recognition) {
        dbow_place_recognition_ptr_.reset(new DBOWPlaceRecognition(params_));
    }

    resetThisClass();
}

ModelGrid::ModelGrid(ModelGrid const& other)
    : ModelBase(other),
    debug_render_images_(other.debug_render_images_),
    segment_color_map_(other.segment_color_map_),
    features_ptr_(other.features_ptr_),
    grid_to_list_map_(other.grid_to_list_map_),
    grid_list_to_grid_cell_map_(other.grid_list_to_grid_cell_map_),
    grids_rendered_last_call_(other.grids_rendered_last_call_),
    grids_updated_last_call_(other.grids_updated_last_call_),
    current_keyframe_(other.current_keyframe_),
    keyframe_edge_list_(other.keyframe_edge_list_),
    keyframe_to_keyframe_edges_(other.keyframe_to_keyframe_edges_),
    volume_to_keyframe_(other.volume_to_keyframe_),
    ordering_container_(other.ordering_container_)
{
    grid_list_.resize(other.grid_list_.size());
    for (size_t i = 0; i < other.grid_list_.size(); ++i) {
        grid_list_[i]->tsdf_ptr.reset(other.grid_list_[i]->tsdf_ptr->clone());
    }

    keyframe_list_.resize(other.keyframe_list_.size());
    for (size_t i = 0; i < other.keyframe_list_.size(); ++i) {
        keyframe_list_[i].reset(new KeyframeStruct(*other.keyframe_list_[i]));
    }
}

ModelGrid* ModelGrid::clone()
{
    return new ModelGrid(*this);
}

void ModelGrid::resetThisClass()
{
    debug_render_images_.clear();
    grid_list_.clear();
    grid_to_list_map_.clear();
    grid_list_to_grid_cell_map_.clear();
    grids_rendered_last_call_.clear();
    grids_updated_last_call_.clear();
    ordering_container_.clear();

    keyframe_list_.clear();
    current_keyframe_ = -1;
    keyframe_edge_list_.clear();
    keyframe_to_keyframe_edges_.clear();
    volume_to_keyframe_.clear();
}

void ModelGrid::reset()
{
    ::ModelBase::reset();
    resetThisClass();
}

int ModelGrid::getVolumeCount()
{
    return grid_list_.size();
}

void ModelGrid::getActiveGridsInFrustum(ParamsCamera const& params_camera, Eigen::Affine3f const& pose, std::vector<int> & result)
{
    result.clear();
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_cell = *grid_list_[i];
        if (!grid_cell.active) continue;

        // frustum check
        Eigen::Array3f bb_min, bb_max;
        grid_cell.tsdf_ptr->getAABB(bb_min, bb_max);
        Eigen::Affine3f grid_cell_pose_world = pose * grid_cell.getExternalTSDFPose();
        Frustum frustum_in_tsdf_space(params_camera, grid_cell_pose_world.inverse());

        // throw in a visual frustum check (remove)
        //cout << "remove this update_interface_" << endl;
        if (false && update_interface_) {
            {
                // frustum in tsdf space
                std::vector<Eigen::Vector3f> frustum_lineset_points = frustum_in_tsdf_space.getLineSetPoints();
                MeshVertexVectorPtr frustum_vertices(new MeshVertexVector);
                for (int i = 0; i < (int)frustum_lineset_points.size(); ++i) {
                    frustum_vertices->push_back(MeshVertex());
                    MeshVertex & v = frustum_vertices->back();
                    v.p.head<3>() = frustum_lineset_points[i];
                    v.p[3] = 1;
                    v.c = Eigen::Vector4ub(200,200,200,0);
                }
                update_interface_->updateLines("debug frustum", frustum_vertices);
            }

            {
                // tsdf in tsdf space
                MeshVertexVectorPtr vertices(new MeshVertexVector);
                Eigen::Vector4ub color_eigen(200,200,200,0);
                grid_cell.tsdf_ptr->getBoundingLines(Eigen::Affine3f::Identity(), color_eigen, *vertices);
                update_interface_->updateLines("dbug tsdf frustm", vertices);
            }
            cout << "debug pause in get active grids" << endl;
            cv::waitKey();
        } // update interface debug

        if (!frustum_in_tsdf_space.doesAABBIntersect(bb_min.matrix(), bb_max.matrix())) {
            continue;
        }

        result.push_back(i);
    }
}

void ModelGrid::deallocateAsNeeded(int & sum_deallocated, int & sum_saved)
{
    if (params_.grid.max_mb_gpu > 0) {
        sum_deallocated += deallocateUntilUnderMB(params_.grid.max_mb_gpu);
    }
    if (params_.grid.max_mb_system > 0) {
        sum_saved += saveToDiskUntilUnderMB(params_.grid.max_mb_system);
    }
}

void ModelGrid::deallocateAsNeeded()
{
    int a, b;
    deallocateAsNeeded(a,b);
}

void ModelGrid::renderModel(
    const ParamsCamera & params_camera,
    const Eigen::Affine3f & model_pose,
    RenderBuffers & render_buffers)
{
    // all other models seem to do this here:
    render_buffers.setSize(params_camera.size.x(), params_camera.size.y());
    render_buffers.resetAllBuffers();

    getActiveGridsInFrustum(params_camera, model_pose, grids_rendered_last_call_);
    int number_deallocated = 0;
    int number_saved = 0;

    // check required size of grids_rendered_last_call_
    uint64_t byte_count = 0;
    BOOST_FOREACH(const int& index, grids_rendered_last_call_) {
        byte_count += grid_list_[index]->tsdf_ptr->getBytesGPUExpected();
    }
    cout << "MB for all grids in frustum: " << (byte_count >> 20) << endl;

    debug_render_images_.clear();

    if (params_.grid.new_render) {
        // "new" render
        for (std::vector<int>::iterator iter = grids_rendered_last_call_.begin(); iter != grids_rendered_last_call_.end(); ++iter) {
            updateLastOperation(*iter);
            deallocateAsNeeded(number_deallocated, number_saved);

            GridStruct & grid_cell = *grid_list_[*iter];

            Eigen::Affine3f grid_cell_pose_world = model_pose * grid_cell.getExternalTSDFPose();
            int this_mask_value = (*iter)+1;

            grid_cell.tsdf_ptr->renderPoints(grid_cell_pose_world,
                params_camera.focal, params_camera.center, params_camera.min_max_depth,
                true, this_mask_value, render_buffers);

            // NOTE: we could do colors only on final points (at the cost of iterating through all grids again)
            grid_cell.tsdf_ptr->renderColors(grid_cell_pose_world,
                params_camera.focal, params_camera.center, params_camera.min_max_depth,
                true, this_mask_value, render_buffers);
        }

        // now normals on the rendered points
        // could avoid repeated allocation (but we're just testing for now...)
        boost::shared_ptr<OpenCLNormals> opencl_normals_ptr(new OpenCLNormals(all_kernels_));
        //const float max_sigmas = params_.normals.max_depth_sigmas; // same?
        //const float smooth_iterations = params_.normals.smooth_iterations; // same?
        const float max_sigmas = 10; // 3 is default..
        const float smooth_iterations = 0;
        opencl_normals_ptr->computeNormalsWithBuffers(render_buffers.getImageBufferPoints(), max_sigmas, smooth_iterations, render_buffers.getImageBufferNormals());
    }
    else {
        // "old" render
        for (std::vector<int>::iterator iter = grids_rendered_last_call_.begin(); iter != grids_rendered_last_call_.end(); ++iter) {
            updateLastOperation(*iter);
            deallocateAsNeeded(number_deallocated, number_saved);

            GridStruct & grid_cell = *grid_list_[*iter];

            Eigen::Affine3f grid_cell_pose_world = model_pose * grid_cell.getExternalTSDFPose();
            int this_mask_value = (*iter)+1;
            grid_cell.tsdf_ptr->renderFrame(grid_cell_pose_world,
                params_camera.focal, params_camera.center, params_camera.min_max_depth,
                true, this_mask_value,
                render_buffers);

            // debug render
            if (params_.grid.debug_render) {
                cout << "Debug render of grid " << *iter << endl;

                cv::Mat render_color = render_buffers.getImageBufferColorImage().getMat();

                Eigen::Vector3f vector_to_light = Eigen::Vector3f(1,1,-1).normalized();
                ImageBuffer normals_image_buffer(all_kernels_->getCL());

                ImageBuffer normals_input_buffer = render_buffers.getImageBufferNormals();
                KernelNormalsToShadedImage _KernelNormalsToShadedImage(*all_kernels_);
                _KernelNormalsToShadedImage.runKernel(normals_input_buffer, normals_image_buffer, vector_to_light);
                cv::Mat render_normals = normals_image_buffer.getMat();

                KernelNormalsToColorImage _KernelNormalsToColorImage(*all_kernels_);
                _KernelNormalsToColorImage.runKernel(normals_input_buffer, normals_image_buffer);
                cv::Mat render_normals_color = normals_image_buffer.getMat();

                // get the depth image as well
                cv::Mat points = render_buffers.getImageBufferPoints().getMat();
                std::vector<cv::Mat> points_channels;
                cv::split(points, points_channels);
                cv::Mat depth_image = points_channels[2]; // has nans
                cv::Mat depth_mask_cpu = depth_image != depth_image;
                depth_image.setTo(0, depth_mask_cpu);
                float max_depth = *std::max_element(depth_image.begin<float>(), depth_image.end<float>());
                //cout << "max_depth: " << max_depth << endl;
                cv::Mat render_depth = floatC1toCharC4(depth_image / max_depth);


                //cv::Mat debug_image = create1x2(render_color, render_normals);
                std::vector<cv::Mat> debug_images;
                debug_images.push_back(render_color);
                debug_images.push_back(render_depth);
                debug_images.push_back(render_normals);
                debug_images.push_back(render_normals_color);
                cv::Mat debug_image = createMxN(2, 2, debug_images);
                debug_render_images_.push_back(debug_image);
            } 	// debug render
        }
    }
}

std::vector<cv::Mat> ModelGrid::getDebugRenderImages()
{
    return debug_render_images_;
}

void ModelGrid::addNewGridCellsFromCloud(Frame const& frame, Eigen::Affine3f const& pose)
{
    /*
    // todo: there's an opencl way to get the cloud now
    // This took 0.013 sec...
    boost::timer t_remove;
    Eigen::Matrix4Xf cloud;
    imageToCloud(params_.camera, frame.mat_depth, cloud);
    Eigen::Affine3f pose_inverse = pose.inverse();
    Eigen::Matrix4Xf cloud_in_grid = pose_inverse * cloud;
    cout << "t_remove: " << t_remove.elapsed() << endl;
    */

    // This alternative takes 0.004 sec.  Hooray.
    //	boost::timer t_remove;

    KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels_);
    KernelTransformPoints _KernelTransformPoints(*all_kernels_);

    ImageBuffer image_buffer_points(all_kernels_->getCL());
    _KernelDepthImageToPoints.runKernel(frame.image_buffer_depth, image_buffer_points, params_.camera.focal, params_.camera.center);
    ImageBuffer image_buffer_points_transformed(all_kernels_->getCL());
    _KernelTransformPoints.runKernel(image_buffer_points, image_buffer_points_transformed, pose.inverse());
    Eigen::Matrix4Xf cloud_in_grid = image_buffer_points_transformed.getMatrix4Xf();
    //	cout << "t_remove: " << t_remove.elapsed() << endl;

    // loop over cloud_in_grid, allocate new grid cells
    for (int i = 0; i < cloud_in_grid.cols(); ++i) {
        Eigen::Vector4f p = cloud_in_grid.col(i);
        if (p.z() != p.z()) continue;
        Eigen::Array3i grid_cell = getGridCell(p.head<3>());
        int existing_active_grid = appendNewGridCellIfNeeded(grid_cell);
    }
}

void ModelGrid::addNewGridCellsFromFrustum(Eigen::Affine3f const& pose)
{
    // need aabb for a grid cell
    Eigen::Array3i volume_cell_counts(params_.grid.grid_size, params_.grid.grid_size, params_.grid.grid_size);
    float volume_cell_size = params_.volume.cell_size;
    Eigen::Array3f bb_min, bb_max;
    getAABB(volume_cell_counts, volume_cell_size, bb_min, bb_max);

    // first get bounding box for frustum in the world
    Eigen::Affine3f camera_pose = pose.inverse();
    Frustum frustum_world(params_.camera, camera_pose);
    Eigen::Array3f min_point, max_point;
    getBBFromList(frustum_world.getPoints(), min_point, max_point);
    Eigen::Array3i min_grid_cell = getGridCell(min_point);
    Eigen::Array3i max_grid_cell = getGridCell(max_point);

    for (int i = min_grid_cell[0]; i <= max_grid_cell[0]; ++i) {
        for (int j = min_grid_cell[1]; j <= max_grid_cell[1]; ++j) {
            for (int k = min_grid_cell[2]; k <= max_grid_cell[2]; ++k) {
                Eigen::Array3i grid_cell(i,j,k);

                // check if in frustum
                Eigen::Affine3f grid_pose = getPoseForGridCell(grid_cell);
                Eigen::Affine3f grid_cell_pose_world = pose * grid_pose;
                Frustum frustum(params_.camera, grid_cell_pose_world.inverse());
                if (!frustum.doesAABBIntersect(bb_min.matrix(), bb_max.matrix())) continue;

                // add new grid cell if needed
                appendNewGridCellIfNeeded(grid_cell);
            }
        }
    }
}

void ModelGrid::addNewCellsGridFree(Frame const& frame, Eigen::Affine3f const& pose, std::vector<int> const& existing_grids_in_frustum)
{
    // dup from addNewGridCellsFromCloud
    KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels_);
    KernelTransformPoints _KernelTransformPoints(*all_kernels_);

    ImageBuffer image_buffer_points(all_kernels_->getCL());
    _KernelDepthImageToPoints.runKernel(frame.image_buffer_depth, image_buffer_points, params_.camera.focal, params_.camera.center);
    ImageBuffer image_buffer_points_transformed(all_kernels_->getCL());
    _KernelTransformPoints.runKernel(image_buffer_points, image_buffer_points_transformed, pose.inverse());

    boost::timer t_point_check;

    // opencl style
    ImageBuffer image_buffer_inside(all_kernels_->getCL());
    image_buffer_inside.resize(frame.image_buffer_depth.getRows(), frame.image_buffer_depth.getCols(), 1, CV_8U);
    KernelSetUChar _KernelSetUChar(*all_kernels_);
    _KernelSetUChar.runKernel(image_buffer_inside.getBuffer(), image_buffer_inside.getSizeBytes() / sizeof(unsigned char), 0);

    // set all invalid to true (aka "taken care of")
    KernelSetInvalidPointsTrue _KernelSetInvalidPointsTrue(*all_kernels_);
    const static bool resize_bool_image = false; // because we already sized and allocated
    _KernelSetInvalidPointsTrue.runKernel(frame.image_buffer_depth, image_buffer_inside, resize_bool_image);

    BOOST_FOREACH(int const& i, existing_grids_in_frustum) {
        GridStruct & grid_struct = *grid_list_[i];
        Eigen::Affine3f pose_for_check = (pose * grid_struct.getExternalTSDFPose()).inverse();
        grid_struct.tsdf_ptr->setPointsInsideBoxTrue(pose_for_check, frame.image_buffer_depth, image_buffer_inside);

        // debug
        if (false) {
            cv::Mat debug = image_buffer_inside.getMat();
            int sum = cv::sum(debug)[0];
            cv::imshow("debug", debug * 255);
            cout << "debug pause: " << sum << endl;
            cv::waitKey();
        }
    }

    cv::Mat mat_inside = image_buffer_inside.getMat();
    cv::Mat valid_not_in_grid = ~mat_inside;

    if (false) {
        cv::imshow("debug", valid_not_in_grid);
    }

    cout << "--- t_point_check: " << t_point_check.elapsed() << endl;


    // set seem to need this still
#if 1
    // DO WE NEED cloud_in_world?  probably, but not for this new opencl style check...
    // new idea...if we can lookup the grid for each point on the GPU, don't need to transfer these points to CPU here...
    // would be faster...
    Eigen::Matrix4Xf cloud_in_world = image_buffer_points_transformed.getMatrix4Xf();
#endif


    typedef std::map<boost::tuple<int,int,int>, int > LocalGridToIntMapT;
    LocalGridToIntMapT local_map;
    uint8_t* valid_not_in_grid_ptr = valid_not_in_grid.ptr(0);
    for (int i = 0; i < cloud_in_world.cols(); ++i) {
        if (!valid_not_in_grid_ptr[i]) continue;
        Eigen::Vector4f p = cloud_in_world.col(i);
        Eigen::Array3i grid_cell = getGridCell(p.head<3>());
        boost::tuple<int,int,int> key(grid_cell[0], grid_cell[1], grid_cell[2]);
        LocalGridToIntMapT::iterator find_iter = local_map.find(key);
        if (find_iter == local_map.end()) {
            // need a new grid cell
            Eigen::Affine3f grid_pose = getPoseForGridCell(grid_cell);
            appendNewGridStruct(grid_pose);
            int new_index = grid_list_.size() - 1;
            local_map[key] = new_index;
        }
    }

#if 0
    // quick check for sanity here
    // todo: remove
    if (!grid_to_list_map_.empty()) {
        std::string s = "!grid_to_list_map_.empty()";
        cout << s << endl;
        throw std::runtime_error(s);
    }
#endif
}

void ModelGrid::updateModel(
    Frame & frame,
    const Eigen::Affine3f & model_pose)
{
    ModelBase::updateModel(frame, model_pose);

    if (params_.grid.add_grids_in_frustum) {
        addNewGridCellsFromFrustum(model_pose);
    }
    else if (params_.grid.grid_free) {
        // need active grids to add more?
        std::vector<int> existing_grids_in_frustum;
        getActiveGridsInFrustum(params_.camera, model_pose, existing_grids_in_frustum);
        addNewCellsGridFree(frame, model_pose, existing_grids_in_frustum);
    }
    else {
        addNewGridCellsFromCloud(frame, model_pose);
    }

    // increase age of all
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        grid_list_[i]->age++;
    }

    // get all active grid cells in the frustum, not just those with points (for empty space)
    getActiveGridsInFrustum(params_.camera, model_pose, grids_updated_last_call_);

    // call add on active_grids_in_frustum
    int number_deallocated = 0;
    int number_saved = 0;
    if (params_.grid.debug_grid_motion) {
        cout << "Debug grid motion " << "---------------------" << endl;
    }

    for (std::vector<int>::iterator iter = grids_updated_last_call_.begin(); iter != grids_updated_last_call_.end(); ++iter) {
        updateLastOperation(*iter);
        deallocateAsNeeded(number_deallocated, number_saved);

        GridStruct & grid_struct = *grid_list_[*iter];
        grid_struct.tsdf_ptr->addFrame(model_pose * grid_struct.getExternalTSDFPose(), frame.image_buffer_depth, frame.image_buffer_color, frame.image_buffer_segments, frame.image_buffer_add_depth_weights, frame.image_buffer_add_color_weights, 0);
        grid_struct.age = 0;

        if (params_.grid.debug_grid_motion) {
            // get grid cell for grid
            boost::tuple<int,int,int> t = grid_list_to_grid_cell_map_[*iter];
            Eigen::Array3i grid_cell(t.get<0>(), t.get<1>(), t.get<2>());
            Eigen::Affine3f pose_for_grid_cell = getPoseForGridCell(grid_cell);
            Eigen::Affine3f current_grid_pose = grid_struct.getExternalTSDFPose();
            float a, d;
            EigenUtilities::getCameraPoseDifference(pose_for_grid_cell, current_grid_pose, a, d);
            cout << "Debug grid motion " << *iter << " : " << " A: " << a << " D: " << d << endl;
        }
    }

    bool changed_keyframe_ignore = UpdateKeyframeAndVolumeGraph(frame);
}

void ModelGrid::generateMesh(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
    activateVolumesBasedOnMerged(false);
    generateMeshForActiveVolumes(vertex_list, triangle_list);
}

void ModelGrid::generateMeshAndValidity(MeshVertexVector & vertex_list, TriangleVector & triangle_list, std::vector<bool> & vertex_validity, std::vector<bool> & triangle_validity)
{
    activateVolumesBasedOnMerged(false);
    generateMeshAndValidityForActiveVolumes(vertex_list, triangle_list, vertex_validity, triangle_validity);
}

void ModelGrid::generateAllMeshes(std::vector<std::pair<std::string, MeshPtr> > & names_and_meshes)
{
    boost::timer t_1;
    ModelBase::generateAllMeshes(names_and_meshes);
    cout << "TIME ModelBase::generateAllMeshes: " << t_1.elapsed() << endl;

    if (!params_.grid.skip_single_mesh) {
        boost::timer t_2;
        MeshPtr mesh(new Mesh);
        generateSingleMesh(mesh);
        names_and_meshes.push_back(std::make_pair("mesh_single", mesh));
        cout << "TIME generateSingleMesh: " << t_2.elapsed() << endl;
    }

    // temporarily turn off these "wasteful" meshes
#if 0
    // also the "full" mesh
    {
        MeshPtr mesh_ptr (new Mesh);
        std::vector<bool> vertex_validity;
        std::vector<bool> triangle_validity;
        model_ptr_->generateMeshAndValidity(mesh_ptr->vertices, mesh_ptr->triangles, vertex_validity, triangle_validity);
        success = success && saveMesh(mesh_ptr->vertices, mesh_ptr->triangles, save_folder / "mesh_all.ply");
    }

    // if grid, also mesh in original poses
    ModelGrid* model_grid = dynamic_cast<ModelGrid*>(model_ptr_.get());
    if (model_grid) {
        // non-merged volumes, original poses
        // maybe make this a function in model_grid?
        {
            std::vector<MeshVertexVectorPtr> vertex_list_list;
            std::vector<TriangleVectorPtr> triangle_list_list;
            model_grid->generateAllMeshes(vertex_list_list, triangle_list_list);

            // get merged status
            std::vector<bool> merged_status;
            model_grid->getMergedStatus(merged_status);

            // get original external pose and tsdf pose
            std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_original;
            model_grid->getPoseOriginalList(pose_original);

            std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_tsdf;
            model_grid->getPoseTSDFList(pose_tsdf);

            // finally, assemble result
            MeshPtr mesh_ptr (new Mesh);
            for (int i = 0; i < vertex_list_list.size(); ++i) {
                if (merged_status[i]) continue;

                Eigen::Affine3f pose = (*pose_original[i]) * (*pose_tsdf[i]);
                OpenCLTSDF::appendMesh(mesh_ptr->vertices, mesh_ptr->triangles, *vertex_list_list[i], *triangle_list_list[i], pose);
            }

            // finally, can save
            success = success && saveMesh(mesh_ptr->vertices, mesh_ptr->triangles, save_folder / "mesh_pose_original.ply");
        }
    }
#endif

}

void ModelGrid::generateMeshForActiveVolumes(MeshVertexVector & vertex_list, TriangleVector & triangle_list)
{
    std::vector<bool> active_bool;
    getActiveStatus(active_bool);
    std::vector<int> active_int;
    boolsToIndices(active_bool, active_int);
    generateMesh(active_int, vertex_list, triangle_list);
}

void ModelGrid::generateMeshAndValidityForActiveVolumes(MeshVertexVector & vertex_list, 
    TriangleVector & triangle_list,
    std::vector<bool> & vertex_validity_list,
    std::vector<bool> & triangle_validity_list)
{
    std::vector<bool> active_bool;
    getActiveStatus(active_bool);
    std::vector<int> active_int;
    boolsToIndices(active_bool, active_int);
    generateMeshAndValidity(active_int, vertex_list, triangle_list, vertex_validity_list, triangle_validity_list);
}

void ModelGrid::generateMesh(
    std::vector<int> const& grid_index_list,
    MeshVertexVector & vertex_list,
    TriangleVector & triangle_list)
{
    vertex_list.clear();
    triangle_list.clear();

    std::vector<MeshVertexVectorPtr> vertex_list_list;
    std::vector<TriangleVectorPtr> triangle_list_list;
    std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list;
    generateMeshList(grid_index_list, vertex_list_list, triangle_list_list, pose_list);

    for (size_t i = 0; i < vertex_list_list.size(); ++i) {
        MeshUtilities::transformMeshVertices(*pose_list[i], *vertex_list_list[i]);
        MeshUtilities::appendMesh(vertex_list, triangle_list, *vertex_list_list[i], *triangle_list_list[i]);
    }
}

void ModelGrid::generateMeshAndValidity(
    std::vector<int> const& grid_index_list,
    MeshVertexVector & vertex_list,
    TriangleVector & triangle_list,
    std::vector<bool> & vertex_validity_list,
    std::vector<bool> & triangle_validity_list)
{
    vertex_list.clear();
    triangle_list.clear();
    vertex_validity_list.clear();
    triangle_validity_list.clear();

    std::vector<MeshVertexVectorPtr> vertex_list_list;
    std::vector<TriangleVectorPtr> triangle_list_list;
    std::vector<boost::shared_ptr<std::vector<bool> > > vertex_validity_list_list;
    std::vector<boost::shared_ptr<std::vector<bool> > > triangle_validity_list_list;
    std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list;
    generateMeshAndValidityList(grid_index_list, vertex_list_list, triangle_list_list, vertex_validity_list_list, triangle_validity_list_list, pose_list);

    for (size_t i = 0; i < vertex_list_list.size(); ++i) {
        // no copy here (local list list);
        MeshUtilities::transformMeshVertices(*pose_list[i], *vertex_list_list[i]);
        MeshUtilities::appendMesh(vertex_list, triangle_list, *vertex_list_list[i], *triangle_list_list[i]);
        std::copy(vertex_validity_list_list[i]->begin(), vertex_validity_list_list[i]->end(), std::back_inserter(vertex_validity_list));
        std::copy(triangle_validity_list_list[i]->begin(), triangle_validity_list_list[i]->end(), std::back_inserter(triangle_validity_list));
    }
}

void ModelGrid::generateAllMeshes(std::vector<MeshVertexVectorPtr> & vertex_list_list, 
    std::vector<TriangleVectorPtr> & triangle_list_list)
{
    std::vector<bool> all_bool(grid_list_.size(), true);
    std::vector<int> all_int;
    boolsToIndices(all_bool, all_int);
    std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list;
    generateMeshList(all_int, vertex_list_list, triangle_list_list, pose_list);
}

void ModelGrid::generateAllMeshesAndValidity(std::vector<MeshVertexVectorPtr> & vertex_list_list, 
    std::vector<TriangleVectorPtr> & triangle_list_list,
    std::vector<boost::shared_ptr<std::vector<bool> > > & vertex_validity_list_list,
    std::vector<boost::shared_ptr<std::vector<bool> > > & triangle_validity_list_list)
{
    std::vector<bool> all_bool(grid_list_.size(), true);
    std::vector<int> all_int;
    boolsToIndices(all_bool, all_int);
    std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list;
    generateMeshAndValidityList(all_int, vertex_list_list, triangle_list_list, vertex_validity_list_list, triangle_validity_list_list, pose_list);
}

void ModelGrid::generateMeshList(
    std::vector<int> const& grid_index_list,
    std::vector<MeshVertexVectorPtr> & vertex_list_list,
    std::vector<TriangleVectorPtr> & triangle_list_list,
    std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    vertex_list_list.clear();
    triangle_list_list.clear();
    pose_list.clear();

    size_t progress_counter = 0;
    const size_t summary_frequency = 10;

    for (std::vector<int>::const_iterator iter = grid_index_list.begin(); iter != grid_index_list.end(); ++iter) {
        //updateLastOperation(*iter);
        deallocateAsNeeded();

        GridStruct & grid_struct = *grid_list_[*iter];

        vertex_list_list.push_back( MeshVertexVectorPtr(new MeshVertexVector) );
        triangle_list_list.push_back( TriangleVectorPtr(new TriangleVector) );

        grid_struct.tsdf_ptr->generateMesh(*vertex_list_list.back(), *triangle_list_list.back());

        cout << "Generated grid mesh " << progress_counter++ << "/" << grid_index_list.size() << endl;
        if (progress_counter % summary_frequency == 0) {
            cout << getSummary() << endl;
        }

        pose_list.push_back(boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_struct.getExternalTSDFPose())));
    }
}

void ModelGrid::generateMeshAndValidityList(
    std::vector<int> const& grid_index_list,
    std::vector<MeshVertexVectorPtr> & vertex_list_list,
    std::vector<TriangleVectorPtr> & triangle_list_list,
    std::vector<boost::shared_ptr<std::vector<bool> > > & vertex_validity_list_list,
    std::vector<boost::shared_ptr<std::vector<bool> > > & triangle_validity_list_list,
    std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    vertex_list_list.clear();
    triangle_list_list.clear();
    pose_list.clear();

    size_t progress_counter = 0;
    const size_t summary_frequency = 10;

    for (std::vector<int>::const_iterator iter = grid_index_list.begin(); iter != grid_index_list.end(); ++iter) {
        //updateLastOperation(*iter);
        deallocateAsNeeded();

        GridStruct & grid_struct = *grid_list_[*iter];

        vertex_list_list.push_back( MeshVertexVectorPtr(new MeshVertexVector) );
        triangle_list_list.push_back( TriangleVectorPtr(new TriangleVector) );
        vertex_validity_list_list.push_back( boost::shared_ptr<std::vector<bool> >(new std::vector<bool>()) );
        triangle_validity_list_list.push_back( boost::shared_ptr<std::vector<bool> >(new std::vector<bool>()) );

        grid_struct.tsdf_ptr->generateMeshAndValidity(*vertex_list_list.back(), *triangle_list_list.back(), *vertex_validity_list_list.back());
        MeshUtilities::getTriangleValidity(*vertex_list_list.back(), *triangle_list_list.back(), *vertex_validity_list_list.back(), *triangle_validity_list_list.back());

        cout << "Generated grid mesh " << progress_counter++ << "/" << grid_index_list.size() << endl;
        if (progress_counter % summary_frequency == 0) {
            cout << getSummary() << endl;
        }

        pose_list.push_back(boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_struct.getExternalTSDFPose())));
    }
}

// try to generate a single mesh...
void ModelGrid::generateSingleMeshAndValidity(
    MeshPtr & mesh_result,
    boost::shared_ptr<std::vector<bool> > & vertex_validity)
{
    std::vector<OpenCLTSDFPtr> tsdf_ptr_list;
    std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_ptr_list;

    //for (std::vector<int>::const_iterator iter = grid_index_list.begin(); iter != grid_index_list.end(); ++iter) {
    //	GridStruct & grid_struct = *grid_list_[*iter];
    for (std::vector<GridStructPtr>::iterator iter = grid_list_.begin(); iter != grid_list_.end(); ++iter) {
        GridStruct & grid_struct = **iter;
        if (!grid_struct.active) continue;

        tsdf_ptr_list.push_back(grid_struct.tsdf_ptr);
        pose_ptr_list.push_back(boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_struct.getExternalTSDFPose())));
    }

    MarchingCubesManyVolumes::generateMeshAndValidity(tsdf_ptr_list, pose_ptr_list, update_interface_, params_.grid.max_mb_gpu, *mesh_result, *vertex_validity);

    // do you fools need triangle validity too?
    // nope...this whole structure of mesh generation is crappy!!!
    //grid_struct.tsdf_ptr->getTriangleValidity(*vertex_list_list.back(), *triangle_list_list.back(), *vertex_validity_list_list.back(), verbose, *triangle_validity_list_list.back());
}

void ModelGrid::generateSingleMesh(MeshPtr & mesh_result)
{
    MeshPtr mesh_all_ptr(new Mesh);
    boost::shared_ptr<std::vector<bool> > vertex_validity_ptr(new std::vector<bool>);
    generateSingleMeshAndValidity(mesh_all_ptr, vertex_validity_ptr);
    MeshUtilities::extractValidVerticesAndTriangles(mesh_all_ptr->vertices, mesh_all_ptr->triangles, *vertex_validity_ptr, mesh_result->vertices, mesh_result->triangles);
}


void ModelGrid::getPoseExternalList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    pose_list.clear();
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        pose_list.push_back( boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_list_[i]->pose_external)) );
    }
}

void ModelGrid::getPoseTSDFList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    pose_list.clear();
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        pose_list.push_back(  boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_list_[i]->pose_tsdf))  );
    }
}

void ModelGrid::getPoseOriginalList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    pose_list.clear();
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        pose_list.push_back( boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_list_[i]->pose_original)) );
    }
}

void ModelGrid::getPoseExternalTSDFList(std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list)
{
    pose_list.clear();
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        pose_list.push_back(  boost::shared_ptr<Eigen::Affine3f>(new Eigen::Affine3f(grid_list_[i]->getExternalTSDFPose())) );
    }
}

cv::Vec3b ModelGrid::getColorCV(int segment_id)
{
    const static int min_value = 50;
    if (segment_color_map_.find(segment_id) == segment_color_map_.end()) {
        segment_color_map_[segment_id] = cv::Vec3b(min_value + rand() % (256 - min_value), min_value + rand() % (256 - min_value), min_value + rand() % (256 - min_value));
    }
    return segment_color_map_[segment_id];
}

Eigen::Vector4ub ModelGrid::getColorEigen(int segment_id)
{
    const static uint8_t alpha = 255;
    cv::Vec3b color = getColorCV(segment_id);
    return Eigen::Vector4ub (color[0], color[1], color[2], 255);
}

cv::Mat ModelGrid::getColorSegmentMat(const cv::Mat & mat_int)
{
    cv::Mat result (mat_int.size(), CV_8UC3);
    cv::MatConstIterator_<int> iter_in;
    cv::MatIterator_<cv::Vec3b> iter_out;
    for (iter_in = mat_int.begin<int>(), iter_out = result.begin<cv::Vec3b>();
        iter_in != mat_int.end<int>();
        ++iter_in, ++iter_out) {
            *iter_out = getColorCV(*iter_in);
    }
    return result;
}

void ModelGrid::generateAllBoundingMeshes(
    std::vector<MeshVertexVectorPtr> & vertex_list_list,
    std::vector<TriangleVectorPtr> & triangle_list_list)
{
    vertex_list_list.clear();
    triangle_list_list.clear();

    for (size_t i = 0; i < grid_list_.size(); ++i) {
        vertex_list_list.push_back( MeshVertexVectorPtr(new MeshVertexVector) );
        triangle_list_list.push_back( TriangleVectorPtr(new TriangleVector) );
        Eigen::Vector4ub color_eigen = getColorEigen(i+1);
        grid_list_[i]->tsdf_ptr->getBoundingMesh(Eigen::Affine3f::Identity(), color_eigen, *vertex_list_list.back(), *triangle_list_list.back());
    }
}

void ModelGrid::generateAllBoundingLines(std::vector<MeshVertexVectorPtr> & vertex_list_list)
{
    vertex_list_list.clear();

    for (size_t i = 0; i < grid_list_.size(); ++i) {
        vertex_list_list.push_back( MeshVertexVectorPtr(new MeshVertexVector) );
        Eigen::Vector4ub color_eigen = getColorEigen(i+1);
        grid_list_[i]->tsdf_ptr->getBoundingLines(Eigen::Affine3f::Identity(), color_eigen, *vertex_list_list.back());
    }
}

void ModelGrid::generateBoundingLinesForGrids(std::vector<int> const& grid_indices, std::vector<MeshVertexVectorPtr> & vertex_list_list)
{
    vertex_list_list.clear();

    for (std::vector<int>::const_iterator iter = grid_indices.begin(); iter != grid_indices.end(); ++iter) {
        vertex_list_list.push_back( MeshVertexVectorPtr(new MeshVertexVector) );
        Eigen::Vector4ub color_eigen = getColorEigen((*iter)+1);
        grid_list_[*iter]->tsdf_ptr->getBoundingLines(Eigen::Affine3f::Identity(), color_eigen, *vertex_list_list.back());
    }
}

void ModelGrid::generateBoundingLinesForGrids(std::vector<int> const& grid_indices, MeshVertexVector & vertex_list)
{
    vertex_list.clear();
    TriangleVector empty_triangles;

    std::vector<MeshVertexVectorPtr> vertex_list_list;
    generateBoundingLinesForGrids(grid_indices, vertex_list_list);
    for (size_t i = 0; i < vertex_list_list.size(); ++i) {
        MeshUtilities::transformMeshVertices(grid_list_[grid_indices[i]]->getExternalTSDFPose(), *vertex_list_list[i]);
        MeshUtilities::appendMesh(vertex_list, empty_triangles, *vertex_list_list[i], empty_triangles);
    }
}

void ModelGrid::generateAllGridMeshes(std::vector<MeshVertexVectorPtr> & vertex_list_list)
{
    // todo
    cout << "generateAllGridMeshes not implemented" << endl;

    vertex_list_list.clear();

    // start of an idea...but not needed...just need original pose, I think...
#if 0
    BOOST_FOREACH(GridToListMapT::value_type const& p, grid_to_list_map_) {
        std::vector<int> const& cells_for_spot = p.second;
        boost::tuple<int,int,int> const& grid_cell_boost = p.first;
        Eigen::Array3i grid_cell_eigen(grid_cell_boost.get<0>(), grid_cell_boost.get<1>(), grid_cell_boost.get<2>());
        BOOST_FOREACH(int const& v, cells_for_spot) {
            // not just active...all!
            MeshVertexVectorPtr ptr(new MeshVertexVector);
            // and more stuff...
        }
    }
#endif

    // reference
#if 0
    boost::tuple<int,int,int> key(grid_cell[0], grid_cell[1], grid_cell[2]);
    std::vector<int> & cells_for_spot = grid_to_list_map_[key];
    int existing_active_grid = -1;
    for (std::vector<int>::iterator c_iter = cells_for_spot.begin(); c_iter != cells_for_spot.end(); ++c_iter) {
        if (!grid_list_[*c_iter]->active) continue; // skip inactive
        existing_active_grid = *c_iter;
        break;
    }
    if (existing_active_grid < 0) {
        Eigen::Affine3f grid_pose = getPoseForGridCell(grid_cell);
        appendNewGridStruct(grid_pose);
        int new_index = grid_list_.size() - 1;
        cells_for_spot.push_back(new_index);
        existing_active_grid = new_index;
        grid_list_to_grid_cell_map_[new_index] = key; // debug?
    }
    return existing_active_grid;
#endif

}


void ModelGrid::getBuffersLists(
    std::vector<boost::shared_ptr<std::vector<float> > >& bufferDVectors,
    std::vector<boost::shared_ptr<std::vector<float> > > & bufferDWVectors,
    std::vector<boost::shared_ptr<std::vector<unsigned char> > > & bufferCVectors,
    std::vector<boost::shared_ptr<std::vector<float> > > & bufferCWVectors,
    std::vector<boost::shared_ptr<Eigen::Affine3f> > & pose_list,
    std::vector<boost::shared_ptr<Eigen::Array3i> > & cell_counts_list)
{
    bufferDVectors.clear();
    bufferDWVectors.clear();
    bufferCVectors.clear();
    bufferCWVectors.clear();
    pose_list.clear();
    cell_counts_list.clear();

    for (size_t i = 0; i < grid_list_.size(); ++i) {
        //updateLastOperation(i);
        deallocateAsNeeded();

        bufferDVectors.push_back(boost::make_shared<std::vector<float> >());
        bufferDWVectors.push_back(boost::make_shared<std::vector<float> >());
        bufferCVectors.push_back(boost::make_shared<std::vector<unsigned char> >());
        bufferCWVectors.push_back(boost::make_shared<std::vector<float> >());

        grid_list_[i]->tsdf_ptr->getAllBuffers(*bufferDVectors.back(), *bufferDWVectors.back(), *bufferCVectors.back(), *bufferCWVectors.back());

        cell_counts_list.push_back(boost::make_shared<Eigen::Array3i> ());
        *cell_counts_list.back() = grid_list_[i]->tsdf_ptr->getVolumeCellCounts();
    }

    getPoseExternalTSDFList(pose_list);
}

fs::path ModelGrid::getSaveFolderGrid(fs::path const& folder, int id)
{
    return folder / (boost::format("grid_%05d") % id).str();
}

fs::path ModelGrid::getSaveFolderKeyframe(fs::path const& folder, int id)
{
    return folder / (boost::format("keyframe_%05d") % id).str();
}


OpenCLTSDFPtr ModelGrid::createTSDF( Eigen::Array3i const& cell_counts )
{
    return OpenCLTSDFPtr (new OpenCLTSDF(all_kernels_,
        params_.camera.focal.x(), params_.camera.focal.y(),
        params_.camera.center.x(), params_.camera.center.y(),
        params_.camera.size.x(), params_.camera.size.y(),
        cell_counts[0], cell_counts[1], cell_counts[2], params_.volume.cell_size,
        params_.volume.max_weight_icp, params_.volume.max_weight_color, params_.volume.use_most_recent_color, params_.volume.min_truncation_distance, params_.grid.temp_folder));
}

void ModelGrid::appendNewGridStruct(Eigen::Affine3f const& pose_external, Eigen::Affine3f const& pose_tsdf, Eigen::Array3i const& cell_counts)
{
    // this is new...free space for this new volume
    deallocateAsNeeded();

    int index = grid_list_.size(); // note that if we start editing the grid list, this could cause problems

    // yeah..should probably make a constructor
    // but it needs params, etc...
    GridStructPtr result(new GridStruct);
    result->tsdf_ptr = createTSDF(cell_counts);
    result->pose_external = pose_external;
    result->pose_original = pose_external;
    result->pose_tsdf = pose_tsdf;
    result->age = 0;
    result->active = true;
    result->merged = false;

    grid_list_.push_back(result);

    updateLastOperation(index); // creation counts as an operation
}

void ModelGrid::appendNewGridStruct(Eigen::Affine3f const& pose)
{
    int cells_with_border = params_.grid.grid_size + 2 * params_.grid.border_size + 1; // added 1 "standard overlap"
    Eigen::Array3i cell_counts(cells_with_border, cells_with_border, cells_with_border);
    appendNewGridStruct(pose, Eigen::Affine3f::Identity(), cell_counts);
}

void ModelGrid::updateLastOperation(int index)
{
    ordering_container_.setIDToMax(index);
}

uint64_t ModelGrid::getBytesGPUExpected() const
{
    uint64_t result = 0;
    BOOST_FOREACH(GridStructPtr const& g, grid_list_) {
        result += g->tsdf_ptr->getBytesGPUExpected();
    }
    return result;
}

uint64_t ModelGrid::getBytesGPUActual() const
{
    uint64_t result = 0;
    BOOST_FOREACH(GridStructPtr const& g, grid_list_) {
        result += g->tsdf_ptr->getBytesGPUActual();
    }
    return result;
}

uint64_t ModelGrid::getBytesRAM() const
{
    uint64_t result = 0;
    BOOST_FOREACH(GridStructPtr const& g, grid_list_) {
        result += g->tsdf_ptr->getBytesRAM();
    }
    return result;
}

uint64_t ModelGrid::getBytesMeshCache() const
{
    uint64_t result = 0;
    BOOST_FOREACH(GridStructPtr const& g, grid_list_) {
        result += g->tsdf_ptr->getBytesMeshCache();
    }
    return result;
}

size_t ModelGrid::getCountByState(TSDFState state) const
{
    size_t result = 0;
    BOOST_FOREACH(GridStructPtr const& g, grid_list_) {
        if (g->tsdf_ptr->getTSDFState() == state) {
            result++;
        }
    }
    return result;
}


void ModelGrid::deallocateAllVolumes()
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        grid_list_[i]->tsdf_ptr->setTSDFState(TSDF_STATE_RAM);
    }
}

void ModelGrid::saveToDiskAllVolumes()
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        grid_list_[i]->tsdf_ptr->setTSDFState(TSDF_STATE_DISK);
    }
}

void ModelGrid::save(fs::path const& folder)
{
    ModelBase::save(folder);

    for (size_t i = 0; i < grid_list_.size(); ++i) {
		fs::path grid_folder = getSaveFolderGrid(folder, i);
        grid_list_[i]->tsdf_ptr->save(grid_folder); // this creates folder

        fs::path filename = grid_folder / "model_grid_info.txt";
        std::ofstream file (filename.string().c_str());
        file << EigenUtilities::transformToString(grid_list_[i]->pose_external) << endl;
        file << EigenUtilities::transformToString(grid_list_[i]->pose_original) << endl;
        file << EigenUtilities::transformToString(grid_list_[i]->pose_tsdf) << endl;
        file << grid_list_[i]->age << endl;
        file << grid_list_[i]->active << endl;
        file << grid_list_[i]->merged << endl;
    }

    {
        fs::path filename = folder / "grid_list_size.txt";
        std::ofstream file(filename.string().c_str());
        file << grid_list_.size() << endl;
    }

    {
        fs::path filename = folder / "grid_to_list_map.txt";
        std::ofstream file(filename.string().c_str());
        boost::archive::text_oarchive a(file);
        a & grid_to_list_map_;
    }

    {
        fs::path filename = folder / "grids_rendered_last_call.bin";
        EigenUtilities::writeVector(grids_rendered_last_call_, filename);
    }

    {
        fs::path filename = folder / "grids_updated_last_call.bin";
        EigenUtilities::writeVector(grids_updated_last_call_, filename);
    }

    // debug only?
    {
        fs::path filename = folder / "grid_list_to_grid_cell_map.txt";
        std::ofstream file(filename.string().c_str());
        boost::archive::text_oarchive a(file);
        a & grid_list_to_grid_cell_map_;
    }

    // keyframe_list
#if 0
    {
        fs::path filename = folder / "keyframe_list.yaml";
        cv::FileStorage fs(filename.string(), cv::FileStorage::WRITE);
        fs << "keyframe_list" << "[";
        for (int i = 0; i < (int)keyframe_list_.size(); ++i) {
            KeyframeStruct & key = *keyframe_list_[i];
            fs << "{";
            fs << "mat_color_bgra" << key.mat_color_bgra;
            fs << "mat_depth" << key.mat_depth;
            fs << "keypoints" << *key.keypoints;
            fs << "camera_index" << key.camera_index;
            std::vector<int> volumes_vec(key.volumes.begin(), key.volumes.end());
            fs << "volumes" << volumes_vec;
            fs << "}";
        }
        fs << "]";
    }
#endif
	{
        fs::path filename = folder / "keyframe_list_size.txt";
        std::ofstream file(filename.string().c_str());
        file << keyframe_list_.size() << endl;
    }
	{
		for (int i = 0; i < (int)keyframe_list_.size(); ++i) {
			fs::path folder_keyframe = getSaveFolderKeyframe(folder, i);
            KeyframeStruct & key = *keyframe_list_[i];
			key.save(folder_keyframe);
        }
	}


    // current_keyframe_
    {
        fs::path filename = folder / "current_keyframe_.txt";
        std::fstream file(filename.string().c_str(), std::ios::out);
        file << current_keyframe_ << endl;
    }

    // just all edge lists with archives:
    {
        fs::path filename = folder / "edge_lists_archive.txt";
        std::fstream file(filename.string().c_str(), std::ios::out);
        boost::archive::text_oarchive a(file);

        a & keyframe_edge_list_;
        a & keyframe_to_keyframe_edges_;
        a & volume_to_keyframe_;
    }

    // deallocate_priority_queue_ archive
    // should make all archives in one?
    {
        fs::path filename = folder / "ordering_container.txt";
        std::fstream file(filename.string().c_str(), std::ios::out);
        boost::archive::text_oarchive a(file);

        a & ordering_container_;
    }

}

void ModelGrid::load(fs::path const& folder)
{
    ModelBase::load(folder);

    size_t grid_list_size = 0;
    {
        fs::path filename = folder / "grid_list_size.txt";
        std::ifstream file(filename.string().c_str());
        file >> grid_list_size;
    }

    grid_list_.clear();
    for (size_t i = 0; i < grid_list_size; ++i) {
        appendNewGridStruct(Eigen::Affine3f::Identity());

        fs::path grid_folder = getSaveFolderGrid(folder, i);
        //grid_list_[i]->tsdf_ptr->load(grid_folder);
        grid_list_[i]->tsdf_ptr->loadLazy(grid_folder);

        fs::path filename = grid_folder / "model_grid_info.txt";
        std::ifstream file(filename.string().c_str());
        std::string line;
        std::getline(file, line);
        grid_list_[i]->pose_external = EigenUtilities::stringToTransform(line);
        std::getline(file, line);
        grid_list_[i]->pose_original = EigenUtilities::stringToTransform(line);
        std::getline(file, line);
        grid_list_[i]->pose_tsdf = EigenUtilities::stringToTransform(line);
        file >> grid_list_[i]->age;
        file >> grid_list_[i]->active;
        file >> grid_list_[i]->merged;
    }

    {
        fs::path filename = folder / "grid_to_list_map.txt";
        std::ifstream file(filename.string().c_str());
        boost::archive::text_iarchive a(file);
        a & grid_to_list_map_;
    }

    {
        fs::path filename = folder / "grids_rendered_last_call.bin";
        EigenUtilities::readVector(grids_rendered_last_call_, filename);
    }

    {
        fs::path filename = folder / "grids_updated_last_call.bin";
        EigenUtilities::readVector(grids_updated_last_call_, filename);
    }

    // debug only?
    {
        fs::path filename = folder / "grid_list_to_grid_cell_map.txt";
        if (fs::exists(filename)) {
            std::ifstream file(filename.string().c_str());
            boost::archive::text_iarchive a(file);
            a & grid_list_to_grid_cell_map_;
        }
        else {
            cout << "Warning: no file: " << filename << endl;
        }
    }

    // keyframe_list
#if 0
    {
        fs::path filename = folder / "keyframe_list.yaml";
        cv::FileStorage fs(filename.string(), cv::FileStorage::READ);
        cv::FileNode node = fs["keyframe_list"];
        keyframe_list_.clear();
        for (cv::FileNodeIterator iter = node.begin(); iter != node.end(); ++iter) {
            keyframe_list_.push_back(KeyframeStructPtr(new KeyframeStruct));
            KeyframeStruct & key = *keyframe_list_.back();
            cv::read((*iter)["mat_color_bgra"], key.mat_color_bgra);
            cv::read((*iter)["mat_depth"], key.mat_depth);
            key.keypoints.reset(new Keypoints);
            cv::read((*iter)["keypoints"], *key.keypoints);
            // what is the difference???
            //cv::read((*iter)["camera_index"], key.camera_index);
            (*iter)["camera_index"] >> key.camera_index;
            std::vector<int> volumes_vec;
            cv::read((*iter)["volumes"], volumes_vec);
            key.volumes = std::set<int>(volumes_vec.begin(), volumes_vec.end());
        }
    }
#endif
	size_t keyframe_list_size = 0;
	{
        fs::path filename = folder / "keyframe_list_size.txt";
        std::ifstream file(filename.string().c_str());
		file >> keyframe_list_size;
    }
	{
		keyframe_list_.clear();
		for (int i = 0; i < (int)keyframe_list_size; ++i) {
			fs::path folder_keyframe = getSaveFolderKeyframe(folder, i);
			keyframe_list_.push_back(KeyframeStructPtr(new KeyframeStruct));
            KeyframeStruct & key = *keyframe_list_.back();
			key.load(folder_keyframe);
        }
	}

    // current_keyframe_
    {
        fs::path filename = folder / "current_keyframe_.txt";
        std::fstream file(filename.string().c_str(), std::ios::in);
        file >> current_keyframe_;
    }

    // just all edge lists with archives:
    {
        fs::path filename = folder / "edge_lists_archive.txt";
        std::fstream file(filename.string().c_str(), std::ios::in);
        boost::archive::text_iarchive a(file);

        a & keyframe_edge_list_;
        a & keyframe_to_keyframe_edges_;
        a & volume_to_keyframe_;
    }

    // deallocate_priority_queue_ archive
    // should make all archives in one?
    {
        fs::path filename = folder / "ordering_container.txt";
        std::fstream file(filename.string().c_str(), std::ios::out);
        boost::archive::text_oarchive a(file);

        a & ordering_container_;
    }

    // repopulate the dbow place recognition with existing keyframes
    if (params_.loop_closure.use_dbow_place_recognition) {
        if (!dbow_place_recognition_ptr_) {
            cout << "Somehow !dbow_place_recognition_ptr_" << endl;
            throw std::runtime_error("load dbow_place_recognition_ptr_");
        }
        BOOST_FOREACH(KeyframeStructPtr const& k, keyframe_list_) {
            std::vector<unsigned int> ignore_results;
            dbow_place_recognition_ptr_->addAndDetectBGRA(k->mat_color_bgra, ignore_results);
        }
    }
}

int ModelGrid::deallocateUntilUnderMB(int max_mb)
{
    int number_deallocated = 0;
#if 0
    for (size_t pair = 0; pair < deallocate_order_.size(); ++pair) {
        int i = deallocate_order_[pair].second;
        if (!grid_list_[i]->tsdf_ptr->buffersAreAllocated()) continue;
        int required_mb, allocated_mb, in_memory_mb;
        getMBSizes(required_mb, allocated_mb, in_memory_mb);
        if (allocated_mb < max_mb) break;

        cout << "deallocating grid: " << i << " with last_operation: " << grid_list_[i]->last_operation << endl;

        grid_list_[i]->tsdf_ptr->deallocateVolumeBuffers();
        ++number_deallocated;
    }
#endif
    std::map<int,int>::const_iterator ordering_begin, ordering_end;
    ordering_container_.getIterators(ordering_begin, ordering_end);
    for (std::map<int,int>::const_iterator iter = ordering_begin; iter != ordering_end; ++iter) {
        std::pair<int,int> priority_and_id = *iter;
        int grid_id = priority_and_id.second;
        if (grid_id < 0) {
            cout << "deallocate grid_id negative!" << endl;
            break;
        }
        if (grid_list_[priority_and_id.second]->tsdf_ptr->getTSDFState() != TSDF_STATE_GPU) continue;

        // instead of total every time you could reduce a running total
        int mb_gpu = getBytesGPUActual() >> 20;
        if (mb_gpu < max_mb) break;

        grid_list_[grid_id]->tsdf_ptr->setTSDFState(TSDF_STATE_RAM);
        ++number_deallocated;
    }

    return number_deallocated;
}

int ModelGrid::saveToDiskUntilUnderMB(int max_mb)
{
    int number_deallocated = 0;
#if 0
    for (size_t pair = 0; pair < deallocate_order_.size(); ++pair) {
        int i = deallocate_order_[pair].second;
        if (!grid_list_[i]->tsdf_ptr->buffersAreInMemory()) continue;
        int required_mb, allocated_mb, in_memory_mb;
        getMBSizes(required_mb, allocated_mb, in_memory_mb);
        if (in_memory_mb < max_mb) break;
        grid_list_[i]->tsdf_ptr->saveToFolderTempAndClearMemory();
        ++number_deallocated;
    }
#endif
    std::map<int,int>::const_iterator ordering_begin, ordering_end;
    ordering_container_.getIterators(ordering_begin, ordering_end);
    for (std::map<int,int>::const_iterator iter = ordering_begin; iter != ordering_end; ++iter) {
        std::pair<int,int> priority_and_id = *iter;
        int grid_id = priority_and_id.second;
        if (grid_id < 0) {
            cout << "deallocate grid_id negative!" << endl;
            break;
        }

        int mb_ram = getBytesRAM() >> 20;
        if (mb_ram < max_mb) break;

        grid_list_[grid_id]->tsdf_ptr->setTSDFState(TSDF_STATE_DISK);
        ++number_deallocated;
    }

    return number_deallocated;
}

std::string ModelGrid::getSummary()
{
    std::string result;

    int mb_gpu_expected = getBytesGPUExpected() >> 20;
    int mb_gpu_actual = getBytesGPUActual() >> 20;
    int mb_ram = getBytesRAM() >> 20;
    int mb_mesh_cache = getBytesMeshCache() >> 20;

    int count_empty = getCountByState(TSDF_STATE_EMPTY);
    int count_gpu = getCountByState(TSDF_STATE_GPU);
    int count_ram = getCountByState(TSDF_STATE_RAM);
    int count_disk = getCountByState(TSDF_STATE_DISK);

    result += (boost::format("Volume count: %d\n") % grid_list_.size()).str();
    result += (boost::format("Empty: %d GPU: %d RAM: %d Disk: %d\n") % count_empty % count_gpu % count_ram % count_disk).str();
    result += (boost::format("MB GPU (expected): %d\n") % mb_gpu_expected).str();
    result += (boost::format("MB GPU (actual): %d\n") % mb_gpu_actual).str();
    result += (boost::format("MB RAM (run length encoded): %d\n") % mb_ram).str();
    result += (boost::format("MB Mesh Cache: %d\n") % mb_mesh_cache).str();

    result += (boost::format("Grids rendered last call: %d\n") % grids_rendered_last_call_.size()).str();

    static uint64_t deallocation_counter_last_value = 0;
    uint64_t deallocation_counter = OpenCLTSDF::getDeallocationCounter();
    result += (boost::format("Deallocations since last call: %d\n") % (deallocation_counter - deallocation_counter_last_value)).str();
    deallocation_counter_last_value = deallocation_counter;

    static uint64_t reallocation_counter_last_value = 0;
    uint64_t reallocation_counter = OpenCLTSDF::getReallocationCounter();
    result += (boost::format("Reallocations since last call: %d\n") % (reallocation_counter - reallocation_counter_last_value)).str();
    reallocation_counter_last_value = reallocation_counter;

    static uint64_t save_to_disk_counter_last_value = 0;
    uint64_t save_to_disk_counter = OpenCLTSDF::getSaveToDiskCounter();
    result += (boost::format("Save to disk since last call: %d\n") % (save_to_disk_counter - save_to_disk_counter_last_value)).str();
    save_to_disk_counter_last_value = save_to_disk_counter;

    static uint64_t load_from_disk_counter_last_value = 0;
    uint64_t load_from_disk_counter = OpenCLTSDF::getLoadFromDiskCounter();
    result += (boost::format("Load from disk since last call: %d\n") % (load_from_disk_counter - load_from_disk_counter_last_value)).str();
    load_from_disk_counter_last_value = load_from_disk_counter;

    size_t keyframe_images_bytes = getKeyframeMemoryBytes() >> 20;
    result += (boost::format("MB for keyframes: %d\n") % keyframe_images_bytes).str();

    return result;
}

size_t ModelGrid::getKeyframeMemoryBytes()
{
    size_t result = 0;
    BOOST_FOREACH(const KeyframeStructPtr & kfp, keyframe_list_) {
        result += kfp->mat_color_bgra.total() * kfp->mat_color_bgra.elemSize();
    }
    return result;
}

void ModelGrid::deallocateBuffers()
{
    deallocateAllVolumes();
}

#if 0
void ModelGrid::updateDeallocateOrder()
{
    deallocate_order_.resize(grid_list_.size());
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        deallocate_order_[i] = std::make_pair(grid_list_[i]->last_operation, i);

        cout << "deallocate_order_ " << i << ":" << deallocate_order_[i].first << endl;
    }
    std::sort(deallocate_order_.begin(), deallocate_order_.end());
}
#endif

void ModelGrid::setMaxWeightInVolume(float new_weight)
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        updateLastOperation(i);
        deallocateAsNeeded();

        grid_list_[i]->tsdf_ptr->setMaxWeightInVolume(new_weight);
    }
}

void ModelGrid::activateVolumesBasedOnAge(int min_age, int max_age)
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_struct = *grid_list_[i];
        grid_struct.active = true;
        if (grid_struct.merged) grid_struct.active = false; // never activate merged volumes
        if (min_age >= 0 && grid_struct.age < min_age) grid_struct.active = false;
        if (max_age >= 0 && grid_struct.age > max_age) grid_struct.active = false;
    }
}

void ModelGrid::activateVolumesBasedOnMerged(bool merged_value)
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_struct = *grid_list_[i];
        grid_struct.active = (grid_struct.merged == merged_value);
    }
}

void ModelGrid::setAllVolumesActive(bool value)
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_struct = *grid_list_[i];
        grid_struct.active = value;
    }
}

void ModelGrid::setAllNonMergedVolumesActive(bool value)
{
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_struct = *grid_list_[i];
        if (grid_struct.merged) continue;
        grid_struct.active = value;
    }
}

void ModelGrid::setVolumesActive(std::vector<int> const& volumes, bool value)
{
    for (std::vector<int>::const_iterator iter = volumes.begin(); iter != volumes.end(); ++iter) {
        grid_list_[*iter]->active = value;
    }
}

void ModelGrid::setNonMergedVolumesActive(std::vector<int> const& volumes, bool value)
{
    for (std::vector<int>::const_iterator iter = volumes.begin(); iter != volumes.end(); ++iter) {
        GridStruct & grid_struct = *grid_list_[*iter];
        if (grid_struct.merged) continue;
        grid_struct.active = value;
    }
}

void ModelGrid::getVolumesRenderedLastCall(std::vector<int> & volumes_rendered)
{
    volumes_rendered = grids_rendered_last_call_; // could generalize
}

void ModelGrid::getVolumesUpdatedLastCall(std::vector<int> & volumes_updated)
{
    volumes_updated = grids_updated_last_call_;
}

Eigen::Affine3f ModelGrid::getVolumePoseExternal(int volume)
{
    return grid_list_[volume]->pose_external;
}

void ModelGrid::setVolumePoseExternal(int volume, Eigen::Affine3f const& pose)
{
    grid_list_[volume]->pose_external = pose;
}

Eigen::Array3i ModelGrid::getGridCell(Eigen::Array3f const& p) const
{
    Eigen::Array3i grid_cell;
    Eigen::Array3f grid_cell_f = p.head<3>() / params_.volume.cell_size / (float)params_.grid.grid_size;
    grid_cell[0] = (int)floor(grid_cell_f[0]);
    grid_cell[1] = (int)floor(grid_cell_f[1]);
    grid_cell[2] = (int)floor(grid_cell_f[2]);
    return grid_cell;
}

Eigen::Affine3f ModelGrid::getPoseForGridCell(Eigen::Array3i const& grid_cell) const
{
    Eigen::Array3f cell_corner = params_.volume.cell_size * (grid_cell.cast<float>() * (float)params_.grid.grid_size - (float)params_.grid.border_size);
    Eigen::Affine3f result;
    result = Eigen::Translation3f(cell_corner);
    return result;
}

int ModelGrid::appendNewGridCellIfNeeded(Eigen::Array3i const& grid_cell)
{
    boost::tuple<int,int,int> key(grid_cell[0], grid_cell[1], grid_cell[2]);
    std::vector<int> & cells_for_spot = grid_to_list_map_[key];
    int existing_active_grid = -1;
    for (std::vector<int>::iterator c_iter = cells_for_spot.begin(); c_iter != cells_for_spot.end(); ++c_iter) {
        if (!grid_list_[*c_iter]->active) continue; // skip inactive
        existing_active_grid = *c_iter;
        break;
    }
    if (existing_active_grid < 0) {
        Eigen::Affine3f grid_pose = getPoseForGridCell(grid_cell);
        appendNewGridStruct(grid_pose);
        int new_index = grid_list_.size() - 1;
        cells_for_spot.push_back(new_index);
        existing_active_grid = new_index;
        grid_list_to_grid_cell_map_[new_index] = key; // debug?
    }
    // assume we will operate on this regardless (ok to bump newly created)
    updateLastOperation(existing_active_grid);
    return existing_active_grid;
}

std::vector<int> ModelGrid::mergeVolumeIntoActive(int volume, bool set_merged)
{
    return mergeVolumeIntoActive(*grid_list_[volume], set_merged);
}

// todo: take an index so we can keep updateOperation for grid_to_merge
std::vector<int> ModelGrid::mergeVolumeIntoActive(ModelGrid::GridStruct & grid_to_merge, bool set_merged)
{
    std::vector<int> grids_created_for_merge;

    // first allocate new volumes if needed
    // get a bounding box for the grid in its current pose
    Eigen::Array3f min_point, max_point;
    grid_to_merge.tsdf_ptr->getBBForPose(grid_to_merge.getExternalTSDFPose(), min_point, max_point);

    //cout << "grid_to_merge.pose:\n" << grid_to_merge.pose.matrix() << endl;
    //cout << "min_point: " << min_point.transpose() << " max_point: " << max_point.transpose() << endl;

    // check all potential locations and ensure an active grid there
    Eigen::Array3i min_grid_cell = getGridCell(min_point);
    Eigen::Array3i max_grid_cell = getGridCell(max_point);

    //cout << "min_grid_cell: " << min_grid_cell.transpose() << " max_grid_cell: " << max_grid_cell.transpose() << endl;


    for (int i = min_grid_cell[0]; i <= max_grid_cell[0]; ++i) {
        for (int j = min_grid_cell[1]; j <= max_grid_cell[1]; ++j) {
            for (int k = min_grid_cell[2]; k <= max_grid_cell[2]; ++k) {
                Eigen::Array3i grid_cell(i,j,k);
                int active_grid_cell = appendNewGridCellIfNeeded(grid_cell);
                if (active_grid_cell == grid_list_.size() - 1) {
                    grids_created_for_merge.push_back(active_grid_cell);
                }

                // just merge here
                GridStruct & grid_target = *grid_list_[active_grid_cell];
                Eigen::Affine3f relative_pose = grid_to_merge.getExternalTSDFPose().inverse() * grid_target.getExternalTSDFPose();
                grid_target.tsdf_ptr->addVolume(*grid_to_merge.tsdf_ptr, relative_pose);
            }
        }
    }

    if (set_merged) {
        grid_to_merge.merged = true;
    }

    return grids_created_for_merge;
}

std::vector<int> ModelGrid::mergeOtherModelGridActiveIntoActive(ModelGrid & other)
{
    std::vector<int> grids_created_for_merge;

    for (size_t i = 0; i < other.grid_list_.size(); ++i) {
        // note that each model_grid tracks memory limits separately
        other.updateLastOperation(i);
        other.deallocateAsNeeded();

        GridStruct & grid_to_merge = *other.grid_list_[i];
        if (!grid_to_merge.active) continue;
        std::vector<int> grids_created_this = mergeVolumeIntoActive(grid_to_merge, false);

        std::copy(grids_created_this.begin(), grids_created_this.end(), std::back_inserter(grids_created_for_merge));


#if 0
        cout << "DEBUG this summary:\n" << getSummary() << endl;
        cout << "DEBUG other summary:\n" << other.getSummary() << endl;
        cout << "DEBUG this map size: " << grid_to_list_map_.size() << endl;
        cout << "DEBUG other map size: " << other.grid_to_list_map_.size() << endl;

        for (GridToListMapT::iterator iter = other.grid_to_list_map_.begin(); iter != other.grid_to_list_map_.end(); ++iter) {
            cout << "Other ";
            cout << iter->first.get<0>() << "," << iter->first.get<1>()  << "," << iter->first.get<2>() << endl;
        }

        cout << "==================================" << endl;

        for (GridToListMapT::iterator iter = grid_to_list_map_.begin(); iter != grid_to_list_map_.end(); ++iter) {
            if (other.grid_to_list_map_.find(iter->first) == other.grid_to_list_map_.end()) {
                cout << "Unique ";
                cout << iter->first.get<0>() << "," << iter->first.get<1>()  << "," << iter->first.get<2>() << endl;
            }
        }
        cout << "-------" << endl;
#endif
    }

    return grids_created_for_merge;
}


void ModelGrid::getActiveStatus(std::vector<bool> & result)
{
    result.resize(grid_list_.size());
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        result[i] = grid_list_[i]->active;
    }
}

void ModelGrid::getMergedStatus(std::vector<bool> & result)
{
    result.resize(grid_list_.size());
    for (size_t i = 0; i < grid_list_.size(); ++i) {
        result[i] = grid_list_[i]->merged;
    }
}

void ModelGrid::boolsToIndices(std::vector<bool> const& include, std::vector<int> & result)
{
    result.clear();
    for (int i = 0; i < (int)include.size(); ++i) {
        if (include[i]) result.push_back(i);
    }
}

void ModelGrid::setValueInSphere(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Vector3f const& center, float radius)
{
    // here we need to allocate grids to do this...

    // just allocate in AABB around sphere
    Eigen::Array3f min_point_world = center.array() - radius;
    Eigen::Array3f max_point_world = center.array() + radius;
    Eigen::Array3i min_grid_cell = getGridCell(min_point_world);
    Eigen::Array3i max_grid_cell = getGridCell(max_point_world);

    std::vector<int> grids_created;
    for (int i = min_grid_cell[0]; i <= max_grid_cell[0]; ++i) {
        for (int j = min_grid_cell[1]; j <= max_grid_cell[1]; ++j) {
            for (int k = min_grid_cell[2]; k <= max_grid_cell[2]; ++k) {
                Eigen::Array3i grid_cell(i,j,k);
                int active_grid_cell = appendNewGridCellIfNeeded(grid_cell);
                if (active_grid_cell == grid_list_.size() - 1) {
                    grids_created.push_back(active_grid_cell);
                }

                GridStruct & grid_target = *grid_list_[active_grid_cell];
                Eigen::Vector3f center_in_grid = grid_target.getExternalTSDFPose().inverse() * center;
                grid_target.tsdf_ptr->setValueInSphere(d_value, dw_value, c_value, cw_value, center_in_grid, radius);
            }
        }
    }
}

// utility you needed for at least setValueInBox
void ModelGrid::getBBFromList(std::vector<Eigen::Vector3f> const& points, Eigen::Array3f & min_point, Eigen::Array3f & max_point)
{
    min_point = Eigen::Array3f::Constant(std::numeric_limits<float>::max());
    max_point = Eigen::Array3f::Constant(-std::numeric_limits<float>::max());
    for (int i = 0; i < points.size(); ++i) {
        Eigen::Array3f p = points[i].matrix();
        min_point = min_point.min(p);
        max_point = max_point.max(p);
    }
}

void ModelGrid::setValueInBox(float d_value, float dw_value, Eigen::Array4ub const& c_value, float cw_value, Eigen::Affine3f const& pose)
{
    // right...I'll just make the corners, transform them by pose, get min/max, and allocate based on those as in the sphere version
    std::vector<Eigen::Vector3f> corners;
    corners.push_back(Eigen::Vector3f(0,0,0));
    corners.push_back(Eigen::Vector3f(0,0,1));
    corners.push_back(Eigen::Vector3f(0,1,0));
    corners.push_back(Eigen::Vector3f(0,1,1));
    corners.push_back(Eigen::Vector3f(1,0,0));
    corners.push_back(Eigen::Vector3f(1,0,1));
    corners.push_back(Eigen::Vector3f(1,1,0));
    corners.push_back(Eigen::Vector3f(1,1,1));

    for (int i = 0; i < corners.size(); ++i) {
        corners[i] = pose * corners[i];
    }

    Eigen::Array3f min_point, max_point;
    getBBFromList(corners, min_point, max_point);

    Eigen::Array3i min_grid_cell = getGridCell(min_point);
    Eigen::Array3i max_grid_cell = getGridCell(max_point);

    std::vector<int> grids_created;
    for (int i = min_grid_cell[0]; i <= max_grid_cell[0]; ++i) {
        for (int j = min_grid_cell[1]; j <= max_grid_cell[1]; ++j) {
            for (int k = min_grid_cell[2]; k <= max_grid_cell[2]; ++k) {
                Eigen::Array3i grid_cell(i,j,k);
                int active_grid_cell = appendNewGridCellIfNeeded(grid_cell);
                if (active_grid_cell == grid_list_.size() - 1) {
                    grids_created.push_back(active_grid_cell);
                }

                GridStruct & grid_target = *grid_list_[active_grid_cell];
                Eigen::Affine3f pose_for_grid = grid_target.getExternalTSDFPose().inverse() * pose;
                grid_target.tsdf_ptr->setValueInBox(d_value, dw_value, c_value, cw_value, pose_for_grid);
            }
        }
    }
}

void ModelGrid::debugCheckOverlap()
{
    // for each non merged grid cell, check all later non-merged grid cells for significant overlap


    std::vector<std::pair<int,int> > overlap_pairs;
    for (int i = 0; i < grid_list_.size(); ++i) {
        GridStruct & grid_check = *grid_list_[i];
        if (grid_check.merged) continue;

        Eigen::Vector3f center_check = grid_check.tsdf_ptr->getSphereCenter();
        float radius_check = grid_check.tsdf_ptr->getSphereRadius();

        for (int j = i+i; j < grid_list_.size(); ++j) {
            GridStruct & grid_against = *grid_list_[j];
            if (grid_against.merged) continue;

            Eigen::Vector3f center_against = grid_against.tsdf_ptr->getSphereCenter();
            float radius_against = grid_against.tsdf_ptr->getSphereRadius();

            //Eigen::Affine3f relative_pose = grid_to_merge.pose.inverse() * grid_target.pose;
            Eigen::Affine3f relative_pose = grid_against.getExternalTSDFPose().inverse() * grid_check.getExternalTSDFPose();
            Eigen::Vector3f center_check_in_relative = relative_pose * center_check;

            float distance = (center_check_in_relative - center_against).norm();
            float overlap_factor = 0.2;
            float amount_for_overlap = (radius_check + radius_against) * overlap_factor;
            if (distance < amount_for_overlap) {
                overlap_pairs.push_back(std::make_pair(i, j));
            }
        }
    }

    // make a set
    std::set<int> overlapping_grids;
    for (std::vector<std::pair<int,int> >::iterator iter = overlap_pairs.begin(); iter != overlap_pairs.end(); ++iter) {
        overlapping_grids.insert(iter->first);
        overlapping_grids.insert(iter->second);
    }

    // and finally a vector
    std::vector<int> overlapping_grids_list;
    std::copy(overlapping_grids.begin(), overlapping_grids.end(), std::back_inserter(overlapping_grids_list));

    // visualize the set
    if (true) {
        MeshVertexVectorPtr vertex_ptr(new MeshVertexVector);
        generateBoundingLinesForGrids(overlapping_grids_list, *vertex_ptr);

        // recolor vertices?
        if (false) {
            Eigen::Vector4ub c(0,0,255,255);
            for (int i = 0; i < vertex_ptr->size(); ++i) {
                vertex_ptr->at(i).c = c;
            }
        }

        update_interface_->updateLines(params_.glfw_keys.debug_overlap, vertex_ptr);
    }

    // walk through the pairs
    if (false) {
        for (std::vector<std::pair<int,int> >::iterator iter = overlap_pairs.begin(); iter != overlap_pairs.end(); ++iter) {
            std::vector<int> pair_list;
            pair_list.push_back(iter->first);
            pair_list.push_back(iter->second);
            MeshVertexVectorPtr vertex_ptr(new MeshVertexVector);
            generateBoundingLinesForGrids(pair_list, *vertex_ptr);
            update_interface_->updateLines(params_.glfw_keys.debug_overlap, vertex_ptr);
            int sleep_ms = 100;
            boost::this_thread::sleep(boost::posix_time::milliseconds(sleep_ms));
        }
    }
}


void ModelGrid::getNonzeroVolumePointCloud(MeshVertexVector & vertex_list)
{
    vertex_list.clear();
    for (int i = 0; i < grid_list_.size(); ++i) {
        ModelGrid::GridStruct & grid_struct = *grid_list_[i];
        MeshVertexVector this_grid_vertices;
        grid_struct.tsdf_ptr->getPrettyVoxelCenters(grid_struct.getExternalTSDFPose(), this_grid_vertices);
        std::copy(this_grid_vertices.begin(), this_grid_vertices.end(), std::back_inserter(vertex_list));
    }
}

bool ModelGrid::UpdateKeyframeAndVolumeGraph(Frame & frame)
{
    boost::timer t;

    // put connections to current ("previous") keyframe (could not exist yet)
    if (!keyframe_list_.empty()) {
        addEdgesToKeyframeGraphForVolumes(current_keyframe_, grids_updated_last_call_);
    }
    bool changed_keyframe = updateKeyframe(frame);
    if (changed_keyframe) {
        addEdgesToKeyframeGraphForVolumes(current_keyframe_, grids_updated_last_call_);
    }

    cout << "[TIME] UpdateKeyframeAndVolumeGraph (removable if no loop closure): " << t.elapsed() << endl;

    return changed_keyframe;
}

bool ModelGrid::updateKeyframe(Frame& frame)
{
    bool add_keyframe = false;
    bool changed_keyframe = false;
    if (keyframe_list_.empty()) {
        add_keyframe = true;
    }
    else {
        Eigen::Affine3f last_camera_pose = camera_list_.back()->pose;
        Eigen::Affine3f current_keyframe_pose = camera_list_[keyframe_list_[current_keyframe_]->camera_index]->pose;

        float translation, positive_angle_degrees;
        EigenUtilities::getCameraPoseDifference(current_keyframe_pose, last_camera_pose, positive_angle_degrees, translation);

        bool search_for_keyframe = false;
        if (translation > params_.loop_closure.keyframe_distance_create) {
            search_for_keyframe = true;
        }
        if (positive_angle_degrees > params_.loop_closure.keyframe_angle_create) {
            search_for_keyframe = true;
        }
        // new bit: try to jump back to an older keyframe
        if (search_for_keyframe) {
            typedef std::map<int,int> Map;
            // find an existing keyframe within both graph distance and creation distance
            // if none exists, add_keyframe->true
            // new: use the keyframe_to_keyframe_edges_ for candidate keyframes
            // note: need a larger distance because can't rely on common volumes to tie them together...
            Map keyframes_within_distance;
            getVerticesWithinDistanceSimpleGraph(keyframe_to_keyframe_edges_, current_keyframe_, params_.loop_closure.keyframe_graph_distance, keyframes_within_distance);
            bool found_old_keyframe = false;
            BOOST_FOREACH(Map::value_type & p, keyframes_within_distance) {
                Eigen::Affine3f other_keyframe_pose = camera_list_[keyframe_list_[p.first]->camera_index]->pose;
                EigenUtilities::getCameraPoseDifference(other_keyframe_pose, last_camera_pose, positive_angle_degrees, translation);
                if (translation > params_.loop_closure.keyframe_distance_create || positive_angle_degrees > params_.loop_closure.keyframe_angle_create) continue;
                // found a suitable old keyframe
                found_old_keyframe = true;
                int new_current_keyframe = p.first;
                // TODO: CORRECT TRANSFORMATION
                keyframe_to_keyframe_edges_.push_back(EdgeStruct(current_keyframe_, new_current_keyframe, Eigen::Affine3f::Identity(), EDGE_STRUCT_EXISTING_KEYFRAME));
                current_keyframe_ = new_current_keyframe;
                changed_keyframe = true;
                break;
            }
            if (!found_old_keyframe) add_keyframe = true;
        }
    }

    // actually add the new keyframe
    if (add_keyframe) {
        features_ptr_->addFeaturesForFrameIfNeeded(frame);

        keyframe_list_.push_back(KeyframeStructPtr(new KeyframeStruct));
        keyframe_list_.back()->camera_index = (int)camera_list_.size() - 1;
        keyframe_list_.back()->mat_color_bgra = frame.mat_color_bgra;
        keyframe_list_.back()->mat_depth = frame.mat_depth;
        keyframe_list_.back()->keypoints = frame.keypoints;

        int new_current_keyframe = keyframe_list_.size() - 1;
        if (current_keyframe_ >= 0) {
            // TODO: CORRECT TRANSFORMATION
            keyframe_to_keyframe_edges_.push_back(EdgeStruct(current_keyframe_, new_current_keyframe, Eigen::Affine3f::Identity(), EDGE_STRUCT_DEFAULT));
        }
        current_keyframe_ = new_current_keyframe;

        changed_keyframe = true;
    }

#if 0
    // just debug text
    if (changed_keyframe) {
        cout << "[DEBUG] keyframe to volume graph:" << endl;
        std::vector<int> keyframe_degree(keyframe_list_.size(), 0);
        for (int i = 0; i < keyframe_edge_list_.size(); ++i) {
            keyframe_degree[keyframe_edge_list_[i].index_0]++;
        }
        for (int i = 0; i < keyframe_degree.size(); ++i) {
            cout << "Keyframe " << i << " : " << keyframe_degree[i] << endl;
        }
    }
#endif

    return changed_keyframe;
}

void ModelGrid::getCameraAndVolumeGraphDistances(int camera_index, std::map<int,int> & camera_distances, std::map<int,int> & volume_distances)
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    //typedef std::pair<int,int> E;
    typedef std::map<int, int> Map;

    // get the property map for vertex indices (needed?)
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    Map camera_to_vertex_map;
    Map volume_to_vertex_map;

    int vertex_counter = 0;
    // new version: graph should contain all vertices
    for (int keyframe_index = 0; keyframe_index < keyframe_list_.size(); ++keyframe_index) {
        camera_to_vertex_map[keyframe_index] = vertex_counter++;
    }
    for (int volume_index = 0; volume_index < grid_list_.size(); ++volume_index) {
        volume_to_vertex_map[volume_index] = vertex_counter++;
    }

    // now have vertex count, make reverse indices
    std::vector<int> vertex_to_camera(vertex_counter, -1);
    std::vector<int> vertex_to_volume(vertex_counter, -1);

    BOOST_FOREACH(Map::value_type const& p, camera_to_vertex_map) {
        vertex_to_camera[p.second] = p.first;
    }
    BOOST_FOREACH(Map::value_type const& p, volume_to_vertex_map) {
        vertex_to_volume[p.second] = p.first;
    }

    // finally build the graph
    Graph g(vertex_counter);
    std::vector<EdgeStruct> & edge_list = keyframe_edge_list_;
    BOOST_FOREACH(EdgeStruct & e, edge_list) {
        boost::add_edge(camera_to_vertex_map[e.index_0], volume_to_vertex_map[e.index_1], g);
#if 0
        cout << "add_edge " << e.camera_index << " - " << e.volume_index
            << "(" << camera_to_vertex_map[e.camera_index] << " - " << volume_to_vertex_map[e.volume_index] << ")"
            << endl;
#endif
    }

    // instead based on camera_index argument
    Vertex s = boost::vertex(camera_to_vertex_map[camera_index], g);

    // ok try distance recorder?
    // so this works....
    std::vector<int> d(boost::num_vertices(g), 0);
    boost::breadth_first_search(g, s, boost::visitor(boost::make_bfs_visitor(boost::record_distances(&d[0], boost::on_tree_edge()))));


    // well d holds distances now...

    camera_distances.clear();
    volume_distances.clear();

    for (int v = 0; v < boost::num_vertices(g); ++v) {
        int dist = d[v];
        if (dist <= 0 && v != s) dist = -1;

        int volume_id = vertex_to_volume[v];
        if (volume_id >= 0) {
            volume_distances[volume_id] = dist;
        }
        else {
            int camera_id = vertex_to_camera[v];
            if (camera_id >= 0) {
                camera_distances[camera_id] = dist;
            }
            else {
                // error
                cout << "neither volume nor camera id" << endl;
                throw std::runtime_error("neither volume nor camera id");
            }
        }
    }
}

void ModelGrid::getVertexDistancesSimpleGraph(std::vector<EdgeStruct> const& edge_struct_list, int index_start, std::map<int,int> & vertex_distances)
{
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    //typedef std::pair<int,int> E;
    typedef std::map<int, int> Map;

    // get the property map for vertex indices (needed?)
    typedef boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;

    Map external_to_internal_map;
    int vertex_counter = 0;

    // NOTE THIS ASSUMES CONNECTED (the way the other one used to...)
    // can't iterate over a particular grid_list_ or whatever
    // because we want to be generic to any edge_struct_list (where both index_0 and index_1 draw from the same pool)
    if (edge_struct_list.empty()) {
        external_to_internal_map.insert(std::make_pair(index_start, vertex_counter++));
    }
    else {
        BOOST_FOREACH(EdgeStruct const& e, edge_struct_list) {
            if (external_to_internal_map.find(e.index_0) == external_to_internal_map.end()) {
                external_to_internal_map.insert(std::make_pair(e.index_0, vertex_counter++));
            }
            if (external_to_internal_map.find(e.index_1) == external_to_internal_map.end()) {
                external_to_internal_map.insert(std::make_pair(e.index_1, vertex_counter++));
            }
        }
    }

    std::vector<int> internal_to_external(vertex_counter, -1);
    BOOST_FOREACH(Map::value_type const& p, external_to_internal_map) {
        internal_to_external[p.second] = p.first;
    }

    Graph g(vertex_counter);
    BOOST_FOREACH(EdgeStruct const& e, edge_struct_list) {
        boost::add_edge(external_to_internal_map[e.index_0], external_to_internal_map[e.index_1], g);
    }

    Vertex s = boost::vertex(external_to_internal_map[index_start], g);
    std::vector<int> d(boost::num_vertices(g), 0); // CAN'T USE BIG DEFAULT DISTANCE...distance recorder just adds 1...
    boost::breadth_first_search(g, s, boost::visitor(boost::make_bfs_visitor(boost::record_distances(&d[0], boost::on_tree_edge()))));

    vertex_distances.clear();

    for (int v = 0; v < boost::num_vertices(g); ++v) {
        int external = internal_to_external[v];
        if (external < 0) throw std::runtime_error("not possible");
        int dist = d[v];
        if (dist <= 0 && v != s) dist = -1;
        vertex_distances[external] = dist;
    }
}

void ModelGrid::getVerticesWithinDistanceSimpleGraph(std::vector<EdgeStruct> const& edge_struct_list, int index_start, int max_distance, std::map<int,int> & vertex_distances)
{
    typedef std::map<int,int> Map;
    vertex_distances.clear();
    Map all_vertex_distances;
    getVertexDistancesSimpleGraph(edge_struct_list, index_start, all_vertex_distances);
    BOOST_FOREACH(Map::value_type & p, all_vertex_distances) {
        // could warn about p.second<0
        if (p.second < 0 || p.second > max_distance) continue;
        vertex_distances.insert(p);
    }
}

void ModelGrid::prepareForRenderCurrent()
{
    const static bool old_loop_closure_bool = false;
    if (params_.loop_closure.activation_mode == ACTIVATION_MODE_ALL) {
        setAllVolumesActive(false);
        setAllNonMergedVolumesActive(true);
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_AGE) {
        activateAge(old_loop_closure_bool);
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_FULL_GRAPH) {
        activateFullGraph(old_loop_closure_bool);
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_KEYFRAME_GRAPH) {
        activateKeyframeGraph(current_keyframe_, false);
    }
    else {
        throw std::runtime_error("unknown activation_mode");
    }
}

void ModelGrid::prepareForRenderLoopClosureAllOld()
{
    const static bool old_loop_closure_bool = true;
    if (params_.loop_closure.activation_mode == ACTIVATION_MODE_ALL) {
        cout << "FATAL: prepareForRenderLoopClosureAllOld with ACTIVATION_MODE_ALL" << endl;
        throw std::runtime_error("ACTIVATION_MODE_ALL prepareForRenderLoopClosureAllOld");
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_AGE) {
        activateAge(old_loop_closure_bool);
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_FULL_GRAPH) {
        activateFullGraph(old_loop_closure_bool);
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_KEYFRAME_GRAPH) {
        activateKeyframeGraph(current_keyframe_, true);
    }
    else {
        throw std::runtime_error("unknown activation_mode");
    }
}

void ModelGrid::prepareForRenderLoopClosureTargetKeyframe(int keyframe_index)
{
    if (params_.loop_closure.activation_mode == ACTIVATION_MODE_ALL) {
        cout << "FATAL: prepareForRenderLoopClosureTargetKeyframe with ACTIVATION_MODE_ALL" << endl;
        throw std::runtime_error("ACTIVATION_MODE_ALL prepareForRenderLoopClosureTargetKeyframe");
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_AGE) {
        cout << "ACTIVATION_MODE_AGE not supported by prepareForRenderLoopClosureTargetKeyframe" << endl;
        throw std::runtime_error("ACTIVATION_MODE_AGE prepareForRenderLoopClosureTargetKeyframe");
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_FULL_GRAPH) {
        cout << "ACTIVATION_MODE_FULL_GRAPH not supported by prepareForRenderLoopClosureTargetKeyframe (yet)" << endl;
        throw std::runtime_error("ACTIVATION_MODE_FULL_GRAPH prepareForRenderLoopClosureTargetKeyframe");
    }
    else if (params_.loop_closure.activation_mode == ACTIVATION_MODE_KEYFRAME_GRAPH) {
        activateKeyframeGraph(keyframe_index, false);
    }
    else {
        throw std::runtime_error("unknown activation_mode");
    }
}


void ModelGrid::activateAge(bool loop_closure)
{
    if (loop_closure) {
        activateVolumesBasedOnAge(params_.loop_closure.activate_age, -1);
    }
    else {
        activateVolumesBasedOnAge(-1, params_.loop_closure.activate_age - 1);
    }
}


void ModelGrid::activateFullGraph(bool loop_closure)
{
    if (params_.loop_closure.activate_full_graph_depth <= 0) throw std::runtime_error("activate_graph_depth <= 0");

    if (keyframe_list_.empty()) return;

    //int camera_index = keyframe_list_[current_keyframe_]->camera_index;
    typedef std::map<int, int> Map;
    Map camera_distances;
    Map volume_distances;
    getCameraAndVolumeGraphDistances(current_keyframe_, camera_distances, volume_distances);

    // debug show activation distances
#if 0
    {
        cout << "------------------- current_keyframe: " << current_keyframe_ << endl;
        cout << "loop closure: " << loop_closure << endl;
        cout << "DEBUG activateGraph camera distances" << endl;
        BOOST_FOREACH(Map::value_type const& p, camera_distances) {
            cout << "C: " << p.first << " -> " << p.second << endl;
        }
        cout << "DEBUG activateGraph volume distances" << endl;
        BOOST_FOREACH(Map::value_type const& p, volume_distances) {
            cout << "V: " << p.first << " -> " << p.second << endl;
        }
    }
#endif

    setAllVolumesActive(false);
    std::vector<int> to_activate;
    for (Map::const_iterator iter = volume_distances.begin(); iter != volume_distances.end(); ++iter) {
        // I don't think we ever have disconnected graphs for this use case
        // feel free to deal with them THE RIGHT WAY later
        if (iter->second < 0) {
            cout << "-- WARNING: unexpected disconnected full graph...not activating volume: " << iter->first << endl;
            continue;
        }

        if (loop_closure && iter->second > params_.loop_closure.activate_full_graph_depth) {
            to_activate.push_back(iter->first);
        }
        else if (!loop_closure && iter->second <= params_.loop_closure.activate_full_graph_depth) {
            to_activate.push_back(iter->first);
        }
    }
    setNonMergedVolumesActive(to_activate, true);
}

void ModelGrid::activateKeyframeGraph(int target_keyframe, bool invert_selection)
{
    // so yeah...collect keyframes, use the keyframe list of volumes to activate volumes

    //if (keyframe_to_keyframe_edges_.empty()) return;
    if (keyframe_list_.empty()) return;

    // create a version with filtered out place recognition edges?
    // this is silly, but what's the alternative?
    std::vector<EdgeStruct> keyframe_to_keyframe_edges_filtered;
    BOOST_FOREACH(EdgeStruct const& e, keyframe_to_keyframe_edges_) {
        if (e.edge_type != EDGE_STRUCT_PLACE_RECOGNITION) {
            keyframe_to_keyframe_edges_filtered.push_back(e);
        }
    }

    typedef std::map<int,int> Map;
    Map vertex_distances_within_distance;
    //getVertexDistancesSimpleGraph(keyframe_to_keyframe_edges_, current_keyframe_, vertex_distances);
    getVerticesWithinDistanceSimpleGraph(keyframe_to_keyframe_edges_filtered, target_keyframe, params_.loop_closure.keyframe_graph_distance, vertex_distances_within_distance);

    std::set<int> volumes_within_distance;
    BOOST_FOREACH(Map::value_type & p, vertex_distances_within_distance) {
        // shouldn't happen:
        if (p.second < 0) {
            cout << "-- WARNING: unexpected disconnected activateKeyframeGraph vertex: " << p.first << endl;
            continue;
        }

        KeyframeStruct const& keyframe = *keyframe_list_[p.first];
        std::copy(keyframe.volumes.begin(), keyframe.volumes.end(), std::inserter(volumes_within_distance, volumes_within_distance.end()));
    }
    std::vector<int> volumes_vector(volumes_within_distance.begin(), volumes_within_distance.end());
    if (invert_selection) {
        setAllVolumesActive(false);
        setAllNonMergedVolumesActive(true);
        setNonMergedVolumesActive(volumes_vector, false);
    }
    else {
        setAllVolumesActive(false);
        setNonMergedVolumesActive(volumes_vector, true);
    }
}


// if this were to take the pose as an argument...well now a version does!
void ModelGrid::addEdgesToKeyframeGraphForVolumes(int keyframe_index, std::vector<int> const& volumes)
{
    KeyframeStruct & keyframe = *keyframe_list_[keyframe_index];
    int camera_index = keyframe.camera_index;
    addEdgesToKeyframeGraphForVolumes(keyframe_index, camera_list_[camera_index]->pose, volumes);
}

void ModelGrid::addEdgesToKeyframeGraphForVolumes(int keyframe_index, Eigen::Affine3f const& keyframe_pose, std::vector<int> const& volumes)
{
    KeyframeStruct & keyframe = *keyframe_list_[keyframe_index];
    Eigen::Affine3f keyframe_pose_inverse = keyframe_pose.inverse();
    for (std::vector<int>::const_iterator iter = volumes.begin(); iter != volumes.end(); ++iter) {
        // only add edges for those not yet seen by keyframe (note: only one edge between each keyframe and volume)
        if (keyframe.volumes.find(*iter) == keyframe.volumes.end()) {
            Eigen::Affine3f relative_pose = keyframe_pose_inverse * getVolumePoseExternal(*iter);
            keyframe_edge_list_.push_back(EdgeStruct(keyframe_index, *iter, relative_pose, EDGE_STRUCT_DEFAULT));
            keyframe.volumes.insert(*iter);
            volume_to_keyframe_[*iter].insert(keyframe_index);
        }
    }
}

bool ModelGrid::addEdgeToPoseGraph(G2OPoseGraph & pose_graph,
    std::map<int,int> const& camera_to_vertex_map, // or keyframe...it's just from edge indices to vertices
    std::map<int,int> const& volume_to_vertex_map,
    EdgeStruct const& edge)
{
    std::map<int, int>::const_iterator camera_find_iter = camera_to_vertex_map.find(edge.index_0);
    if (camera_find_iter == camera_to_vertex_map.end()) throw std::runtime_error("camera not found");
    int camera_vertex_id = camera_find_iter->second;

    std::map<int, int>::const_iterator volume_find_iter = volume_to_vertex_map.find(edge.index_1);
    if (volume_find_iter == volume_to_vertex_map.end()) throw std::runtime_error("volume not found");
    int volume_vertex_id = volume_find_iter->second;

    bool added = false;
    if (edge.edge_type == EDGE_STRUCT_LOOP_CLOSURE) {
        added = pose_graph.addEdge(camera_vertex_id, volume_vertex_id, EigenUtilities::getIsometry3d(edge.relative_pose), params_.loop_closure.loop_closure_edge_strength);
    }
    else {
        added = pose_graph.addEdge(camera_vertex_id, volume_vertex_id, EigenUtilities::getIsometry3d(edge.relative_pose));
    }
    return added;
}

bool ModelGrid::loopClosure(Frame & frame)
{
    if (params_.loop_closure.use_dbow_place_recognition) {
        return loopClosureWithPlaceRecognition(frame);
    }
    else {
        return loopClosureOld(frame);
    }
}

bool ModelGrid::loopClosureWithPlaceRecognition(Frame & frame)
{
    // todo: remove this unnecessary clear (it is super fast now though)
    render_buffers_.resetAllBuffers(); // hack for debug viewing

    if (keyframe_list_[current_keyframe_]->camera_index != (int)camera_list_.size() - 1) {
        cout << "loopClosureWithPlaceRecognition: Skipping loop closure (and resetting render buffers) because frame was not a new keyframe" << endl;
        return false;
    }
    // OK, so now you know that the last frame was a new keyframe

    if (params_.loop_closure.debug_max_total_loop_closures > 0) {
        // count existing loop closures by edges
        // might be worth having overall graph summary function(s)
        size_t existing_loop_closure_count = 0;
        BOOST_FOREACH(EdgeStruct const& e, keyframe_to_keyframe_edges_) {
            if (e.edge_type == EDGE_STRUCT_LOOP_CLOSURE) {
                existing_loop_closure_count++;
            }
        }
        if (existing_loop_closure_count >= params_.loop_closure.debug_max_total_loop_closures) {
            cout << "loopClosureWithPlaceRecognition: Skipping loop closure because existing_loop_closure_count: " << existing_loop_closure_count << endl;
            return false;
        }
    }

    Eigen::Affine3f current_camera_pose = getLastCameraPose();
    //Eigen::Affine3f current_camera_pose_inverse = current_camera_pose.inverse();

    // now can do the place recognition
    // put the loop detection here?  Only want it for new keyframes
    std::vector<unsigned int> place_detection_result;
    if (dbow_place_recognition_ptr_) {
        dbow_place_recognition_ptr_->addAndDetectBGRA(keyframe_list_.back()->mat_color_bgra, place_detection_result);
    }

    if (place_detection_result.empty()) {
        cout << "loopClosureWithPlaceRecognition: Failed to get a place recognition result" << endl;
        return false;
    }

    unsigned int target_keyframe_index = place_detection_result[0];
    KeyframeStruct & target_keyframe = *keyframe_list_[target_keyframe_index];

    // next get the feature-based alignment against this keyframe
    features_ptr_->addFeaturesForFrameIfNeeded(frame); // redundant (for keyframes)

    // find match
    std::vector<cv::DMatch> matches;
    features_ptr_->matchDescriptors(*frame.keypoints, *target_keyframe.keypoints, matches);
    Eigen::Affine3f feature_relative_pose;
    std::vector<cv::DMatch> inlier_matches;
    bool ransac_success = features_ptr_->ransac(*frame.keypoints, *target_keyframe.keypoints, matches, feature_relative_pose, inlier_matches);

    // go ahead and always create match image (for debugging loop closure)
    cv::Mat image_matches;
    {
        cv::Mat frame_bgr;
        cv::cvtColor(frame.mat_color_bgra, frame_bgr, CV_BGRA2BGR);
        cv::Mat target_bgr;
        cv::cvtColor(target_keyframe.mat_color_bgra, target_bgr, CV_BGRA2BGR);
        cv::drawMatches(frame_bgr, frame.keypoints->keypoints, target_bgr, target_keyframe.keypoints->keypoints, inlier_matches, image_matches);
        debug_images_["image_matches_loop"] = image_matches;
    }
    // only show if not command line
    if (!params_.volume_modeler.command_line_interface) {
        cv::imshow("image_matches_loop", image_matches);
    }

    // some debug save
    if (params_.dbow_place_recognition.debug_images_save) {
        std::string ransac_string = ransac_success ? "ransac_success" : "ransac_failure";
        std::string filename = (boost::format("%d - %d image_matches_loop (%s).png") % current_keyframe_ % target_keyframe_index % ransac_string).str();
        cout << "Debug save: " << filename << endl;
        cv::imwrite( (params_.volume_modeler.output / filename).string(), image_matches );
    }


    if (!ransac_success) {
        cout << "loopClosureWithPlaceRecognition: Ransac failure.  Inlier count: " << inlier_matches.size() << endl;
        return false;
    }

    // now have ransac success

    // do dense alignment + check
#if 0
    Eigen::Affine3f model_pose_from_features = feature_relative_pose.inverse() * camera_list_[target_keyframe.camera_index]->pose.inverse();
    Eigen::Affine3f initial_relative_pose = model_pose_from_features * current_camera_pose;
#endif

    // hmm
    ImageBuffer ib_target_color(all_kernels_->getCL());
    ImageBuffer ib_target_depth(all_kernels_->getCL());
    ImageBuffer ib_target_points(all_kernels_->getCL());
    ImageBuffer ib_target_normals(all_kernels_->getCL());

    ib_target_color.setMat(target_keyframe.mat_color_bgra);
    ib_target_depth.setMat(target_keyframe.mat_depth);
    KernelDepthImageToPoints _KernelDepthImageToPoints(*all_kernels_);
    _KernelDepthImageToPoints.runKernel(ib_target_depth, ib_target_points, params_.camera.focal, params_.camera.center);

    OpenCLNormals opencl_normals(all_kernels_);
    opencl_normals.computeNormalsWithBuffers(ib_target_points, params_.normals.max_depth_sigmas, params_.normals.smooth_iterations, ib_target_normals);

    Eigen::Affine3f align_result_pose_previous_keyframe = Eigen::Affine3f::Identity();
    std::vector<int> iterations_previous_keyframe;

    const static bool debug_images_dbow_loop_closure = true;
    if (debug_images_dbow_loop_closure) alignment_ptr_->setAlignDebugImages(true);

    // not just align...we have no mask now
    alignment_ptr_->alignMultiscaleNew(
        frame.image_buffer_color,
        frame.image_buffer_points,
        frame.image_buffer_normals,
        frame.image_buffer_align_weights,
        ib_target_color,
        ib_target_points,
        ib_target_normals,
        params_.camera,
        Eigen::Affine3f::Identity(),
        feature_relative_pose.inverse(),
        align_result_pose_previous_keyframe,
        iterations_previous_keyframe);
    // get before turning back off?  shouldn't matter
    std::vector<cv::Mat> align_debug_images_old_keyframe;
    alignment_ptr_->getAlignDebugImages(align_debug_images_old_keyframe);
    alignment_ptr_->setAlignDebugImages(params_.alignment.generate_debug_images);

    // images for reference
    // debug show:
    if (params_.dbow_place_recognition.debug_images_show) {
        cv::imshow("debug loop current frame", frame.mat_color_bgra);
        cv::imshow("debug loop target keyframe", target_keyframe.mat_color_bgra);

        // just ram them in right here
        for (size_t i = 0 ; i < align_debug_images_old_keyframe.size(); ++i) {
            cv::imshow("debug loop closure alignment", align_debug_images_old_keyframe[i]);
            cout << "pause for loop debug images..." << endl;
            cv::waitKey();
        }
    }

    // debug save:
    if (params_.dbow_place_recognition.debug_images_save) {
        if (!align_debug_images_old_keyframe.empty()){
            std::string filename = (boost::format("%d - %d align_debug_image.png") % current_keyframe_ % target_keyframe_index).str();
            cout << "Saving: " << filename << endl;
            cv::imwrite( (params_.volume_modeler.output / filename).string(), align_debug_images_old_keyframe.back() );
        }
    }

    // add a place recognition edge always
    // TODO: fix pose
    keyframe_to_keyframe_edges_.push_back(EdgeStruct(current_keyframe_, target_keyframe_index, Eigen::Affine3f::Identity(), EDGE_STRUCT_PLACE_RECOGNITION));

    // check the transform size?
    // check the error of the keyframe dense alignment?
    // note that you don't even need the dense alignment result to initialize dense against model...

    int keyframe_index_difference = abs(current_keyframe_ - (int)target_keyframe_index);
    if (keyframe_index_difference < params_.loop_closure.debug_min_keyframe_index_difference) {
        cout << "Ignoring place recognition because keyframe_index_difference: " << keyframe_index_difference << endl;
        return false;
    }

    /////////////////////////
    // ok, we're gonna trust this now

    // reference : feature_relative_pose.inverse() * camera_list_[target_keyframe.camera_index]->pose.inverse();
    Eigen::Affine3f loop_closure_render_model_pose =  align_result_pose_previous_keyframe * camera_list_[target_keyframe.camera_index]->pose.inverse();

    prepareForRenderLoopClosureTargetKeyframe(target_keyframe_index);

    {
        refreshUpdateInterface(); // show the new active volumes...
        // hack frustum
        {
            Frustum frustum(params_.camera, loop_closure_render_model_pose.inverse());
            std::vector<Eigen::Vector3f> frustum_lineset_points = frustum.getLineSetPoints();
            MeshVertexVectorPtr frustum_vertices(new MeshVertexVector);
            for (int i = 0; i < (int)frustum_lineset_points.size(); ++i) {
                frustum_vertices->push_back(MeshVertex());
                MeshVertex & v = frustum_vertices->back();
                v.p.head<3>() = frustum_lineset_points[i];
                v.p[3] = 1;
                v.c[0] = v.c[1] = v.c[2] = 200;
            }
            update_interface_->updateLines("hack loop closure frustum", frustum_vertices);

#if 0
            // also camera with frustum?
            UpdateInterface::PoseListPtrT frustum_camera(new UpdateInterface::PoseListT);
            frustum_camera->push_back(camera_pose);
            update_interface_->updateCameraList(params_.glfw_keys.frustum, frustum_camera);
            update_interface_->updateScale(params_.glfw_keys.frustum, 2);
            update_interface_->updateColor(params_.glfw_keys.frustum, Eigen::Array4ub(255,255,255,255));
#endif
        }

    }
    renderModel(params_.camera, loop_closure_render_model_pose, render_buffers_);

#if 0
    // always look at this render for now
    {
        cv::Mat render_color, render_normals;
        render_buffers_.getRenderPretty(render_color, render_normals);
        cv::Mat both = create1x2(render_color, render_normals);
        cv::imshow("loop closure old volumes render", both);
        cout << "pause for loop debug images..." << endl;
        cv::waitKey();
    }
  #endif

    // debug save:
    if (params_.dbow_place_recognition.debug_images_save) {
        cv::Mat render_color, render_normals;
        render_buffers_.getRenderPretty(render_color, render_normals);
        cv::Mat both = create1x2(render_color, render_normals);

        std::string filename = (boost::format("%d - %d render old volumes.png") % current_keyframe_ % target_keyframe_index).str();
        cout << "Saving: " << filename << endl;
        cv::imwrite( (params_.volume_modeler.output / filename).string(), both );
    }


    /////////////// align
    Eigen::Affine3f align_result_pose_old_volumes = Eigen::Affine3f::Identity();  // the result, duh

    // always debug right now
    alignment_ptr_->setAlignDebugImages(true);

    std::vector<int> iterations_old_volumes;
    bool align_success = alignment_ptr_->alignMultiscaleNew(
        frame.image_buffer_color,
        frame.image_buffer_points,
        frame.image_buffer_normals,
        frame.image_buffer_align_weights,
        render_buffers_.getImageBufferColorImage(),
        render_buffers_.getImageBufferPoints(),
        render_buffers_.getImageBufferNormals(),
        params_.camera,
        loop_closure_render_model_pose,
        Eigen::Affine3f::Identity(), // start from identity guess (no initial relative pose)
        align_result_pose_old_volumes,
        iterations_old_volumes);
    // get before turning back off?  shouldn't matter
    std::vector<cv::Mat> align_debug_images_old_volumes;
    alignment_ptr_->getAlignDebugImages(align_debug_images_old_volumes);
    alignment_ptr_->setAlignDebugImages(params_.alignment.generate_debug_images);


#if 0
    {
        cv::imshow("debug loop current frame", frame.mat_color_bgra);
        cv::imshow("debug loop target keyframe", target_keyframe.mat_color_bgra);

        // just ram them in right here
        for (size_t i = 0 ; i < align_debug_images_old_volumes.size(); ++i) {
            cv::imshow("debug loop closure alignment old volumes", align_debug_images_old_volumes[i]);
            cout << "pause for loop debug images..." << endl;
            cv::waitKey();
        }
    }
#endif


    Eigen::Affine3f camera_pose_from_loop_closure = align_result_pose_old_volumes.inverse();


    // add the loop closure edges:
#if 1
    addEdgesToKeyframeGraphForVolumes(current_keyframe_, camera_pose_from_loop_closure, grids_rendered_last_call_);
    keyframe_to_keyframe_edges_.push_back(EdgeStruct(current_keyframe_, target_keyframe_index, Eigen::Affine3f::Identity(), EDGE_STRUCT_LOOP_CLOSURE));
#endif

#if 1
    {
        G2OPoseGraph pose_graph;
        std::map<int, int> volume_to_vertex_map;
        std::map<int, int> keyframe_to_vertex_map;
        createG2OPoseGraphKeyframes(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map);
        // can debug iterations here as in old code
        optimizeG2OPoseGraphKeyframes(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map, params_.loop_closure.optimize_iterations);
    }
#else
     return false;
#endif
   


    cout << "Reached end of loopClosureWithPlaceRecognition successfully!" << endl;
    return true;
}

bool ModelGrid::loopClosureOld(Frame & frame)
{
    // observation: it's weird that alignment happens in volume_modeler, but loop closure alignment happens here.
    // everything will end up in ModelGrid...


    if (keyframe_list_[current_keyframe_]->camera_index != (int)camera_list_.size() - 1) {
        if (params_.volume_modeler.verbose) cout << "Skipping loop closure (and reseting render buffers) because frame was not a new keyframe" << endl;
        boost::timer t_stupid;
        render_buffers_.resetAllBuffers(); // hack for debug viewing
        if (params_.volume_modeler.verbose) cout << "[TIMING] Stupid time wasted to clear buffers: " << t_stupid.elapsed() << endl;
        return false;
    }
    // OK, so now you know that the last frame was a new keyframe

    // currently checks for loop closure based on rendering (slow?)
    Eigen::Affine3f loop_closure_render_camera_pose = getLastCameraPose();
    Eigen::Affine3f loop_closure_render_model_pose = loop_closure_render_camera_pose.inverse();
    prepareForRenderLoopClosureAllOld();
    {
        boost::timer t;
        render_buffers_.resetAllBuffers(); // had to add this...happened automatically with volume_modeler render(pose)
        renderModel(params_.camera, loop_closure_render_model_pose, render_buffers_);
        if (params_.volume_modeler.verbose) cout << "[TIMING] render for loop closure: " << t.elapsed() << endl;
    }
    std::set<int> volumes_rendered_for_loop_check_set(grids_rendered_last_call_.begin(), grids_rendered_last_call_.end());

    // todo: accelerate this with an OpenCL version
    cv::Mat render_mask = render_buffers_.getImageBufferMask().getMat();
    int rendered_point_count = cv::countNonZero(render_mask);

    float fraction_of_screen = (float)rendered_point_count / (float)render_mask.total();
    if (fraction_of_screen < params_.loop_closure.min_fraction) {
        if (params_.volume_modeler.verbose) cout << "Loop closure: failed min fraction test" << endl;
        return false;
    }


    //////////////// keyframe
    // need features for loop closure...keyframes, anyway
    features_ptr_->addFeaturesForFrameIfNeeded(frame);

    // try checking for loop closure with features against keyframes always
    // note that this automatically goes from oldest to newest
    Eigen::Affine3f initial_relative_pose = Eigen::Affine3f::Identity(); // to be updated by keyframe matching
    int which_keyframe_success = -1;
    int keyframes_within_distance = 0;
    int keyframes_with_overlapping_volumes = 0;
    for (size_t i = 0; i < keyframe_list_.size(); ++i) {
        KeyframeStruct & keyframe = *keyframe_list_[i];
        Eigen::Affine3f keyframe_pose = camera_list_[keyframe.camera_index]->pose;
        Eigen::Affine3f relative_pose = keyframe_pose * loop_closure_render_model_pose;
        float translation = relative_pose.translation().norm();
        Eigen::AngleAxisf aa(relative_pose.rotation());
        float angle_degrees = 180.f / M_PI * aa.angle();

        // continue if we're too far away
        if (translation > params_.loop_closure.keyframe_distance_match || angle_degrees > params_.loop_closure.keyframe_angle_match) continue;
        keyframes_within_distance++;

        // check for volume intersection
        // only want keyframes which saw at least one of the "old" volumes
        // now thinking about it...if a keyframe sees ANY old volume, we add it.
        // seems like a PROXY for being an "old" keyframe
        int number_of_volumes_in_intersection = 0;
        for (std::set<int>::iterator iter = keyframe.volumes.begin(); iter != keyframe.volumes.end(); ++iter) {
            if (volumes_rendered_for_loop_check_set.find(*iter) != volumes_rendered_for_loop_check_set.end()) {
                number_of_volumes_in_intersection++;
            }
        }
        if (number_of_volumes_in_intersection == 0) continue;
        keyframes_with_overlapping_volumes++;

        // find match
        std::vector<cv::DMatch> matches;
        features_ptr_->matchDescriptors(*frame.keypoints, *keyframe.keypoints, matches);
        Eigen::Affine3f feature_relative_pose;
        std::vector<cv::DMatch> inlier_matches;
        bool ransac_success = features_ptr_->ransac(*frame.keypoints, *keyframe.keypoints, matches, feature_relative_pose, inlier_matches);

        if (ransac_success) {
            // debug
            if (!params_.volume_modeler.command_line_interface) {
                cv::Mat image_matches;
                cv::drawMatches(frame.mat_color_bgra, frame.keypoints->keypoints, keyframe.mat_color_bgra, keyframe.keypoints->keypoints, inlier_matches, image_matches);
                //cv::imshow("image_matches_loop", image_matches);
                debug_images_["image_matches_loop"] = image_matches;
            }

            // there's probably a better way to do this:
            Eigen::Affine3f model_pose_from_features = feature_relative_pose.inverse() * camera_list_[keyframe.camera_index]->pose.inverse();
            initial_relative_pose = model_pose_from_features * loop_closure_render_camera_pose; // note camera pose offsets model pose...

            which_keyframe_success = i;
            break;
        }
    }

    if (params_.volume_modeler.verbose) cout << "keyframes_within_distance: " << keyframes_within_distance << endl;
    if (params_.volume_modeler.verbose) cout << "keyframes_with_overlapping_volumes: " << keyframes_with_overlapping_volumes << endl;

    if (which_keyframe_success < 0) {
        if (params_.volume_modeler.verbose) cout << "Loop Closure: Failed keyframe alignment" << endl;
        return false;
    }


    /////////////// align
    Eigen::Affine3f result_pose = Eigen::Affine3f::Identity();  // the result, duh
    //alignment.prepareFrame(whatever); shouldn't be needed...already prepared from initial alignment
    std::vector<int> result_iterations;
    bool align_success = alignment_ptr_->align(
        frame.image_buffer_color,
        frame.image_buffer_points,
        frame.image_buffer_normals,
        frame.image_buffer_align_weights,
        render_buffers_.getImageBufferColorImage(),
        render_buffers_.getImageBufferPoints(),
        render_buffers_.getImageBufferNormals(),
        render_buffers_.getImageBufferMask(),
        params_.camera,
        loop_closure_render_model_pose,
        initial_relative_pose,
        result_pose,
        result_iterations);
    if (params_.volume_modeler.verbose) {
        cout << "result_iterations (loop): ";
        for (int i = 0; i < result_iterations.size(); ++i) cout << result_iterations[i] << " ";
        cout << endl;
    }

    Eigen::Affine3f camera_pose_from_loop_closure = result_pose.inverse();

    // debug:
    {
        Eigen::Affine3f model_pose_from_features = initial_relative_pose * loop_closure_render_model_pose;
        Eigen::Affine3f model_pose_result = result_pose;
        cout << "Doing loop closure magnitude comparison (remove?)..." << endl;
        float a, d;
        EigenUtilities::getCameraPoseDifference(model_pose_from_features.inverse(), model_pose_result.inverse(), a, d);
        cout << "Angle: " << a << " Distance: " << d << endl;
    }

    if (align_success) {
        if (params_.volume_modeler.verbose) cout << "Loop closure: align_success" << endl;

        //////////
        // add loop closure constraints
        addEdgesToKeyframeGraphForVolumes(current_keyframe_, camera_pose_from_loop_closure, grids_rendered_last_call_);

        /////////
        // save mesh before loop
        if (params_.loop_closure.debug_save_meshes) {
            MeshVertexVector vertex_list;
            TriangleVector triangle_list;
            generateMesh(vertex_list, triangle_list);
            fs::path save_file = params_.volume_modeler.output / (boost::format("%05d_before_loop.ply") % getCameraListSize()).str();
            MeshUtilities::saveMesh(vertex_list, triangle_list, save_file);
        }

        //////////
        // recreate pose graph freshly
        {
            std::string name_debug_optimize = "name_debug_optimize";

            G2OPoseGraph pose_graph;
            std::map<int, int> volume_to_vertex_map;
            std::map<int, int> keyframe_to_vertex_map;
            createG2OPoseGraphKeyframes(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map);
            if (params_.loop_closure.debug_optimize) {
                refreshUpdateInterface();
                {
                    // mesh
                    MeshPtr mesh_ptr (new Mesh);
                    generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
                    update_interface_->updateMesh(name_debug_optimize, mesh_ptr);
                }
                cout << "debug_optimize pause..." << endl;
                cv::waitKey();
                for (int i = 0; i < params_.loop_closure.optimize_iterations; ++i) {
                    optimizeG2OPoseGraphKeyframes(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map, 1);
                    refreshUpdateInterface();
                    {
                        // mesh
                        MeshPtr mesh_ptr (new Mesh);
                        generateMesh(mesh_ptr->vertices, mesh_ptr->triangles);
                        update_interface_->updateMesh(name_debug_optimize, mesh_ptr);
                    }
                    cout << "debug_optimize pause..." << endl;
                    cv::waitKey();
                }
            }
            else {
                optimizeG2OPoseGraphKeyframes(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map, params_.loop_closure.optimize_iterations);
            }
        }

        //////////
        // save after loop
        if (params_.loop_closure.debug_save_meshes) {
            MeshVertexVector vertex_list;
            TriangleVector triangle_list;
            generateMesh(vertex_list, triangle_list);
            fs::path save_file = params_.volume_modeler.output / (boost::format("%05d_after_loop.ply") % getCameraListSize()).str();
            MeshUtilities::saveMesh(vertex_list, triangle_list, save_file);
        }

        ////////
        // record the loop closure
        if (which_keyframe_success < 0) throw std::runtime_error("you screwed up");
        keyframe_to_keyframe_edges_.push_back(EdgeStruct(current_keyframe_, which_keyframe_success, Eigen::Affine3f::Identity(), EDGE_STRUCT_LOOP_CLOSURE));

        /////////////
        // TODO: MERGING? (CORRECTLY!)
    }
    else {
        if (params_.volume_modeler.verbose) cout << "Loop closure: align_success was FALSE!" << endl;
    }

    return align_success;
}



void ModelGrid::createG2OPoseGraphKeyframes(
    G2OPoseGraph & pose_graph,
    std::map<int, int> & keyframe_to_vertex_map,
    std::map<int, int> & volume_to_vertex_map)
{
    volume_to_vertex_map.clear();
    keyframe_to_vertex_map.clear();

    for (int i = 0; i < (int)keyframe_list_.size(); ++i) {
        KeyframeStruct & keyframe = *keyframe_list_[i];
        CameraStruct & keyframe_cam = *camera_list_[keyframe.camera_index];
        Eigen::Isometry3d vertex_pose_g2o = EigenUtilities::getIsometry3d(keyframe_cam.pose);
        int vertex_id = pose_graph.addVertex(vertex_pose_g2o, false);
        keyframe_to_vertex_map[i] = vertex_id;
    }
    int volume_count = getVolumeCount();
    for (int i = 0; i < volume_count; ++i) {
        Eigen::Isometry3d vertex_pose_g2o = EigenUtilities::getIsometry3d(getVolumePoseExternal(i));
        int vertex_id = pose_graph.addVertex(vertex_pose_g2o, false);
        volume_to_vertex_map[i] = vertex_id;
    }
    BOOST_FOREACH(EdgeStruct const& e, keyframe_edge_list_) {
        addEdgeToPoseGraph(pose_graph, keyframe_to_vertex_map, volume_to_vertex_map, e);
    }

    // set graph params and fixed vertex
    pose_graph.setVerbose(true); // debug
    int fixed_vertex = keyframe_to_vertex_map.find(current_keyframe_)->second;
    pose_graph.setVertexFixed(fixed_vertex, true);
}

void ModelGrid::optimizeG2OPoseGraphKeyframes(
    G2OPoseGraph & pose_graph,
    std::map<int, int> const& keyframe_to_vertex_map,
    std::map<int, int> const& volume_to_vertex_map,
    int iterations)
{
    // optimize graph
    pose_graph.optimize(iterations);

    // update cameras and grids
    for (std::map<int, int>::const_iterator iter = keyframe_to_vertex_map.begin(); iter != keyframe_to_vertex_map.end(); ++iter) {
        camera_list_[keyframe_list_[iter->first]->camera_index]->pose = EigenUtilities::getAffine3f(pose_graph.getVertexPose(iter->second));
    }
    for (std::map<int, int>::const_iterator iter = volume_to_vertex_map.begin(); iter != volume_to_vertex_map.end(); ++iter) {
        grid_list_[iter->first]->pose_external = EigenUtilities::getAffine3f(pose_graph.getVertexPose(iter->second));
    }
}

void ModelGrid::refreshUpdateInterface()
{
    ModelBase::refreshUpdateInterface();

    if (update_interface_) {
        // now always do this
        {
            // keyframes
            {
                UpdateInterface::PoseListPtrT pose_list_ptr(new UpdateInterface::PoseListT);
                for (int i = 0; i < keyframe_list_.size(); ++i) {
                    pose_list_ptr->push_back(camera_list_[keyframe_list_[i]->camera_index]->pose);
                }
                update_interface_->updateCameraList(params_.glfw_keys.cameras_keyframes_and_graph, pose_list_ptr);
            }

            // current keyframe
            // NOTE: CANNOT BE TURNED OFF RIGHT NOW
            if (current_keyframe_ >= 0) {
                UpdateInterface::PoseListPtrT pose_list_ptr(new UpdateInterface::PoseListT);
                pose_list_ptr->push_back(camera_list_[keyframe_list_[current_keyframe_]->camera_index]->pose);
                std::string name = "active_keyframe";
                update_interface_->updateCameraList(name, pose_list_ptr);
                update_interface_->updateScale(name, 5.0);
            }
        }

        // keyframe_to_keyframe_edges_ edges
        {
            MeshVertexVectorPtr vertices(new MeshVertexVector);
            // first vector of vector3fs
            std::vector<std::pair<Eigen::Vector3f, Eigen::Vector4ub> > points_and_colors;
            for (int i = 0; i < keyframe_to_keyframe_edges_.size(); ++i) {
                EdgeStruct const& e = keyframe_to_keyframe_edges_[i];
                Eigen::Vector3f v1 = camera_list_[keyframe_list_[e.index_0]->camera_index]->pose.translation();
                Eigen::Vector3f v2 = camera_list_[keyframe_list_[e.index_1]->camera_index]->pose.translation();
                Eigen::Vector4ub c(255,255,255,255);
                if (e.edge_type == EDGE_STRUCT_LOOP_CLOSURE) c = Eigen::Vector4ub(255,0,0,255);
                else if (e.edge_type == EDGE_STRUCT_PLACE_RECOGNITION) c = Eigen::Vector4ub(255,0,255,255);
                points_and_colors.push_back(std::make_pair(v1,c));
                points_and_colors.push_back(std::make_pair(v2,c));
            }
            // then put in vertices
            for (int i = 0; i < points_and_colors.size(); ++i) {
                vertices->push_back(MeshVertex());
                MeshVertex & v = vertices->back();
                v.p.head<3>() = points_and_colors[i].first;
                v.p[3] = 1;
                v.c = points_and_colors[i].second;
            }

            // match number to keyframes?
            update_interface_->updateLines(params_.glfw_keys.cameras_keyframes_and_graph, vertices);
        }

        // need grid poses
        std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list;
        getPoseExternalTSDFList(pose_list);

        // and original poses (before loop closure)
        std::vector<boost::shared_ptr<Eigen::Affine3f> > pose_list_original;
        getPoseOriginalList(pose_list_original);

        // and the merged status
        std::vector<bool> merged_list;
        getMergedStatus(merged_list);

#if 0
        // bounding meshes
        std::vector<MeshVertexVectorPtr> bounding_vll;
        std::vector<TriangleVectorPtr> bounding_tll;
        generateAllBoundingMeshes(bounding_vll, bounding_tll);
#endif

        // and bounding lines?
        std::vector<MeshVertexVectorPtr> bounding_lines_vll;
        generateAllBoundingLines(bounding_lines_vll);


        // all (not merged) volumes
        {
            MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
            TriangleVector empty_triangles;
            for (size_t i = 0; i < bounding_lines_vll.size(); ++i) {
                if (merged_list[i]) continue;
                MeshVertexVector vertices_copy = *bounding_lines_vll[i];
                MeshUtilities::transformMeshVertices(*pose_list[i], vertices_copy);

                // color as gray those in RAM
                if (grid_list_[i]->tsdf_ptr->getTSDFState() == TSDF_STATE_RAM) {
                    BOOST_FOREACH(MeshVertex & v, vertices_copy) {
                        v.c = Eigen::Array4ub::Constant(64);
                    }
                }

                // color as black those on disk
                if (grid_list_[i]->tsdf_ptr->getTSDFState() == TSDF_STATE_DISK) {
                    BOOST_FOREACH(MeshVertex & v, vertices_copy) {
                        v.c = Eigen::Array4ub::Constant(0);
                    }
                }

                MeshUtilities::appendMesh(*vertices_ptr, empty_triangles, vertices_copy, empty_triangles);
            }
            update_interface_->updateLines(params_.glfw_keys.volumes_all, vertices_ptr);
        }

        // active volumes
        {
            // recently removed reset of active to non-loop closure
            std::vector<bool> active_list;
            getActiveStatus(active_list);

            {
                MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
                TriangleVector empty_triangles;
                for (size_t i = 0; i < bounding_lines_vll.size(); ++i) {
                    if (!active_list[i]) continue;
                    MeshVertexVector vertices_copy = *bounding_lines_vll[i];
                    MeshUtilities::transformMeshVertices(*pose_list[i], vertices_copy);
                    MeshUtilities::appendMesh(*vertices_ptr, empty_triangles, vertices_copy, empty_triangles);
                }
                update_interface_->updateLines(params_.glfw_keys.volumes_active, vertices_ptr);
            }
        }


        // pose graph lines
        {
            std::vector<EdgeStruct> & edges = keyframe_edge_list_;
            MeshVertexVectorPtr graph_line_vertices(new MeshVertexVector);
            for (std::vector<EdgeStruct>::iterator iter = edges.begin(); iter != edges.end(); ++iter) {
                MeshVertex v1, v2;
                v1.c[0] = v1.c[1] = v1.c[2] = 0;
                v2.c[0] = v2.c[1] = v2.c[2] = 0;
                if (iter->edge_type == EDGE_STRUCT_LOOP_CLOSURE) {
                    // blue
                    v1.c[0] = v2.c[0] = 255;
                }
                else {
                    // red
                    v1.c[2] = v2.c[2] = 255;
                }

                v1.p.head<3>() = camera_list_[keyframe_list_[iter->index_0]->camera_index]->pose.translation();
                v1.p[3] = 1; // not needed probably
                v2.p.head<3>() = getVolumePoseExternal(iter->index_1).translation();
                v2.p[3] = 1; // not needed probably

                graph_line_vertices->push_back(v1);
                graph_line_vertices->push_back(v2);
            }
            update_interface_->updateLines(params_.glfw_keys.pose_graph_all, graph_line_vertices);
        }

        // YOU WERE DOING THIS UNTIL VERY RECENTLY...NOT QUITE SURE WHAT THE POINT WAS
        // I think to test bipartite graph display?
#if 0
        // and pose graph lines as a graph
        // same check??
        {
            // sadly, must reform vertices to be unique...
            // this is stupid...do a bipartite graph display...
            UpdateInterface::PoseListPtrT vertices_first_ptr(new UpdateInterface::PoseListT);
            UpdateInterface::PoseListPtrT vertices_second_ptr(new UpdateInterface::PoseListT);
            UpdateInterface::EdgeListPtrT edges_ptr(new UpdateInterface::EdgeListT);

            for (int i = 0; i < keyframe_list_.size(); ++i) {
                vertices_first_ptr->push_back(camera_list_[keyframe_list_[i]->camera_index]->pose);
            }
            for (int i = 0; i < grid_list_.size(); ++i) {
                vertices_second_ptr->push_back(grid_list_[i]->getExternalTSDFPose());
            }
            for (int i = 0; i < keyframe_edge_list_.size(); ++i) {
                edges_ptr->push_back(std::make_pair(keyframe_edge_list_[i].index_0, keyframe_edge_list_[i].index_1));
            }
            update_interface_->updateBipartiteGraph("bipartite_graph", vertices_first_ptr, vertices_second_ptr, edges_ptr);
        }
#endif

        // only active pose graph
        // gotta fix this to reflect actual activation
        // note active pose graph is not actually well defined in general (only active volumes)
#if 0
        if (current_keyframe_ >= 0) {
            int camera_index = keyframe_list_[current_keyframe_]->camera_index;

            typedef std::map<int, int> Map;
            Map camera_distances;
            Map volume_distances;
            getCameraAndVolumeGraphDistances(current_keyframe_, camera_distances, volume_distances);

            std::vector<EdgeStruct> edges;
            getFullKeyframeEdgeList(edges);
            MeshVertexVectorPtr graph_line_vertices(new MeshVertexVector);
            for (std::vector<EdgeStruct>::iterator iter = edges.begin(); iter != edges.end(); ++iter) {
                // here is the interesting distance check:
                if (camera_distances.find(iter->index_0) == camera_distances.end() ||
                    volume_distances.find(iter->index_1) == volume_distances.end()) {
                        cout << "FAIL" << endl;
                        throw std::runtime_error("fail");
                }
                if (camera_distances[iter->index_0] < 0 || camera_distances[iter->index_0] > params_.loop_closure.activate_graph_depth) continue;
                if (volume_distances[iter->index_1] < 0 || volume_distances[iter->index_1] > params_.loop_closure.activate_graph_depth) continue;

                MeshVertex v1, v2;
                v1.c[0] = v1.c[1] = v1.c[2] = 0;
                v2.c[0] = v2.c[1] = v2.c[2] = 0;
                if (iter->is_loop_closure) {
                    // blue
                    v1.c[0] = v2.c[0] = 255;
                }
                else {
                    // red
                    v1.c[2] = v2.c[2] = 255;
                }

                v1.p.head<3>() = camera_list_[keyframe_list_[iter->index_0]->camera_index]->pose.translation();
                v1.p[3] = 1; // not needed probably
                v2.p.head<3>() = getVolumePoseExternal(iter->index_1).translation();
                v2.p[3] = 1; // not needed probably

                graph_line_vertices->push_back(v1);
                graph_line_vertices->push_back(v2);
            }
            update_interface_->updateLines("6 or something", graph_line_vertices);
        }
#endif

        // todo: take a look at the regular grid structure
#if 0
        std::vector<MeshVertexVectorPtr> grid_lines_vll;
        model_grid->generateAllGridMeshes(grid_lines_vll);
#endif

#if 0
        // can get same idea with just original poses
        {
            prepareForRender(false);
            std::vector<bool> active_list;
            getActiveStatus(active_list);

            const std::string string_key(params_.glfw_keys.volumes_original_pose);
            {
                MeshVertexVectorPtr vertices_ptr (new MeshVertexVector);
                TriangleVector empty_triangles;
                for (size_t i = 0; i < bounding_lines_vll.size(); ++i) {
                    if (!active_list[i]) continue;
                    MeshVertexVector vertices_copy = *bounding_lines_vll[i];
                    MeshUtilities::transformMeshVertices(*pose_list_original[i], vertices_copy);
                    MeshUtilities::appendMesh(*vertices_ptr, empty_triangles, vertices_copy, empty_triangles);
                }
                update_interface_->updateLines(string_key, vertices_ptr);
            }
        }
#endif


    }
}

