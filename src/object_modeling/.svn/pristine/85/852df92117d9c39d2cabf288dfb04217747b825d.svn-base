#include "stdafx.h" // why are you doing this still?

#include "MarchingCubesManyVolumes.h"

#include "MarchingCubes.h"

#include "opencv_utilities.h"
#include "MeshUtilities.h"
#include "ordering_container.h"

using std::cout;
using std::endl;

MarchingCubesManyVolumes::MarchingCubesManyVolumes()
{
}

// static
void MarchingCubesManyVolumes::generateMeshAndValidity(std::vector<OpenCLTSDFPtr> & tsdf_ptr_list, std::vector<boost::shared_ptr<Eigen::Affine3f> > const& pose_ptr_list,
    boost::shared_ptr<UpdateInterface> update_interface, size_t max_mb_gpu, 
    Mesh & result_mesh, std::vector<bool> & result_validity)
{
    cout << "DEBUG MarchingCubesManyVolumes: " << "begin..." << endl;

    if (tsdf_ptr_list.size() != pose_ptr_list.size()) {
        cout << "you fail" << endl;
        exit(1);
    }

    // probably not necessary
    result_mesh.vertices.clear();
    result_mesh.triangles.clear();
    result_validity.clear();

    if (tsdf_ptr_list.empty()) return;

#undef min
#undef max
    // first get bounding box of all volumes
    Eigen::Array3f min_point_all = Eigen::Array3f::Constant(std::numeric_limits<float>::max());
    Eigen::Array3f max_point_all = Eigen::Array3f::Constant(-std::numeric_limits<float>::max());
    for (size_t i = 0; i < tsdf_ptr_list.size(); ++i) {
        Eigen::Array3f min_point, max_point;
        tsdf_ptr_list[i]->getBBForPose(*pose_ptr_list[i], min_point, max_point);
        min_point_all = min_point_all.min(min_point);
        max_point_all = max_point_all.max(max_point);
    }

    // could get cell sizes and block size as arguments, I suppose
    float cell_size = tsdf_ptr_list[0]->getVolumeCellSize();
    Eigen::Array3f cell_sizes(cell_size, cell_size, cell_size);
    int block_size = 64;
    Eigen::Array3i block_sizes(block_size, block_size, block_size);

    // get minimal block
    Eigen::Array3i big_block_origin = EigenUtilities::floorArray3fToInt(min_point_all / cell_size);
    Eigen::Array3i big_block_minimal_size = EigenUtilities::floorArray3fToInt((max_point_all - min_point_all) / cell_size + 1);
    Eigen::Array3i block_counts = (big_block_minimal_size - 1) / (block_sizes - 1) + 1; // might be ok for big_block_minimal_size - 2?
    Eigen::Array3i big_block_size = block_counts * (block_sizes - 1) + 1;

    // at some point here, make the temp TSDF
    // could do this once in the constructor!
    OpenCLTSDF block_tsdf(*tsdf_ptr_list[0], block_size, block_size, block_size);

    // only need one marching cubes object (no state)
    MarchingCubes<float> marching_cubes;
    MarchingCubes<float>::IDToVertexPlusT id_to_vertex_plus;
    MarchingCubes<float>::IDTriangleVector id_triangle_vector;
    MarchingCubes<float>::IDToDebugT id_debug;

    //////////////////
    // all stupid debug stuff
    cv::Mat mat_d_last_previous;
    cv::Mat mat_dw_last_previous;
    bool last_x_previous = false;
    bool pause_next_debug = false;
    float max_d_diff = 0;
    float max_dw_diff = 0;

    const static bool mean_hack_enabled = true;

    int block_counter = 0;

    // use an ordering container here for memory management
    // initialize ordering container just with grids in order
    OrderingContainer ordering_container;
    for (int i = 0; i < tsdf_ptr_list.size(); ++i) {
        ordering_container.setIDToMax(i);
    }

    for (int z_block = 0; z_block < block_counts.z(); z_block++) {
        bool last_z = z_block == block_counts.z() - 1;
        for (int y_block = 0; y_block < block_counts.y(); y_block++) {
            bool last_y = y_block == block_counts.y() - 1;
            for (int x_block = 0; x_block < block_counts.x(); x_block++, block_counter++) {
                bool last_x = x_block == block_counts.x() - 1;

                // get index offset from block_index
                Eigen::Array3i block_index(x_block, y_block, z_block);
                Eigen::Array3i block_offset = block_index * (block_sizes - 1);


                // this is wrong, but if my borders between blocks are inconsistent, this "almost" fixes it
                if (mean_hack_enabled) {
                    last_x = last_y = last_z = true;
                }

                // clear block
                block_tsdf.makeEmptyInPlace();

                // get block pose
                Eigen::Array3f translation = ((big_block_origin + block_offset).cast<float>() * cell_size);
                Eigen::Affine3f block_pose = Eigen::Affine3f::Identity();
                block_pose.translate(translation.matrix());

                // some debug info
                cout << "Doing block_offset: " << block_offset.transpose() << " of size: " << block_sizes.transpose() << " in big_block_size: " << big_block_size.transpose() << " based on big_block_minimal_size: " << big_block_minimal_size.transpose() << endl;

                // go through all blocks, get relative pose, and add if any overlap
                bool any_changed = false;
                for (size_t i = 0; i < tsdf_ptr_list.size(); ++i) {
                    Eigen::Affine3f relative_pose = pose_ptr_list[i]->inverse() * block_pose;
                    // reference other merge code:
                    //Eigen::Affine3f relative_pose = grid_to_merge.getExternalTSDFPose().inverse() * grid_target.getExternalTSDFPose();

#if 0
                    // I think this is ok, but I don't believe I use it elsewhere
                    // STOP TRUSTING THIS ONE...it totally has different behavior than addVolumeSphereTest
                    bool maybe_changed = block_tsdf.addVolumeSmart(*tsdf_ptr_list[i], relative_pose);
                    if (maybe_changed) any_changed = true;
#endif

                    // this seems to work differently than addVolumeSmart...not a good sign
                    bool maybe_changed = block_tsdf.addVolumeSphereTest(*tsdf_ptr_list[i], relative_pose);
                    if (maybe_changed) {
                        any_changed = true;
                        ordering_container.setIDToMax(i);
                    }

                    // try just adding always...slow but certain
                    // still had same problems as sphere
                    //					block_tsdf.addVolume(*tsdf_ptr_list[i], relative_pose);
                    //					any_changed = true;
                }

                // don't need to do anything if nothing fell in this block
                if (!any_changed) continue;

                std::vector<float> bufferD;
                std::vector<float> bufferDW;
                std::vector<unsigned char> bufferC;
                std::vector<float> bufferCW;
                block_tsdf.getAllBuffers(bufferD, bufferDW, bufferC, bufferCW);

                marching_cubes.generateSurfaceForBlock(bufferD.data(), bufferDW.data(), bufferC.data(), last_x, last_y, last_z, big_block_size, block_offset, block_sizes, cell_sizes, id_to_vertex_plus, id_triangle_vector, id_debug);


                // ALL DEBUG
#if 0

                // debug look at images
                {
                    ImageBuffer image_buffer_d_last(block_tsdf.getCL());
                    ImageBuffer image_buffer_dw_last(block_tsdf.getCL());
                    block_tsdf.extractSlice(0, block_size - 1, image_buffer_d_last, image_buffer_dw_last);
                    cv::Mat mat_d_last = image_buffer_d_last.getMat();
                    cv::Mat mat_dw_last = image_buffer_dw_last.getMat();

                    ImageBuffer image_buffer_d_first(block_tsdf.getCL());
                    ImageBuffer image_buffer_dw_first(block_tsdf.getCL());
                    block_tsdf.extractSlice(0, 0, image_buffer_d_first, image_buffer_dw_first);
                    cv::Mat mat_d_first = image_buffer_d_first.getMat();
                    cv::Mat mat_dw_first = image_buffer_dw_first.getMat();

#if 0
                    imshowScale("mat_d_first", mat_d_first);
                    imshowScale("mat_dw_first", mat_dw_first);
                    imshowScale("mat_d_last_previous", mat_d_last_previous);
                    imshowScale("mat_dw_last_previous", mat_dw_last_previous);
#endif

                    // only if have meaningful previous
                    if (!mat_d_last_previous.empty()) {
                        if (block_index.x() > 0) {
                            cv::Mat mat_d_diff;
                            cv::absdiff(mat_d_first, mat_d_last_previous, mat_d_diff);
                            cv::Mat mat_dw_diff;
                            cv::absdiff(mat_dw_first, mat_dw_last_previous, mat_dw_diff);
                            imshowScale("mat_d_diff", mat_d_diff);
                            imshowScale("mat_dw_diff", mat_dw_diff);
                            double min, max;
                            cv::minMaxLoc(mat_d_diff, &min, &max);
                            max_d_diff = std::max(max_d_diff, (float)fabs(max));
                            if ((float)fabs(max) > 1e-6) pause_next_debug = true;

                            cv::minMaxLoc(mat_dw_diff, &min, &max);
                            max_dw_diff = std::max(max_dw_diff, (float)fabs(max));
                        }
                    }

                    mat_d_last_previous = mat_d_last;
                    mat_dw_last_previous = mat_dw_last;
                    last_x_previous = last_x;
                }

                // debug stuff
                if (update_interface) {
                    // block outline
                    {
                        MeshVertexVectorPtr lines (new MeshVertexVector);
                        Eigen::Vector4ub color_eigen(255,255,255,255);
                        block_tsdf.getBoundingLines(block_pose, color_eigen, *lines);
                        update_interface->updateLines("mc_debug_block_outline", lines);
                    }

                    // block points
                    {
                        MeshVertexVectorPtr point_vertices(new MeshVertexVector);
                        block_tsdf.getPrettyVoxelCenters(block_pose, true, *point_vertices);
                        update_interface->updatePointCloud("mc_debug_block_points", point_vertices);
                    }
                }

                // debug pause
                if (pause_next_debug) {
                    cout << "pause_next_debug.." << endl;
                    cv::waitKey(0);
                    pause_next_debug = false;
                }
#endif

                ////////////// todo: something about the memory
                {
                    // get memory usage
                    typedef std::uint_fast64_t ByteCounterT;
                    ByteCounterT gpu_bytes_used = 0;
                    for (size_t i = 0; i < tsdf_ptr_list.size(); ++i) {
                        size_t bytes_this = tsdf_ptr_list[i]->getBytesGPUActual();
                        gpu_bytes_used += bytes_this;
                    }
                    //ByteCounterT gpu_mb_used = gpu_bytes_used >> 20;
                    ByteCounterT max_bytes_gpu = (ByteCounterT)max_mb_gpu << (ByteCounterT)20;
                    // decrease by moving block size
                    ByteCounterT moving_block_size = block_tsdf.getBytesGPUActual(); // or expected?  shouldn't matter
                    if (max_bytes_gpu > moving_block_size) max_bytes_gpu -= moving_block_size;

                    if (gpu_bytes_used >= max_bytes_gpu) {
                        cout << "MarchingCubesManyVolumes must free some GPU memory: " << gpu_bytes_used << " > " << max_bytes_gpu << endl;

                        // use the ordering container
                        int tsdfs_deallocated = 0;

                        std::map<int,int>::const_iterator ordering_begin, ordering_end;
                        ordering_container.getIterators(ordering_begin, ordering_end);

                        for (std::map<int,int>::const_iterator iter = ordering_begin; iter != ordering_end; ++iter) {
                            int which_grid = iter->second;
                            size_t bytes_this = tsdf_ptr_list[which_grid]->getBytesGPUActual();
                            if (bytes_this > 0) {
                                tsdf_ptr_list[which_grid]->setTSDFState(TSDF_STATE_RAM);
                                gpu_bytes_used -= bytes_this;
                                tsdfs_deallocated++;
                            }

                            if (gpu_bytes_used < max_bytes_gpu) {
                                cout << "Deallocated " << tsdfs_deallocated << " volumes.  Now gpu_bytes_used: " << gpu_bytes_used << endl;
                                break;
                            }
                        }
                    } // if any deallocation
                }// something about memory
            } // x
        } // y 
    } // z


    //cout << "---------------------------------------max differences d dw: " << max_d_diff << " " << max_dw_diff << endl;

    // finalize to result
    marching_cubes.finalizeMesh(id_to_vertex_plus, id_triangle_vector, id_debug, result_mesh, result_validity);

    // move into correct position
    Eigen::Affine3f final_transform = Eigen::Affine3f::Identity();
    final_transform.translate((big_block_origin.cast<float>() * cell_sizes).matrix());
    MeshUtilities::transformMeshVertices(final_transform, result_mesh.vertices);
}
