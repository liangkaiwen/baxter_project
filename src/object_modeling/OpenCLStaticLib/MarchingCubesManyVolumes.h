#pragma once

#include <vector>

#include "OpenCLTSDF.h"

// this is ugly, depends back volume modeler for now
#include "update_interface.h"


class MarchingCubesManyVolumes {
private:
	// all methods static...should probably just be a namespace
	MarchingCubesManyVolumes();

public:
	static void generateMeshAndValidity(std::vector<OpenCLTSDFPtr> & tsdf_ptr_list, std::vector<boost::shared_ptr<Eigen::Affine3f> > const& pose_ptr_list,
		boost::shared_ptr<UpdateInterface> update_interface, size_t max_mb_gpu, 
		Mesh & result_mesh, std::vector<bool> & result_validity);

};