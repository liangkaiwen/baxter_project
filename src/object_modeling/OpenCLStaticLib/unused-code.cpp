	// original validity:
#if 0
	//#pragma omp parallel for // doesn't help much
	// also this one causes problems maybe on linux?? but why??
	for (int i = 0; i < (int)all_vertices.size(); i++) {
		MeshVertex const& v = all_vertices[i];
		Eigen::Vector3f voxel_float = v.p.head<3>() / cell_size;
		vertex_validity[i] = allSurroundingVerticesNonzeroOrOutsideVolume(volume_cell_counts, bufferDW, voxel_float);
	}
	if (verbose) cout << "time: " << t.elapsed() << endl;
#endif	

// alternate validity hack:
#if 0
	// alternative method for validity
	for (int i = 0; i < (int)all_vertices.size(); i++) {
		MeshVertex const& v = all_vertices[i];
		Eigen::Vector3f voxel_float = v.p.head<3>() / cell_size;
		// determine which axis this voxel was on
		// could be faster I'm sure...
		Eigen::Vector3i voxel_round = roundPositiveVector3fToInt(voxel_float);
		Eigen::Array3f voxel_diffs = (voxel_float - voxel_round.cast<float>()).array().abs();
		int max_index = -1;
		float max_diff = voxel_diffs.maxCoeff(&max_index);

		//cout << "voxel_diffs: " << voxel_diffs.transpose() << " --- max_index: " << max_index << endl;

		// now use this axis...
		// this is weird, but so is not using this:
		//const float epsilon = 1e-6;
		//Eigen::Array3i voxel_floor = floorVector3fToInt(voxel_float + Eigen::Vector3f(epsilon, epsilon, epsilon)).array();
		Eigen::Array3i voxel_floor = floorVector3fToInt(voxel_float);
		
#if 0
				bool valid = true;
		// when I didn't trust my code:
		// really shouldn't happen
		if ( (voxel_floor < 0).any() || (voxel_floor > volume_cell_counts - 1).any() ) throw std::runtime_error("unexpected out of volume");

		if ( (voxel_floor < 0).any() || (voxel_floor > volume_cell_counts - 1).any() ) valid = false;
		else if (bufferDW[getBufferIndex(volume_cell_counts, voxel_floor)] < 0.f) valid = false;
		if (valid) {
			// also check + along axis
			Eigen::Array3i axis_add = Eigen::Array3i::Zero();
			axis_add[max_index] = 1;
			Eigen::Array3i voxel_axis = voxel_floor + axis_add;

			// really shouldn't happen
			if ( (voxel_axis < 0).any() || (voxel_axis > volume_cell_counts - 1).any() ) throw std::runtime_error("unexpected out of volume");
			
			if ( (voxel_axis < 0).any() || (voxel_axis > volume_cell_counts - 1).any() ) valid = false;
			else if (bufferDW[getBufferIndex(volume_cell_counts, voxel_axis)] < 0.f) valid = false;
		}
				vertex_validity[i] = valid;

#endif
		Eigen::Array3i axis_add = Eigen::Array3i::Zero();
		axis_add[max_index] = 1;
		Eigen::Array3i voxel_axis = voxel_floor + axis_add;

		float weight_floor = bufferDW[getBufferIndex(volume_cell_counts, voxel_floor)];
		float weight_axis = bufferDW[getBufferIndex(volume_cell_counts, voxel_axis)];

		//cout << "weights: " << weight_floor << "," << weight_axis << endl;

		vertex_validity[i] = weight_floor > 0 && weight_axis > 0;

		//cout << "vertex_validity[i]: " << vertex_validity[i] << endl;

	}
	if (verbose) cout << "time: " << t.elapsed() << endl;
#endif