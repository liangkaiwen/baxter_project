
// static
bool OpenCLTSDF::checkSurroundingVoxelsAreWithinVolume(const Eigen::Array3i & volume_cell_counts, const Eigen::Vector3f& voxel_coords_f)
{
	Eigen::Array3i floor_corner = EigenUtilities::truncateVector3fToInt(voxel_coords_f).array();
	if ( (floor_corner < 0).any() || (floor_corner > volume_cell_counts - 2).any() ) return false;
	else return true;
}

// static
bool OpenCLTSDF::allSurroundingVerticesNonzeroOrOutsideVolume(const Eigen::Array3i & volume_cell_counts, const std::vector<float> & weight_vector, const Eigen::Vector3f& voxel_coords_f)
{
    // for meshing...for a point from marching cubes
    if (!checkSurroundingVoxelsAreWithinVolume(volume_cell_counts, voxel_coords_f)) return true;

    Eigen::Vector3i floor_corner = EigenUtilities::truncateVector3fToInt( voxel_coords_f );

    const float w_000 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0], floor_corner[1], floor_corner[2]))];
    const float w_001 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0], floor_corner[1], floor_corner[2]+1))];
    const float w_010 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0], floor_corner[1]+1, floor_corner[2]))];
    const float w_011 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0], floor_corner[1]+1, floor_corner[2]+1))];
    const float w_100 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0]+1, floor_corner[1], floor_corner[2]))];
    const float w_101 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0]+1, floor_corner[1], floor_corner[2]+1))];
    const float w_110 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2]))];
    const float w_111 = weight_vector[getVolumeIndex(volume_cell_counts, Eigen::Array3i(floor_corner[0]+1, floor_corner[1]+1, floor_corner[2]+1))];

    bool result = (
        (w_000 > 0) &&
        (w_001 > 0) &&
        (w_010 > 0) &&
        (w_011 > 0) &&
        (w_100 > 0) &&
        (w_101 > 0) &&
        (w_110 > 0) &&
        (w_111 > 0) );
    return result;
}


// static
Eigen::Matrix<uint8_t, 4, 1> OpenCLTSDF::interpolateColorForMesh(const Eigen::Array3i & volume_cell_counts, const std::vector<unsigned char> & color_vector, const Eigen::Vector3f& voxel_coords_f)
{
    // todo: replace with arrayu4b
	typedef Eigen::Matrix<unsigned char, 4, 1> Vector4b;
	const static Eigen::Vector4f add_round(0.5,0.5,0.5,0.5);
	const static Eigen::Array4f min_array(0,0,0,0);
	const static Eigen::Array4f max_array(255,255,255,255);

	//if (!checkSurroundingVoxelsAreWithinVolume(voxel_coords_f)) return Vector4b(0,0,0,0);

	Eigen::Vector3i fc = EigenUtilities::truncateVector3fToInt(voxel_coords_f);

	// set to 1 if inside volume
	// This will allow an adjustment for < 8 valid vertices
	float w_000 = 0;
	float w_001 = 0;
	float w_010 = 0;
	float w_011 = 0;
	float w_100 = 0;
	float w_101 = 0;
	float w_110 = 0;
	float w_111 = 0;

	Vector4b d_000;
	Vector4b d_001;
	Vector4b d_010;
	Vector4b d_011;
	Vector4b d_100;
	Vector4b d_101;
	Vector4b d_110;
	Vector4b d_111;

	Eigen::Array3i v;
	v = Eigen::Array3i(fc[0], fc[1], fc[2]);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_000 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_000 = 1;
	}
	v = Eigen::Array3i(fc[0], fc[1], fc[2]+1);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_001 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_001 = 1;
	}
	v = Eigen::Array3i(fc[0], fc[1]+1, fc[2]);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_010 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_010 = 1;
	}
	v = Eigen::Array3i(fc[0], fc[1]+1, fc[2]+1);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_011 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_011 = 1;
	}
	v = Eigen::Array3i(fc[0]+1, fc[1], fc[2]);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_100 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_100 = 1;
	}
	v = Eigen::Array3i(fc[0]+1, fc[1], fc[2]+1);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_101 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_101 = 1;
	}
	v = Eigen::Array3i(fc[0]+1, fc[1]+1, fc[2]);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_110 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_110 = 1;
	}
	v = Eigen::Array3i(fc[0]+1, fc[1]+1, fc[2]+1);
	if (isVertexInVolume(volume_cell_counts, v)) {
		d_111 = Vector4b::Map(&color_vector[4 * getVolumeIndex(volume_cell_counts, v)]);
		w_111 = 1;
	}

	Eigen::Vector3f offset = voxel_coords_f - fc.cast<float>();
	float off_x = offset[0];
	float off_y = offset[1];
	float off_z = offset[2];

	Eigen::Vector4f result_float(0,0,0,0);
	w_000 *= (1 - off_x) * (1 - off_y) * (1 - off_z);
	w_001 *= (1 - off_x) * (1 - off_y) * (off_z);
	w_010 *= (1 - off_x) * (off_y) * (1 - off_z);
	w_011 *= (1 - off_x) * (off_y) * (off_z);
	w_100 *= (off_x) * (1 - off_y) * (1 - off_z);
	w_101 *= (off_x) * (1 - off_y) * (off_z);
	w_110 *= (off_x) * (off_y) * (1 - off_z);
	w_111 *= (off_x) * (off_y) * (off_z);

	result_float += ( w_000 * d_000.cast<float>()
					+ w_001 * d_001.cast<float>()
					+ w_010 * d_010.cast<float>()
					+ w_011 * d_011.cast<float>()
					+ w_100 * d_100.cast<float>()
					+ w_101 * d_101.cast<float>()
					+ w_110 * d_110.cast<float>()
					+ w_111 * d_111.cast<float>() );
	// correct for missing weights
	float weight_sum = w_000
					 + w_001
					 + w_010
					 + w_011
					 + w_100
					 + w_101
					 + w_110
					 + w_111;
	if (weight_sum > 0) {
		result_float *= 1.f / weight_sum;
	}

	Eigen::Vector4f result_to_cast = result_float + add_round;
	result_to_cast = result_to_cast.array().max(min_array).matrix();
	result_to_cast = result_to_cast.array().min(max_array).matrix();
	Vector4b result = result_to_cast.cast<unsigned char>();

	return result;
}



	void generateMeshAndValidity(
		const std::vector<float> & bufferD,
		const std::vector<float> & bufferDW,
		const std::vector<unsigned char> & bufferC,
		const Eigen::Array3i & volume_cell_counts,
		const float & cell_size,
		MeshVertexVector & vertices,
		TriangleVector & triangles,
		std::vector<bool> & vertex_validity)
	{
		// switch to new marching cubes
#if 1
		MarchingCubes<float> marching_cubes;
		Mesh mesh;
		// note that new one takes actual volume_cell_counts as argument, not "-1"
		const unsigned char* color_ptr = NULL;
		if (!bufferC.empty()) color_ptr = bufferC.data();
		marching_cubes.generateSurface(bufferD.data(), bufferDW.data(), color_ptr, volume_cell_counts, Eigen::Array3f(cell_size, cell_size, cell_size), mesh, vertex_validity);

		// does colors inside now
#if 0
		// add color (should be included in surface!)
#pragma omp parallel for
		for (int i = 0; i < (int)mesh.vertices.size(); ++i) {
			MeshVertex & v = mesh.vertices[i];
			Eigen::Vector3f voxel_float = v.p.head<3>() / cell_size;
			v.c = interpolateColorForMesh(volume_cell_counts, bufferC, voxel_float);
		}
#endif

		// useless copy (this function should take "Mesh", not both separately)
		vertices = mesh.vertices;
		triangles = mesh.triangles;

#else

		if (verbose) cout << "Running marching cubes..." << endl;
		CIsoSurface<float> iso_surface;
		iso_surface.GenerateSurface(bufferD.data(), bufferDW.data(), 0, volume_cell_counts.x() - 1, volume_cell_counts.y() - 1, volume_cell_counts.z() - 1, cell_size, cell_size, cell_size);
		if (verbose) cout << "time: " << t.elapsed() << endl;
		t.restart();


		if (verbose) cout << "Placing result in return variables and assigning colors..." << endl;
		all_vertices.resize(iso_surface.m_nVertices);
		// from 66ms to 22ms or so
#pragma omp parallel for
		for (int i = 0; i < (int)all_vertices.size(); ++i) {
			MeshVertex & v = all_vertices[i];
			v.p.head<3>() = Eigen::Vector3f::Map(iso_surface.m_ppt3dVertices[i]);
			v.p[3] = 1;
			v.n.head<3>() = Eigen::Vector3f::Map(iso_surface.m_pvec3dNormals[i]);
			v.n[3] = 1;
			Eigen::Vector3f voxel_float = v.p.head<3>() / cell_size;
			v.c = interpolateColorForMesh(volume_cell_counts, bufferC, voxel_float);
		}

		all_triangles.resize(iso_surface.m_nTriangles);
		//#pragma omp parallel for // doesn't help
		for (int i = 0; i < (int)all_triangles.size(); ++i) {
			all_triangles[i] = Eigen::Array3i::Map ((int*)&iso_surface.m_piTriangleIndices[3*i]);
		}
		if (verbose) cout << "time: " << t.elapsed() << endl;
		t.restart();


		if (verbose) cout << "Setting vertex validity..." << endl;
		vertex_validity.resize(all_vertices.size());

		// now we have validity from the surface!
		for (int i = 0; i < vertex_validity.size(); ++i) {
			vertex_validity[i] = iso_surface.m_pboolValidity[i];
		}
		if (verbose) cout << "time: " << t.elapsed() << endl;

		if (verbose) cout << "time overall: " << t_overall.elapsed() << endl;

#endif

	}
