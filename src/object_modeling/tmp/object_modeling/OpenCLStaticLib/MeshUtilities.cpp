#include "MeshUtilities.h"

#include <boost/foreach.hpp>

#include "MarchingCubes.h"
#include "rply.h"

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace MeshUtilities {

	// utility version for depth-only from volumebuffers
	// could also modify reference args I suppose...
	MeshPtr generateMesh(
		const VolumeBuffer & volume_distance,
		const VolumeBuffer & volume_weights,
		const float cell_size,
		const Eigen::Array4ub & color,
		const Eigen::Affine3f & model_pose)
	{
		MeshPtr mesh(new Mesh);

		std::vector<float> volume_distance_vector(volume_distance.getSizeInCells());
		volume_distance.getBufferWrapper().readToFloatVector(volume_distance_vector);
		std::vector<float> volume_weights_vector(volume_weights.getSizeInCells());
		volume_weights.getBufferWrapper().readToFloatVector(volume_weights_vector);
		std::vector<uint8_t> empty_color;
		MeshUtilities::generateMesh(volume_distance_vector, volume_weights_vector, empty_color, volume_distance.getVolumeCellCounts(), cell_size, mesh->vertices, mesh->triangles);

		// don't have to do these here, but it's fun
		MeshUtilities::transformMeshVertices(model_pose, mesh->vertices);
		MeshUtilities::setAllVerticesColor(mesh->vertices, color);

		return mesh;
	}

	void generateMesh(
		const std::vector<float> & bufferD,
		const std::vector<float> & bufferDW,
		const std::vector<unsigned char> & bufferC,
		const Eigen::Array3i & volume_cell_counts,
		const float & cell_size,
		MeshVertexVector & vertices,
		TriangleVector & triangles)
	{
		MeshVertexVector all_vertices;
		TriangleVector all_triangles;
		std::vector<bool> vertex_validity;
		generateMeshAndValidity(bufferD, bufferDW, bufferC, volume_cell_counts, cell_size, all_vertices, all_triangles, vertex_validity);
		extractValidVerticesAndTriangles(all_vertices, all_triangles, vertex_validity, vertices, triangles);
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
		MarchingCubes<float> marching_cubes;
		Mesh mesh;
		// note that new one takes actual volume_cell_counts as argument, not "-1"
		const unsigned char* color_ptr = NULL;
		if (!bufferC.empty()) color_ptr = bufferC.data();
		marching_cubes.generateSurface(bufferD.data(), bufferDW.data(), color_ptr, volume_cell_counts, Eigen::Array3f(cell_size, cell_size, cell_size), mesh, vertex_validity);

        // useless copy (this function should take Mesh, not both separately)
		vertices = mesh.vertices;
		triangles = mesh.triangles;
	}

    // deprecated..should use extractVerticesAndTriangles
	void extractValidVerticesAndTriangles(
		const MeshVertexVector & all_vertices,
		const TriangleVector & all_triangles,
		const std::vector<bool> & vertex_validity, 
		MeshVertexVector & valid_vertices,
		TriangleVector & valid_triangles)
	{
		valid_vertices.clear();
		std::map<uint32_t, uint32_t> vertex_map;
		for (int i = 0; i < (int)all_vertices.size(); i++) {
			if (vertex_validity[i]) {
				valid_vertices.push_back(all_vertices[i]);
				vertex_map[i] = valid_vertices.size() - 1;
			}
		}

		// remap and add valid triangles
		valid_triangles.clear();
		for (int i = 0; i < (int)all_triangles.size(); i++) {
			Eigen::Array3i const& old_triangle = all_triangles[i];
			if (vertex_validity[old_triangle[0]] &&
				vertex_validity[old_triangle[1]] &&
				vertex_validity[old_triangle[2]]) {
					valid_triangles.push_back(Eigen::Array3i());
					valid_triangles.back()[0] = vertex_map[old_triangle[0]];
					valid_triangles.back()[1] = vertex_map[old_triangle[1]];
					valid_triangles.back()[2] = vertex_map[old_triangle[2]];
			}
		}
	}

    void extractVerticesAndTriangles(
        const Mesh & input_mesh,
        const std::vector<bool> & bool_vector,
        const bool value_to_keep,
        Mesh & result_mesh)
    {
        result_mesh.vertices.clear();
        result_mesh.triangles.clear();

        std::map<uint32_t, uint32_t> vertex_map;
        for (int i = 0; i < (int)input_mesh.vertices.size(); i++) {
            if (bool_vector[i] == value_to_keep) {
                result_mesh.vertices.push_back(input_mesh.vertices[i]);
                vertex_map[i] = result_mesh.vertices.size() - 1;
            }
        }

        // remap and add valid triangles
        for (int i = 0; i < (int)input_mesh.triangles.size(); i++) {
            Eigen::Array3i const& old_triangle = input_mesh.triangles[i];
            if (bool_vector[old_triangle[0]] == value_to_keep &&
                bool_vector[old_triangle[1]] == value_to_keep &&
                bool_vector[old_triangle[2]] == value_to_keep) {
                    Eigen::Array3i remapped_triangle(vertex_map[old_triangle[0]], vertex_map[old_triangle[1]], vertex_map[old_triangle[2]]);
                    result_mesh.triangles.push_back(remapped_triangle);
            }
        }
    }

	void getTriangleValidity(
		const MeshVertexVector & all_vertices,
		const TriangleVector & all_triangles,
		const std::vector<bool> & vertex_validity,
		std::vector<bool> & triangle_validity)
	{
		triangle_validity.resize(all_triangles.size());
		for (int i = 0; i < (int)all_triangles.size(); i++) {
			Eigen::Array3i const& old_triangle = all_triangles[i];
			triangle_validity[i] = (vertex_validity[old_triangle[0]] && vertex_validity[old_triangle[1]] && vertex_validity[old_triangle[2]]);
		}
	}



	bool saveMesh(
		const MeshVertexVector & vertices, 
		const TriangleVector & triangles,
		const fs::path & filename,
		bool color)
	{
		p_ply ply_file = ply_create(filename.string().c_str(), PLY_LITTLE_ENDIAN, NULL, 0, NULL);
		if (!ply_file) return false;
		// can do additional error checking on all of these...but why...
		ply_add_element(ply_file, "vertex", vertices.size());
		// last 2 args ignored
		ply_add_scalar_property(ply_file, "x", PLY_FLOAT32);
		ply_add_scalar_property(ply_file, "y", PLY_FLOAT32);
		ply_add_scalar_property(ply_file, "z", PLY_FLOAT32);
		ply_add_scalar_property(ply_file, "normal_x", PLY_FLOAT32);
		ply_add_scalar_property(ply_file, "normal_y", PLY_FLOAT32);
		ply_add_scalar_property(ply_file, "normal_z", PLY_FLOAT32);
		if (color) {
			ply_add_scalar_property(ply_file, "red", PLY_UCHAR);
			ply_add_scalar_property(ply_file, "green", PLY_UCHAR);
			ply_add_scalar_property(ply_file, "blue", PLY_UCHAR);
		}
		ply_add_element(ply_file, "face", triangles.size());
		ply_add_property(ply_file, "vertex_index", PLY_LIST, PLY_UCHAR, PLY_INT32);

		ply_write_header(ply_file);
		for (size_t i = 0; i < vertices.size(); ++i) {
			MeshVertex const& v = vertices[i];
			ply_write(ply_file, v.p.x());
			ply_write(ply_file, v.p.y());
			ply_write(ply_file, v.p.z());
			ply_write(ply_file, v.n.x());
			ply_write(ply_file, v.n.y());
			ply_write(ply_file, v.n.z());
			if (color) {
				ply_write(ply_file, v.c[2]);
				ply_write(ply_file, v.c[1]);
				ply_write(ply_file, v.c[0]);
			}
		}
		for (size_t i = 0; i < triangles.size(); ++i) {
			Eigen::Array3i const& t = triangles[i];
			ply_write(ply_file, 3);
			ply_write(ply_file, t[0]);
			ply_write(ply_file, t[1]);
			ply_write(ply_file, t[2]);
		}
		ply_close(ply_file);

		return true;
	}

	void appendMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list,
		const MeshVertexVector & vertex_list_to_append,
		const TriangleVector & triangle_list_to_append)
	{
		size_t vertex_offset = vertex_list.size();

		// need to add on transformed vertices
		for (size_t v = 0; v < vertex_list_to_append.size(); ++v) {
			vertex_list.push_back(vertex_list_to_append[v]);
		}

		// and offset triangles
		for (size_t j = 0; j < triangle_list_to_append.size(); ++j) {
			triangle_list.push_back(triangle_list_to_append[j] + vertex_offset);
		}
	}

	void transformMeshVertex(
		const Eigen::Affine3f & pose,
		MeshVertex & vertex)
	{
		vertex.p = pose * vertex.p;
		vertex.n = pose * vertex.n;
	}

	void transformMeshVertices(
		const Eigen::Affine3f & pose,
		MeshVertexVector & vertices)
	{
		BOOST_FOREACH(MeshVertex & v, vertices) {
			transformMeshVertex(pose, v);
		}
	}

	bool checkMeshValid(
		const MeshVertexVector & vertices,
		const TriangleVector & triangles)
	{
		BOOST_FOREACH(const Triangle & t, triangles) {
			if ( (t < 0).any() || (t >= vertices.size()).any() ) return false;
		}
		return true;
	}

	void setAllVerticesColor(
		MeshVertexVector & vertices,
		const Eigen::Array4ub & color)
	{
		BOOST_FOREACH(MeshVertex & v, vertices) {
			v.c = color;
		}
	}

    void getBoundaryEdges(const Mesh & mesh, std::vector<std::pair<int,int> > & result_edges)
    {
        typedef std::map<std::pair<int,int>, int >  EdgeCountMap;
        EdgeCountMap edge_counts;
        BOOST_FOREACH(const Triangle & t, mesh.triangles) {
            for (size_t i = 0; i < 3; ++i) {
                int v1 = t[i];
                int v2 = i < 2 ? t[i+1] : t[0];
                std::pair<int,int> edge;
                if (v1 < v2) edge = std::make_pair(v1,v2);
                else edge = std::make_pair(v2,v1);
                edge_counts[edge]++;
            }
        }

        result_edges.clear();
        BOOST_FOREACH(const EdgeCountMap::value_type & v, edge_counts) {
            if (v.second == 1) result_edges.push_back(v.first);
        }
    }

    void appendLine(MeshVertexVector & vertices, const Eigen::Vector4f & p1, const Eigen::Vector4f & p2, const Eigen::Vector4ub & c)
    {
        vertices.push_back(MeshVertex());
        vertices.back().p = p1;
        vertices.back().c = c;
        vertices.push_back(MeshVertex());
        vertices.back().p = p2;
        vertices.back().c = c;
    }



} // ns
