#pragma once

#include <vector>

#include "EigenUtilities.h"

#include "MeshTypes.h"

#include "VolumeBuffer.h"

namespace MeshUtilities {

	MeshPtr generateMesh(
		const VolumeBuffer & volume_distance,
		const VolumeBuffer & volume_weights,
		const float cell_size,
		const Eigen::Array4ub & color,
		const Eigen::Affine3f & model_pose);

	void generateMesh(
		const std::vector<float> & bufferD,
		const std::vector<float> & bufferDW,
		const std::vector<unsigned char> & bufferC,
		const Eigen::Array3i & volume_cell_counts,
		const float & cell_size,
		MeshVertexVector & vertices,
		TriangleVector & triangles);

	void generateMeshAndValidity(
		const std::vector<float> & bufferD,
		const std::vector<float> & bufferDW,
		const std::vector<unsigned char> & bufferC,
		const Eigen::Array3i & volume_cell_counts,
		const float & cell_size,
		MeshVertexVector & vertices,
		TriangleVector & triangles,
		std::vector<bool> & vertex_validity);

	void extractValidVerticesAndTriangles(
		const MeshVertexVector & all_vertices,
		const TriangleVector & all_triangles,
		const std::vector<bool> & vertex_validity, 
		MeshVertexVector & valid_vertices,
		TriangleVector & valid_triangles);

    void extractVerticesAndTriangles(const Mesh & input_mesh,
            const std::vector<bool> & bool_vector,
            const bool value_to_keep,
            Mesh & result_mesh);

	void getTriangleValidity(
		const MeshVertexVector & all_vertices,
		const TriangleVector & all_triangles,
		const std::vector<bool> & vertex_validity,
		std::vector<bool> & triangle_validity);

	bool saveMesh(
		const MeshVertexVector & vertices, 
		const TriangleVector & triangles,
		const fs::path & filename,
		bool color = true);

	void appendMesh(
		MeshVertexVector & vertex_list,
		TriangleVector & triangle_list,
		const MeshVertexVector & vertex_list_to_append,
		const TriangleVector & triangle_list_to_append);

	void transformMeshVertex(
		const Eigen::Affine3f & pose,
		MeshVertex & vertex);

	void transformMeshVertices(
		const Eigen::Affine3f & pose,
		MeshVertexVector & vertices);

	bool checkMeshValid(
		const MeshVertexVector & vertices,
		const TriangleVector & triangles);

	void setAllVerticesColor(
		MeshVertexVector & vertices,
		const Eigen::Array4ub & color);

    void getBoundaryEdges(const Mesh & mesh, std::vector<std::pair<int,int> > & result_edges);

        void appendLine(MeshVertexVector & vertices, const Eigen::Vector4f & p1, const Eigen::Vector4f & p2, const Eigen::Vector4ub & c);

} // ns
