#pragma once

// Author: Raghavendra Chandrashekara (basesd on source code
// provided by Paul Bourke and Cory Gene Bloyd)
// Email: rc99@doc.ic.ac.uk, rchandrashekara@hotmail.com
//
// Description: This is the interface file for the CIsoSurface class.
// CIsoSurface can be used to construct an isosurface from a scalar
// field.

// Heavily modified by peter

#include <map>
#include <vector>

#include "OpenCLTSDF.h"

#include <cstdint>


struct MarchingCubesParams {
	float max_value;

	MarchingCubesParams()
		: max_value(-1)
	{}
};


template <class T> class MarchingCubes {
public:

	typedef std::uint_fast64_t IDT;

	// types
	struct VertexPlus {
		MeshVertex v;
		unsigned int new_id;
		bool valid;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};

	typedef Eigen::Array<typename MarchingCubes<T>::IDT, 3, 1> IDTriangle;
	typedef std::vector<typename MarchingCubes<T>::IDTriangle> IDTriangleVector;
	typedef std::map<typename MarchingCubes<T>::IDT, VertexPlus> IDToVertexPlusT;

	// debug only
	struct DebugT {
		Eigen::Array3i xyz;
		Eigen::Array3i cell_offset;
		Eigen::Array3i last_bool;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	typedef std::map<IDT, DebugT> IDToDebugT;


	/////////////////////////
	MarchingCubes(const MarchingCubesParams & params = MarchingCubesParams());

	void generateSurface(const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField, const Eigen::Array3i & num_cells, const Eigen::Array3f & cell_sizes, Mesh & result_mesh, std::vector<bool> & result_valid);

	// broken out so you can build up a single mesh
	void generateSurfaceForBlock(const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField,
		bool last_x, bool last_y, bool last_z, const Eigen::Array3i &global_num_cells,
		const Eigen::Array3i & cell_offset, const Eigen::Array3i & num_cells, const Eigen::Array3f & cell_sizes,
		IDToVertexPlusT & id_to_vertex_plus, IDTriangleVector & id_triangle_list, IDToDebugT & id_debug);

	void finalizeMesh(IDToVertexPlusT & id_to_vertex_plus, IDTriangleVector const& id_triangle_vector, IDToDebugT & id_debug, Mesh & result_mesh, std::vector<bool> & result_valid);


protected:

	// Returns the edge ID.
	IDT getEdgeID(const Eigen::Array3i & global_num_cells, const Eigen::Array3i & cell_offset, unsigned int nX, unsigned int nY, unsigned int nZ, unsigned int nEdgeNo);

	// Returns the vertex ID.
	IDT getVertexID(const Eigen::Array3i & global_num_cells, const Eigen::Array3i & cell_offset, unsigned int nX, unsigned int nY, unsigned int nZ);

	// Calculates the intersection point of the isosurface with an
	// edge.
	VertexPlus calculateIntersection(
		const T* ptScalarField, const float * ptWeightField, const unsigned char * ptColorField,
		const Eigen::Array3i & num_cells, const Eigen::Array3i & cell_offset, const Eigen::Array3f & cell_sizes,
		unsigned int nX, unsigned int nY, unsigned int nZ, unsigned int nEdgeNo);

	// Interpolates between two grid points to produce the point at which
	// the isosurface intersects an edge.
	Eigen::Vector4f interpolate(Eigen::Vector4f const& p1, Eigen::Vector4f const& p2, T tVal1, T tVal2);

	// similar for color
	Eigen::Array4ub interpolateColor(Eigen::Vector4f const& p1, Eigen::Vector4f const& p2, Eigen::Vector4f const& p_interpolated, Eigen::Array4ub const& c1, Eigen::Array4ub const& c2);

	// Renames vertices and triangles so that they can be accessed better
	void renameVerticesAndTriangles(IDToVertexPlusT & id_to_vertex_plus, const IDTriangleVector & id_triangle_vector, const IDToDebugT & id_debug, Mesh & result_mesh, std::vector<bool> & result_valid);

	// Calculates the normals.
	void calculateNormals(Mesh &input_and_result_mesh);



	// Lookup tables used in the construction of the isosurface.
	static const unsigned int m_edgeTable[256];
	static const unsigned int m_triTable[256][16];




	// Members

	MarchingCubesParams params_;
};


