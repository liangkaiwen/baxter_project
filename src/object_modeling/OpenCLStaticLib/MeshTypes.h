#pragma once

#include "EigenUtilities.h"


struct MeshVertex {
	Eigen::Vector4f p;
	Eigen::Vector4f n;
	Eigen::Vector4ub c;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
typedef std::vector<MeshVertex, Eigen::aligned_allocator<MeshVertex> > MeshVertexVector;

typedef Eigen::Array3i Triangle;
typedef std::vector<Triangle> TriangleVector;

typedef boost::shared_ptr<MeshVertexVector> MeshVertexVectorPtr;
typedef boost::shared_ptr<TriangleVector> TriangleVectorPtr;

struct Mesh {
	MeshVertexVector vertices;
	TriangleVector triangles;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
typedef boost::shared_ptr<Mesh> MeshPtr;