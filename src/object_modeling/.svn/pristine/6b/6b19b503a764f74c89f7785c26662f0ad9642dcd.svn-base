#pragma once

#include "EigenUtilities.h"

enum EdgeStructEdgeType {
    EDGE_STRUCT_DEFAULT,
    EDGE_STRUCT_EXISTING_KEYFRAME,
    EDGE_STRUCT_LOOP_CLOSURE,
    EDGE_STRUCT_PLACE_RECOGNITION
};

struct EdgeStruct {
	EdgeStruct()
		: index_0(-1),
		index_1(-1),
		relative_pose(Eigen::Affine3f::Identity()),
		edge_type(EDGE_STRUCT_DEFAULT)
	{}

	EdgeStruct(int index_0, int index_1, Eigen::Affine3f const& relative_pose, EdgeStructEdgeType edge_type)
		: index_0(index_0),
		index_1(index_1),
		relative_pose(relative_pose),
		edge_type(edge_type)
	{}

	template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
		ar & index_0;
		ar & index_1;
		ar & relative_pose.matrix();
        ar & edge_type;
	}


	int index_0;
	int index_1;
	Eigen::Affine3f relative_pose;
    EdgeStructEdgeType edge_type;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};