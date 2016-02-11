#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/tuple/tuple.hpp>

namespace Eigen {
	typedef Eigen::Matrix<uint8_t, 4, 1> Vector4ub; 
	typedef Eigen::Array<uint8_t, 4, 1> Array4ub; 
} // Eigen NS

// utility typedefs
typedef boost::shared_ptr<Eigen::Affine3f> PosePtr;
typedef std::vector<PosePtr> PosePtrList;

namespace EigenUtilities {

	std::string transformToString(const Eigen::Affine3f& x);
	Eigen::Affine3f stringToTransform(const std::string& s);
	Eigen::Isometry3d getIsometry3d(const Eigen::Affine3f& t);
	Eigen::Affine3f getAffine3f(const Eigen::Isometry3d& i);

	template <typename Real>
	Eigen::Matrix<Real,7,1> convertTransformToVector(Eigen::Transform<Real,3,Eigen::Affine> const& t);

	template <typename Real>
	Eigen::Transform<Real,3,Eigen::Affine> convertVectorToTransform(Eigen::Matrix<Real,7,1> const& v);

	template <typename T>
	bool writeVector(const std::vector<T>& v, fs::path p);

	template <typename T>
	bool readVector(std::vector<T>& v, fs::path p);

	// angle is in degrees
	void getAngleAndDistance(const Eigen::Affine3f& t, float & angle, float & distance);
	void getCameraPoseDifference(const Eigen::Affine3f& p1, const Eigen::Affine3f& p2, float & angle, float & distance);

	Eigen::Vector3i truncateVector3fToInt(const Eigen::Vector3f& float_v);
	Eigen::Vector3i roundPositiveVector3fToInt(const Eigen::Vector3f& float_v);

	Eigen::Array3i truncateArray3fToInt(Eigen::Array3f const& float_a);
	Eigen::Array3i floorArray3fToInt(Eigen::Array3f const& float_a);

	boost::tuple<int,int,int> array3iToTuple(Eigen::Array3i const& array_3i);
	Eigen::Array3i tupleToArray3i(boost::tuple<int,int,int> const& array_3i);

	void loadPosesFromFile(fs::path filename, PosePtrList & result);

	Eigen::Vector3f depthToPoint(const Eigen::Array2f & camera_focal, const Eigen::Array2f & camera_center, const Eigen::Array2i & pixel, float depth);

} // EigenUtilities ns

// try to make eigen work with boost serialize
// just lazy and throw it here in the header
namespace boost
{
	template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	inline void serialize(
		Archive & ar, 
		Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t, 
		const unsigned int file_version
		) 
	{
		size_t rows = t.rows(), cols = t.cols();
		ar & rows;
		ar & cols;
		if( rows * cols != t.size() )
			t.resize( rows, cols );

		for(size_t i=0; i<t.size(); i++)
			ar & t.data()[i];
	}
}

#include "EigenUtilities.hpp"
