#pragma once

#include "EigenUtilities.h" // ?

#include <fstream>

namespace EigenUtilities {

template <typename Real>
Eigen::Matrix<Real,7,1> convertTransformToVector(Eigen::Transform<Real,3,Eigen::Affine> const& t)
{
	Eigen::Quaternion<Real> q = Eigen::Quaternion<Real>(t.linear());
	Eigen::Matrix<Real,3,1> translation = t.translation();

	Eigen::Matrix<Real,7,1> vector;
	// old way (pre-freiburg)
#if 0
	vector[0] = q.w();
	vector[1] = q.x();
	vector[2] = q.y();
	vector[3] = q.z();
	vector[4] = translation.x();
	vector[5] = translation.y();
	vector[6] = translation.z();
#endif
	// new freiburg way
	vector[0] = translation.x();
	vector[1] = translation.y();
	vector[2] = translation.z();
	vector[3] = q.w();
	vector[4] = q.x();
	vector[5] = q.y();
	vector[6] = q.z();

	return vector;
}

template <typename Real>
Eigen::Transform<Real,3,Eigen::Affine> convertVectorToTransform(Eigen::Matrix<Real,7,1> const& v)
{
#if 0
	Eigen::Quaternion<Real> q(v[0],v[1],v[2],v[3]);
	Eigen::Matrix<Real,3,1> t(v[4],v[5],v[6]);
#endif
	// freiburg way:
	Eigen::Matrix<Real,3,1> t(v[0],v[1],v[2]);
	Eigen::Quaternion<Real> q(v[3],v[4],v[5],v[6]);
	
	Eigen::Transform<Real,3,Eigen::Affine> transform(q.normalized().toRotationMatrix());
	transform.pretranslate(t);

	return transform;
}

template <typename T>
bool writeVector(const std::vector<T>& v, fs::path p)
{
	std::fstream file;
	file.open (p.string().c_str(), std::ios::out | std::ios::binary);
	if (!file) return false;
	bool io_result = file.write ((char*)v.data(), v.size() * sizeof(T));
	file.close();
	return io_result;
}

template <typename T>
bool readVector(std::vector<T>& v, fs::path p)
{
	std::fstream file;
	file.open (p.string().c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (!file) return false;
	std::fstream::pos_type size = file.tellg();
	v.resize(size / sizeof(T));
	if (size > 0) {
		file.seekg (0, std::ios::beg);
		bool io_result = file.read ((char*)v.data(), size);
		file.close();
		return io_result;
	}
	else {
		return true; // empty
	}
}

} // ns