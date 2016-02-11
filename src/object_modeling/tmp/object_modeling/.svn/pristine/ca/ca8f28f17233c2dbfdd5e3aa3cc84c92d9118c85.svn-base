#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <Eigen/StdVector>
#include <vector>

class Frustum
{
private:
	Frustum(const Frustum& other);

public:
	Frustum(Eigen::Affine3f const& camera_pose, Eigen::Vector2f const& resolution, Eigen::Vector2f const& proj_f, Eigen::Vector2f const& proj_c, float z_min, float z_max);

	std::vector<Eigen::Vector3f> const& getPoints() const {return points;}

	bool isPointInside(Eigen::Vector3f const& p);
	bool doesAABBIntersect(Eigen::Vector3f const& p_min, Eigen::Vector3f const& p_max);

protected:
	// assumes base point, first ray end, second ray end (in right hand rule for the 2 rays)
	Eigen::Vector4f planeFromPoints(Eigen::Vector3f const& point0, Eigen::Vector3f const& point1, Eigen::Vector3f const& point2);

	Eigen::Vector3f pointFromPixel(Eigen::Vector2f const& pixel, float z, Eigen::Vector2f const& proj_f, Eigen::Vector2f const& proj_c);

	std::vector<Eigen::Vector3f> cornersFromAABB(Eigen::Vector3f const& bb_min, Eigen::Vector3f const& bb_max);

	// members

	// binary order 000, 001, 010, etc...
	std::vector<Eigen::Vector3f> points;

	// near, far, left, right, bottom, top
	std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > planes;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

