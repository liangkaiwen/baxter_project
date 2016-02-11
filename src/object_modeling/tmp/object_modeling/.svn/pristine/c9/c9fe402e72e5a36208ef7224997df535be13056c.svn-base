#include "frustum.h"

Frustum::Frustum(ParamsCamera const& cam, Eigen::Affine3f const& camera_pose)
{
	// uses knowledge that corners are in "binary" order (000, 001, 010, 011, etc.)
	points.resize(8);
	Eigen::Vector2f pixel_max = (cam.size - 1).cast<float>();
	points[0] = pointFromPixel(Eigen::Vector2f(0,0), cam.min_max_depth[0], cam.focal, cam.center);
	points[1] = pointFromPixel(Eigen::Vector2f(pixel_max.x(),0), cam.min_max_depth[0], cam.focal, cam.center);
	points[2] = pointFromPixel(Eigen::Vector2f(0,pixel_max.y()), cam.min_max_depth[0], cam.focal, cam.center);
	points[3] = pointFromPixel(Eigen::Vector2f(pixel_max.x(),pixel_max.y()), cam.min_max_depth[0], cam.focal, cam.center);
	points[4] = pointFromPixel(Eigen::Vector2f(0,0), cam.min_max_depth[1], cam.focal, cam.center);
	points[5] = pointFromPixel(Eigen::Vector2f(pixel_max.x(),0), cam.min_max_depth[1], cam.focal, cam.center);
	points[6] = pointFromPixel(Eigen::Vector2f(0,pixel_max.y()), cam.min_max_depth[1], cam.focal, cam.center);
	points[7] = pointFromPixel(Eigen::Vector2f(pixel_max.x(),pixel_max.y()), cam.min_max_depth[1], cam.focal, cam.center);

	// transform points by camera pose
	for (size_t i = 0; i < points.size(); ++i) {
		points[i] = camera_pose * points[i];
	}

	// all planes point into frustum
	planes.resize(6);
	planes[0] = planeFromPoints(points[0], points[1], points[2]); // near
	planes[1] = planeFromPoints(points[7], points[5], points[6]); // far
	planes[2] = planeFromPoints(points[2], points[6], points[0]); // left
	planes[3] = planeFromPoints(points[3], points[1], points[7]); // right
	planes[4] = planeFromPoints(points[3], points[7], points[2]); // bottom
	planes[5] = planeFromPoints(points[0], points[4], points[1]); // top
}

Eigen::Vector4f Frustum::planeFromPoints(Eigen::Vector3f const& point0, Eigen::Vector3f const& point1, Eigen::Vector3f const& point2)
{
	Eigen::Vector3f ray1 = point1 - point0;
	Eigen::Vector3f ray2 = point2 - point0;
	Eigen::Vector3f normal = ray1.cross(ray2).normalized();

	Eigen::Vector4f result;
	result.head<3>() = normal;
	result[3] = -normal.dot(point0);
	return result;
}

Eigen::Vector3f Frustum::pointFromPixel(Eigen::Vector2f const& pixel, float z, Eigen::Vector2f const& proj_f, Eigen::Vector2f const& proj_c)
{
	Eigen::Vector3f result;
	result.z() = z;
	result.head<2>() = ((pixel - proj_c).array() * z / proj_f.array()).matrix();
	return result;
}

std::vector<Eigen::Vector3f> Frustum::cornersFromAABB(Eigen::Vector3f const& bb_min, Eigen::Vector3f const& bb_max)
{
	std::vector<Eigen::Vector3f> result(8);
	result[0] = Eigen::Vector3f(bb_min[0], bb_min[1], bb_min[2]);
	result[1] = Eigen::Vector3f(bb_min[0], bb_min[1], bb_max[2]);
	result[2] = Eigen::Vector3f(bb_min[0], bb_max[1], bb_min[2]);
	result[3] = Eigen::Vector3f(bb_min[0], bb_max[1], bb_max[2]);
	result[4] = Eigen::Vector3f(bb_max[0], bb_min[1], bb_min[2]);
	result[5] = Eigen::Vector3f(bb_max[0], bb_min[1], bb_max[2]);
	result[6] = Eigen::Vector3f(bb_max[0], bb_max[1], bb_min[2]);
	result[7] = Eigen::Vector3f(bb_max[0], bb_max[1], bb_max[2]);
	return result;
}

bool Frustum::isPointInside(Eigen::Vector3f const& p)
{
	Eigen::Vector4f p_test;
	p_test.head<3>() = p;
	p_test[3] = 1;
	bool result = true;
	for (int i = 0; i < 6; ++i) {
		result = result && planes[i].dot(p_test) > 0;
	}
	return result;
}

bool Frustum::doesAABBIntersect(Eigen::Vector3f const& p_min, Eigen::Vector3f const& p_max)
{
	// cout << "DEBUG REMOVE call to doesAABBIntersect: " << p_min.transpose() << " - " << p_max.transpose() << endl;
	std::vector<Eigen::Vector3f> corners = cornersFromAABB(p_min, p_max);

	for (int plane = 0; plane < 6; ++plane) {
		// count vertices inside plane
		int vertices_inside = 8;
		for (int corner = 0; corner < 8; ++corner) {
			Eigen::Vector4f useful_corner;
			useful_corner.head<3>() = corners[corner];
			useful_corner[3] = 1;
			if (useful_corner.dot(planes[plane]) < 0) {
				--vertices_inside;
			}
		}
		if (vertices_inside == 0) return false;
	}

	// not all vertices were outside all planes
	return true;
}

std::vector<Eigen::Vector3f> const& Frustum::getPoints() const
{
	return points;
}

std::vector<Eigen::Vector3f> Frustum::getLineSetPoints() const
{
	std::vector<Eigen::Vector3f> result;

	// near
	result.push_back(points[0]);
	result.push_back(points[1]);
	result.push_back(points[1]);
	result.push_back(points[3]);
	result.push_back(points[3]);
	result.push_back(points[2]);
	result.push_back(points[2]);
	result.push_back(points[0]);

	// far
	result.push_back(points[4]);
	result.push_back(points[5]);
	result.push_back(points[5]);
	result.push_back(points[7]);
	result.push_back(points[7]);
	result.push_back(points[6]);
	result.push_back(points[6]);
	result.push_back(points[4]);

	// sides
	result.push_back(points[0]);
	result.push_back(points[4]);
	result.push_back(points[1]);
	result.push_back(points[5]);
	result.push_back(points[2]);
	result.push_back(points[6]);
	result.push_back(points[3]);
	result.push_back(points[7]);

	return result;
}

