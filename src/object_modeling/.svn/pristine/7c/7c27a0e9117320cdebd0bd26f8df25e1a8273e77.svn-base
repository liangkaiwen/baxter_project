#include "EigenUtilities.h"

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <iostream>
using std::cout;
using std::endl;

namespace EigenUtilities {

	std::string transformToString(const Eigen::Affine3f& x)
	{
		std::ostringstream outstr;
		const Eigen::Matrix<float, 7, 1> v = convertTransformToVector(x);
		std::copy(v.data(), v.data() + 7, std::ostream_iterator<float>(outstr, " "));
		return outstr.str();
	}

	Eigen::Affine3f stringToTransform(const std::string& s)
	{
		std::istringstream instr(s);
		Eigen::Matrix<float, 7, 1> v;
		std::copy(std::istream_iterator<float>(instr), std::istream_iterator<float>(), v.data());
		return convertVectorToTransform(v);
	}

	Eigen::Isometry3d getIsometry3d(const Eigen::Affine3f& t)
	{
		Eigen::Isometry3d result;
		result.matrix() = t.matrix().cast<double>();
		return result;
	}

	Eigen::Affine3f getAffine3f(const Eigen::Isometry3d& i)
	{
		Eigen::Affine3f result;
		result.matrix() = i.matrix().cast<float>();
		return result;
	}

	void getAngleAndDistance(const Eigen::Affine3f& t, float & angle, float & distance)
	{
		distance = t.translation().norm();
		Eigen::AngleAxisf aa(t.rotation());
		angle = fabs(180.f / M_PI * aa.angle());
	}

	void getCameraPoseDifference(const Eigen::Affine3f& p1, const Eigen::Affine3f& p2, float & angle, float & distance)
	{
		getAngleAndDistance(p1.inverse() * p2, angle, distance);
	}

	Eigen::Vector3i truncateVector3fToInt(const Eigen::Vector3f& float_v)
	{
		return Eigen::Vector3i( (int)float_v[0], (int)float_v[1], (int)float_v[2] );
	}

	Eigen::Array3i truncateArray3fToInt(Eigen::Array3f const& float_a)
	{
		return Eigen::Array3i( (int)float_a[0], (int)float_a[1], (int)float_a[2] );
	}

	Eigen::Array3i floorArray3fToInt(Eigen::Array3f const& float_a)
	{
		return Eigen::Array3i( std::floor(float_a[0]), std::floor(float_a[1]), std::floor(float_a[2]) );
	}

	Eigen::Vector3i roundPositiveVector3fToInt(const Eigen::Vector3f& float_v)
	{
		return truncateVector3fToInt(float_v + Eigen::Vector3f(0.5,0.5,0.5));
	}

	boost::tuple<int,int,int> array3iToTuple(Eigen::Array3i const& array_3i)
	{
		return boost::tuple<int,int,int>(array_3i[0], array_3i[1], array_3i[2]);
	}

	Eigen::Array3i tupleToArray3i(boost::tuple<int,int,int> const& array_3i)
	{
		return Eigen::Array3i(array_3i.get<0>(), array_3i.get<1>(), array_3i.get<2>());
	}

	void loadPosesFromFile(fs::path filename, PosePtrList & result)
	{
		result.clear();
		if (filename.empty()) return;
		std::fstream file(filename.string().c_str(), std::ios::in);
		std::vector<std::string> lines;

		std::string line;
		while (std::getline(file, line)) lines.push_back(line);

		result.clear();
		for (std::vector<std::string>::iterator iter = lines.begin(); iter != lines.end(); ++iter) {
			std::string trimmed = boost::trim_copy(*iter);

			std::string transform_string;

			std::vector<std::string> split;
			boost::split(split, trimmed, boost::is_any_of(" "));
			if (split.size() == 8) {
				// there's a better way to do this
				// assume: ROSTIMESTAMP tx ty tz qw qx qy qz
				transform_string = (boost::format("%s %s %s %s %s %s %s") % split[1] % split[2] % split[3] % split[4] % split[5] % split[6] % split[7]).str();
			}
			else if (split.size() == 7) {
				// assume perfect
				transform_string = trimmed;
			}
			else {
				cout << "unexpected line in loadPosesFromFile: " << trimmed << endl;
				continue;
			}

			result.push_back(PosePtr(new Eigen::Affine3f(EigenUtilities::stringToTransform(transform_string))));
		}
	}

	Eigen::Vector3f depthToPoint(const Eigen::Array2f & camera_focal, const Eigen::Array2f & camera_center, const Eigen::Array2i & pixel, float depth)
	{
		Eigen::Vector3f result = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
		if (depth > 0) {
			result.head<2>() = (pixel.cast<float>() - camera_center) * depth / camera_focal;
			result[2] = depth;
		}
		return result;
	}


} // ns
