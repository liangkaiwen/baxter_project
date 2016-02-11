#include "stdafx.h"

#include "convertOpenCLEigen.h"

Eigen::Vector4f toEigenVector4f(const cl_float4& v)
{
	return Eigen::Vector4f(v.s[0], v.s[1], v.s[2], v.s[3]);
}

Eigen::Vector4i toEigenVector4i(const cl_int4& v)
{
	return Eigen::Vector4i(v.s[0], v.s[1], v.s[2], v.s[3]);
}

// ignores 4th element
Eigen::Vector3f toEigenVector3f(const cl_float4& v)
{
	return Eigen::Vector3f(v.s[0], v.s[1], v.s[2]);
}

// ignores 4th element
Eigen::Vector3i toEigenVector3i(const cl_int4& v)
{
	return Eigen::Vector3i(v.s[0], v.s[1], v.s[2]);
}