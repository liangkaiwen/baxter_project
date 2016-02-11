#pragma once

#include <Eigen/Core>

struct ParamsCamera
{
	Eigen::Array2i size;
	Eigen::Array2f focal;
	Eigen::Array2f center;
	Eigen::Array2f min_max_depth;

	// note that center is not automatically reset if you change size (yet...)
	ParamsCamera()
		: size(640,480),
		focal(525,525),
		min_max_depth(0.4, 5.0)
	{
		setCenterFromSize();
	}

	void setCenterFromSize() 
	{
		center = (size-1).cast<float>() * 0.5;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // needed?
};
