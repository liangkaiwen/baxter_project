#pragma once

enum FeatureType
{
	FEATURE_TYPE_FAST,
	FEATURE_TYPE_SURF,
	FEATURE_TYPE_ORB
};

struct ParamsFeatures
{
	FeatureType feature_type;

	int fast_threshold;
	bool fast_pyramid_adapter;
	bool fast_grid_adapter;


	int max_features; // where supported

	bool use_ratio_test;
	float ratio_test_ratio;

	ParamsFeatures()
		: feature_type(FEATURE_TYPE_FAST),
		fast_threshold(12),
		fast_pyramid_adapter(true),
		fast_grid_adapter(true),
		max_features(500),
		use_ratio_test(false),
		ratio_test_ratio(0.8f)
	{}
};