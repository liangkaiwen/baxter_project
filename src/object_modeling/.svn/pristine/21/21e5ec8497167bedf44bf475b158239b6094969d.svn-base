#pragma once



#include "DLoopDetector.h"
#include "DBoW2.h"


/////////////////
// copied from demo_brief.cpp
/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<DBoW2::FBrief::TDescriptor>
{
public:
  /**
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<DVision::BRIEF::bitset> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};
