#include "dbow_place_recognition.h"

#include <boost/timer.hpp>

#include <opencv2/opencv.hpp>

#include <iostream>
using std::cout;
using std::endl;

DBOWPlaceRecognition::DBOWPlaceRecognition(const VolumeModelerAllParams &params)
    : params_(params),
      brief_loop_detector_params_(params.camera.size[1], params.camera.size[0])
{
    // copied from demoDetector.h
    // We are going to change these values individually:
    brief_loop_detector_params_.use_nss = true; // use normalized similarity score instead of raw score
    brief_loop_detector_params_.alpha = 0.3; // nss threshold
    brief_loop_detector_params_.k = 1; // a loop must be consistent with 1 previous matches
    brief_loop_detector_params_.geom_check = DLoopDetector::GEOM_DI; // use direct index for geometrical checking
    brief_loop_detector_params_.di_levels = 2; // use two direct index levels

    if (params_.dbow_place_recognition.debug_allow_sequential_closures) {
        brief_loop_detector_params_.dislocal = 0; // try to get lots of matches for debugging
    }

    // load vocabulary
    //fs::path vocabulary_file = "/home/peter/checkout/dbow_place_recognition/DLoopDetector/build/resources/brief_k10L6.voc.gz";
    fs::path vocabulary_file = params_.dbow_place_recognition.resources_folder / "brief_k10L6.voc.gz";
    if (!fs::exists(vocabulary_file)) {
        cout << "vocabulary_file does not exist: " << vocabulary_file << endl;
        throw std::runtime_error("vocabulary_file");
    }

    boost::timer t_load_vocab;
    cout << "Loading vocabulary file " << vocabulary_file << endl;
    BriefVocabulary vocabulary(vocabulary_file.string());
    cout << "Finished loading vocabulary in time: " << t_load_vocab.elapsed() << endl;

    brief_loop_detector_ptr_.reset(new BriefLoopDetector(vocabulary, brief_loop_detector_params_));

    fs::path extractor_pattern = params_.dbow_place_recognition.resources_folder / "brief_pattern.yml";
    if (!fs::exists(extractor_pattern)) {
        cout << "extractor_pattern does not exist: " << extractor_pattern << endl;
        throw std::runtime_error("extractor_pattern");
    }
    brief_extractor_ptr_.reset(new BriefExtractor(extractor_pattern.string()));
}

void DBOWPlaceRecognition::addAndDetectBGRA(const cv::Mat & image_bgra, std::vector<unsigned int> & result_loops)
{
    // though it claims to accept color, doesn't like bgra
    cv::Mat image_gray;
    cv::cvtColor(image_bgra, image_gray, CV_BGRA2GRAY);
    addAndDetectGray(image_gray, result_loops);
}

void DBOWPlaceRecognition::addAndDetectGray(const cv::Mat & image_gray, std::vector<unsigned int> & result_loops)
{

    // right now only get one success, but could imagine candidates
    // you have to get inside dloopdetector to get them though
    result_loops.clear();

    std::vector<cv::KeyPoint> keys;
    std::vector<DVision::BRIEF::bitset> descriptors;
    (*brief_extractor_ptr_)(image_gray, keys, descriptors);

    DLoopDetector::DetectionResult result;
    brief_loop_detector_ptr_->detectLoop(keys, descriptors, result);

    // debug output:
    cout << "[DBOW PLACE RECOGNITION]:" << endl;

    if (result.detection()) {
        // debug output:
        cout << "- Loop found with image " << result.match << "!"
          << endl;

        result_loops.push_back(result.match);
    }
    else {
        // debug output:
        // dopied from demoDetector.h
        cout << "- No loop: ";
        switch(result.status)
        {
          case DLoopDetector::CLOSE_MATCHES_ONLY:
            cout << "All the images in the database are very recent" << endl;
            break;

          case DLoopDetector::NO_DB_RESULTS:
            cout << "There are no matches against the database (few features in"
              " the image?)" << endl;
            break;

          case DLoopDetector::LOW_NSS_FACTOR:
            cout << "Little overlap between this image and the previous one"
              << endl;
            break;

          case DLoopDetector::LOW_SCORES:
            cout << "No match reaches the score threshold (alpha: " <<
              brief_loop_detector_params_.alpha << ")" << endl;
            break;

          case DLoopDetector::NO_GROUPS:
            cout << "Not enough close matches to create groups. "
              << "Best candidate: " << result.match << endl;
            break;

          case DLoopDetector::NO_TEMPORAL_CONSISTENCY:
            cout << "No temporal consistency (k: " << brief_loop_detector_params_.k << "). "
              << "Best candidate: " << result.match << endl;
            break;

          case DLoopDetector::NO_GEOMETRICAL_CONSISTENCY:
            cout << "No geometrical consistency. Best candidate: "
              << result.match << endl;
            break;

          default:
            break;
        }
    }
}

