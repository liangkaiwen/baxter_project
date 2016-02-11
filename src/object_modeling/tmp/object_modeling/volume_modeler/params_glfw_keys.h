#pragma once

#include <string>

struct ParamsGLFWKeys
{
    std::string input_cloud;

    std::string mesh;

    std::string volumes_all;
    std::string volumes_active;

    std::string pose_graph_all;

    std::string cameras_all;
    std::string cameras_keyframes_and_graph;

    std::string frustum;

    std::string debug_overlap;

    ParamsGLFWKeys()
        : input_cloud("1"),
          mesh("2"),
          volumes_all("3"),
          volumes_active("4"),
          pose_graph_all("5"),
          cameras_all("8"),
          cameras_keyframes_and_graph("9"),
          frustum("0"),
          debug_overlap("debug_overlap")
    {}
};
