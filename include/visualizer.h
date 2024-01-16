//
// Created by ldx on 24-1-16.
//

#ifndef FEATURE_MATCHING_VISUALIZER_H
#define FEATURE_MATCHING_VISUALIZER_H
#include <opencv2/opencv.hpp>

class Visualizer{
public:
    struct Keypoint{
        float x;
        float y;
        float radius;
        float orientation;
    };

public:
    static void draw_keypoint(cv::Mat& img,const std::vector<Keypoint>& matches );
};
#endif //FEATURE_MATCHING_VISUALIZER_H
