//
// Created by ldx on 23-12-14.
//
#include<opencv2/opencv.hpp>
#include "include/Sift.h"
int main(){




    cv::Mat img1 = cv::imread("../data/img1.jpg");
    cv::Mat img2 = cv::imread("../data/img2.jpg");

    Sift::Keypoints sift_keypoints;
    {
        Sift::Options sift_options;
        sift_options.verbose_output = true;
        sift_options.debug_output = true;
        Sift sift(sift_options);
        sift.set_image(img1);
        sift.process();


    }


    return 0;
}