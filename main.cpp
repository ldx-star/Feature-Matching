//
// Created by ldx on 23-12-14.
//
#include<opencv2/opencv.hpp>
#include "include/Sift.h"
#include "include/visualizer.h"

bool sift_compare(const Sift::Descriptor& d1 ,const Sift::Descriptor& d2){
    return d1.scale > d2.scale;
}
int main(){




    cv::Mat img1 = cv::imread("../data/img1.jpg");
    cv::Mat img2 = cv::imread("../data/img2.jpg");

    Sift::Keypoints sift_keypoints;
    Sift::Descriptors sift_descriptors;
    {
        Sift::Options sift_options;
        sift_options.verbose_output = true;
        sift_options.debug_output = true;
        Sift sift(sift_options);
        sift.set_image(img1);
        sift.process();
        std::cout << "Computed SIFT features" << std::endl;
        sift_descriptors = sift.get_descriptors();
        sift_keypoints = sift.get_keypoints();
    }
    // 按照尺度排序
    std::sort(sift_descriptors.begin(),sift_descriptors.end(), sift_compare);
    std::vector<Visualizer::Keypoint> sift_drawing;
    for(int i = 0; i < sift_descriptors.size(); i++){
        Visualizer::Keypoint kp;
        kp.orientation = sift_descriptors[i].orientation;
        kp.radius = sift_descriptors[i].scale;
        kp.x = sift_descriptors[i].x;
        kp.y = sift_descriptors[i].y;
        sift_drawing.push_back(kp);
    }
    Visualizer::draw_keypoint(img1,sift_drawing);
    cv::imshow("img",img1);
    cv::waitKey(0);

    return 0;
}