
#include "../include/visualizer.h"

//
// Created by ldx on 24-1-16.
//

void Visualizer::draw_keypoint(cv::Mat &img, const std::vector <Visualizer::Keypoint> &matches) {
    for(int i = 0; i < matches.size(); i++){
        Visualizer::Keypoint kp(matches[i]);
        int const x = static_cast<int>(kp.x+0.5f);
        int const y = static_cast<int>(kp.y+0.5f);
        int const width = img.cols;
        int const height = img.rows;
        int const channels = img.channels();

        if(x < 0 || x >= width || y < 0 || y >= height){
            return;
        }
        int required_space = static_cast<int>(kp.radius);
        bool draw_orientation = true;

        if(x < required_space || x + required_space >= width || y - required_space < 0 || y + required_space >= height){
            required_space = 0;
            draw_orientation = false;
        }
        cv::Point center(x,y);
        cv::Scalar color(0,255,0);
        cv::circle(img,center,required_space,color);
        if(draw_orientation){
            const float sin_ori = std::sin(kp.orientation);
            const float cos_ori = std::cos(kp.orientation);
            const float x1 = cos_ori * kp.radius;
            const float y1 = sin_ori * kp.radius;
            cv::line(img,cv::Point(x,y),cv::Point(x+x1,y+y1),color);
        }
    }

}
