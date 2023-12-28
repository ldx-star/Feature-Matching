//
// Created by ldx on 23-12-25.
//

#ifndef FEATURE_MATCHING_SIFT_H
#define FEATURE_MATCHING_SIFT_H

#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>

#endif //FEATURE_MATCHING_SIFT_H

class Sift {
public:

    struct Options {

        Options(void);
        // 每个有效的DoG个数S, 每阶需要DoG图像个数S+2（非极大值抑制，选取值的时候需要考虑上下DoG图像，因此第一个和最后一个DoG图像为无效DoG图像）
        // 需要高斯平滑图像个数 N=S+3 (每两个高斯平滑图像合成一个DoG图像)
        int num_samples_per_octave; //有效DoG个数
        int min_octave;
        int max_octave;

        bool verbose_output; // 是否在控制台显示信息
        bool debug_output;

        float inherent_blur_sigma;
        float base_blur_sigma;
    };


    //关键点
    struct Keypoint {
        int octave;
        float sample;
        float x;
        float y;
    };

public:
    typedef std::vector<Keypoint> Keypoints;

public:
    explicit Sift(Options const &options);

    void set_image(cv::Mat img);


    void process();


//protected 可以在类内和子内中访问
protected:
    struct Octave{
        typedef std::vector<cv::Mat> ImageVector;
        ImageVector img; // S + 3
        ImageVector dog; // S + 2
        ImageVector grad; //梯度图
        ImageVector origin; // 原始图
    };

protected:
    typedef std::vector<Octave> Octaves;

protected:
    void create_octaves();
    void add_octave(cv::Mat image,float has_sigma,float target_sigma);


//private 只能在类内访问
private:
    Options _options;
    Octaves _octaves;
    cv::Mat _orig; //原始输入图像
};
//inline 关键字，编译器在编译时会插入该函数，而不是调用函数，提升效率
inline Sift::Options::Options()
    :num_samples_per_octave(3),
    min_octave(0),
    max_octave(4),
    inherent_blur_sigma(0.5),
    base_blur_sigma(1.6)
    {}