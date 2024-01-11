//
// Created by ldx on 23-12-25.
//

#ifndef FEATURE_MATCHING_SIFT_H
#define FEATURE_MATCHING_SIFT_H

#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
#include<Eigen/Core>
#include<Eigen/Dense>
#include "defines.h"


class Sift {
public:

    struct Options {

        Options(void);

        /**
         * 每个有效的DoG个数S, 每阶需要DoG图像个数S+2（非极大值抑制，选取值的时候需要考虑上下DoG图像，因此第一个和最后一个DoG图像为无效DoG图像）
         * 需要高斯平滑图像个数 N=S+3 (每两个高斯平滑图像合成一个DoG图像)
         */

        int num_samples_per_octave; //有效DoG个数
        int min_octave;
        int max_octave;

        bool verbose_output; // 是否在控制台显示信息
        bool debug_output;

        /**
         * base_blur_sigma 为图像初始的sigma值，默认为0.5，也就是说对于原图，默认它经过有sigma=0.5的高斯卷积
         * inherent_blur_sigma 为初始的目标sigma值， 默认为1.6
         */
        float inherent_blur_sigma;
        float base_blur_sigma;

        /// 进行亚像素精度定位时，对极值点处的DoG值的阈值
        float contrast_threshold;
        /// 用于消除边缘响应
        float edge_ratio_threshold;
    };


    ///关键点
    struct Keypoint {
        int octave;
        float sample;
        float row;
        float col;
    };

    /// 特征描述子
    struct Descriptor {
        float x;
        float y;
        float scale;
        float orientation;
        float data[128] = {};
    };

public:
    typedef std::vector<Keypoint> Keypoints;
    typedef std::vector<Descriptor>  Descriptors;
public:
    explicit Sift(Options const &options);

    void set_image(cv::Mat img);


    void process();


//protected 可以在类内和子内中访问
protected:
    struct Octave {
        typedef std::vector<cv::Mat> ImageVector;
        ImageVector img; //高斯空间  S + 3
        ImageVector dog; // S + 2
        ImageVector grad; //梯度响应值
        ImageVector ori; // 梯度方向
    };

protected:
    typedef std::vector<Octave> Octaves;

protected:
    void create_octaves();

    void add_octave(cv::Mat image, float has_sigma, float target_sigma);

    void extrema_detection();

    void extrema_detection(cv::Mat samples[3], int oi, int si);

    void keypoint_localization();

    void descriptor_generation();

    void generate_grad_ori_images(Octave* octave);

    void orientation_assignment(const Keypoint &kp,const Octave *octave, std::vector<float> &orientations);
    bool descriptor_assignment(const Keypoint &kp, Descriptor &desc, const Octave *octave);

    float keypoint_relative_scale(const Keypoint &kp) const;
    float keypoint_absolute_scale(const Keypoint &kp) const;
//private 只能在类内访问
private:
    Options _options;
    Octaves _octaves;
    cv::Mat _orig; //原始输入图像
    Keypoints _keypoints;
    Descriptors _descriptors;
};

//inline 关键字，编译器在编译时会插入该函数，而不是调用函数，提升效率
inline Sift::Options::Options()
        : num_samples_per_octave(3),
          min_octave(0),
          max_octave(4),
          inherent_blur_sigma(0.5),
          base_blur_sigma(1.6),
          contrast_threshold(-1.0f),
          edge_ratio_threshold(10.0f) {}

#endif //FEATURE_MATCHING_SIFT_H
