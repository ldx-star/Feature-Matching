//
// Created by ldx on 23-12-25.
//
#include "../include/Sift.h"

Sift::Sift(const Sift::Options &options)
        : _options(options) {
    if (this->_options.min_octave < -1 || this->_options.min_octave > this->_options.max_octave) {
        throw std::invalid_argument("Invalid octave range");
    }
    if (this->_options.debug_output) this->_options.verbose_output = true;
}

void Sift::set_image(cv::Mat img) {
    if (img.channels() != 3 && img.channels() != 1) {
        throw std::invalid_argument("Gray or color image expected");
    }
    //转为灰度图
    if (img.channels() == 3) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    _orig = img;
}

void Sift::process() {
    if (_options.verbose_output) {
        std::cout << "SIFT: Creating " << _options.max_octave - _options.min_octave << " octaves (" << _options.min_octave << " to " << _options.max_octave << ")" << std::endl;
    }
    create_octaves();
}

void Sift::create_octaves() {
    _octaves.clear();

    cv::Mat img = _orig;
    float img_sigma = _options.inherent_blur_sigma;
    for (int i = std::max(0, _options.min_octave); i <= _options.max_octave; i++) {
        add_octave(img, img_sigma, _options.base_blur_sigma);
    }
}

/**
 * 根据当前 sigma 和 目标sigma 得到中转sigma
 * img ->(has_sigma) img1;   img ->(target_sigma) img2;  img1 ->(sqrt(target_sigma^2 - has_sigma^2)) img2;
 * target_sigma > sqrt(target_sigma^2 - has_sigma^2) 提升效率
 * @param image
 * @param has_sigma
 * @param target_sigma
 */
void Sift::add_octave(cv::Mat image, float has_sigma, float target_sigma) {
    float sigma = std::sqrt(std::pow(target_sigma, 2) - pow(has_sigma, 2));


    //卷积核大小一般为6sigma,必须为奇数
    int window_size = int(sigma * 6) % 2 == 0 ? int(sigma * 6) + 1 : int(sigma * 6);
    cv::Mat base = target_sigma > has_sigma ? (cv::GaussianBlur(image, image, cv::Size(window_size, window_size), sigma), image) : image.clone();
    _octaves.push_back(Octave());
    Octave& oct = _octaves.back();
    oct.img.push_back(base);

    float const k = std::pow(2,1/_options.num_samples_per_octave); //当 k = pow(2,1/S) 输出尺度是连续的

}