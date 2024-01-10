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
    img.convertTo(img, CV_32F);
    _orig = img;
}

void Sift::process() {
    if (_options.verbose_output) {
        std::cout << "SIFT: Creating " << _options.max_octave - _options.min_octave << " octaves (" << _options.min_octave << " to " << _options.max_octave << ")..." << std::endl;
    }
    create_octaves();
    if (_options.debug_output) {
        std::cout << "SIFT: Creating octaves took" << std::endl;
    }
    if (_options.debug_output) {
        std::cout << "SIFT: Detecting local extrema..." << std::endl;
    }
    extrema_detection();
    if (_options.debug_output) {
        std::cout << "SIFT: Detected " << _keypoints.size() << " keypoints" << std::endl;
    }
    if (_options.debug_output) {
        std::cout << "SIFT: Localizing and filtering keypoints... " << std::endl;
    }
    keypoint_localization();
}

void Sift::create_octaves() {
    _octaves.clear();

    cv::Mat img = _orig;
    float img_sigma = _options.inherent_blur_sigma;
    for (int i = std::max(0, _options.min_octave); i <= _options.max_octave; i++) {
        add_octave(img, img_sigma, _options.base_blur_sigma);

        //用上一阶octave的倒数第三张图，作为下一个octave的base
        cv::Mat pre_base = _octaves[_octaves.size() - 1].img[0];
        cv::pyrDown(pre_base, img);
        img_sigma = _options.base_blur_sigma;
    }
}


void Sift::add_octave(cv::Mat image, float has_sigma, float target_sigma) {
    float sigma = std::sqrt(std::pow(target_sigma, 2) - pow(has_sigma, 2));


    //卷积核大小一般为6sigma,必须为奇数
    int window_size = int(sigma * 6) % 2 == 0 ? int(sigma * 6) + 1 : int(sigma * 6);
    //如果sigma=0，说明has_sigma==target_sigma
    cv::Mat base = target_sigma > has_sigma ? (cv::GaussianBlur(image, image, cv::Size(window_size, window_size), sigma), image) : image.clone();
    _octaves.push_back(Octave());
    Octave &oct = _octaves.back();
    oct.img.push_back(base);

    float const k = std::pow(2.0f, 1.0f / _options.num_samples_per_octave); //当 k = pow(2,1/S) 输出尺度是连续的
    sigma = target_sigma;

    for (int i = 1; i < _options.num_samples_per_octave + 3; i++) {
        float sigma_k = sigma * k;
        /**
         * img ->(sigma1) img1;   img ->(sigma2) img2;  img1 ->(sqrt(sigma2^2 - sigma1^2)) img2;
         * sigma2 > sqrt(sigma2^2 - sigma1^2) 提升效率
         */
        float blur_sigma = std::sqrt(std::pow(sigma_k, 2) - std::pow(sigma, 2));
        window_size = int(blur_sigma * 6) % 2 == 0 ? int(blur_sigma * 6) + 1 : int(blur_sigma * 6);
        cv::Mat tmp;
        cv::GaussianBlur(base, tmp, cv::Size(window_size, window_size), blur_sigma);
        oct.img.push_back(base);

        //计算差分拉普拉斯
        cv::Mat dog = base - tmp;
        oct.dog.push_back(dog);
        base = tmp;
        sigma = sigma_k;
    }
}

void Sift::extrema_detection() {
    _keypoints.clear();
    for (std::size_t i = 0; i < _octaves.size(); i++) {
        Octave const &oct(_octaves[i]);
        for (int s = 0; s < oct.dog.size() - 2; s++) {
            //有效dog个数dog.size() - 2，因为需要从三个dog中选出响应最大的值，而首尾dog没法组成三个，因此要去掉
            cv::Mat samples[3] = {oct.dog[s + 0], oct.dog[s + 1], oct.dog[s + 2]};
            extrema_detection(samples, _options.min_octave + i, s);
        }
    }
}

/**
 * 
 * @param samples 
 * @param oi 所在的octave阶数
 * @param si 所在的dog阶数
 */
void Sift::extrema_detection(cv::Mat *samples, int oi, int si) {
    const int w = samples[1].cols;
    const int h = samples[1].rows;

    //9个邻居位置的偏置
    int noff_c[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int noff_r[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int detected = 0;
    for (int r = 1; r < h - 1; r++) {
        for (int c = 1; c < w - 1; c++) {
            bool largest = true;
            bool smallest = true;
            float center_value = samples[1].at<float>(r, c);
            //遍历每层的9个邻居
            for (int l = 0; (largest || smallest) && l < 3; l++) {
                for (int i = 0; i < 9; i++) {
                    if (i == 4) continue; // 跳过中心点
                    //判断中心点是不是27个邻域中的极值点
                    if (samples[l].at<float>(r + noff_r[i], c + noff_c[i]) >= center_value) {
                        largest = false;
                    }
                    if (samples[l].at<float>(r + noff_r[i], c + noff_c[i]) >= center_value) {
                        smallest = false;
                    }
                }
            }
            if (!smallest && !largest) {
                continue;
            }
            //是极值点
            Keypoint kp;
            kp.octave = oi;
            kp.sample = (float) si;
            kp.row = (float) r;
            kp.col = (float) c;
            _keypoints.push_back(kp);
            detected += 1;
        }
    }
}


/**
 * 亚像素精度求解
 */
void Sift::keypoint_localization() {
    int num_keypoints = 0;
    int num_singular = 0;
    for (int i = 0; i < _keypoints.size(); i++) {
        Keypoint kp(_keypoints[i]);
        Octave const &oct(_octaves[kp.octave - _options.min_octave]);
        int sample = (int) kp.sample;
        cv::Mat dogs[3] = {oct.dog[sample + 0], oct.dog[sample + 1], oct.dog[sample + 2]};

        int const w = dogs[0].cols;
        int const h = dogs[0].rows;

        int ir = (int) kp.row;
        int ic = (int) kp.col;
        int is = (int) kp.sample;
        float delta_r, delta_c, delta_s;

        float Dr, Dc, Ds;
        float Drr, Dcc, Dss;
        float Dcr, Dcs, Drs;
        for (int i = 0; i < 5; i++) {
            //最多迭代5次
            Dr = (dogs[1].at<float>(ir + 1, ic) - dogs[1].at<float>(ir - 1, ic)) * 0.5f;
            Dc = (dogs[1].at<float>(ir, ic + 1) - dogs[1].at<float>(ir, ic - 1)) * 0.5f;
            Ds = (dogs[0].at<float>(ir, ic) - dogs[2].at<float>(ir, ic)) * 0.5f;
            Drr = dogs[1].at<float>(ir + 1, ic) + dogs[1].at<float>(ir - 1, ic) - 2 * dogs[1].at<float>(ir, ic);
            Dcc = dogs[1].at<float>(ir, ic + 1) + dogs[1].at<float>(ir, ic - 1) - 2 * dogs[1].at<float>(ir, ic);
            Dss = dogs[0].at<float>(ir, ic) + dogs[2].at<float>(ir, ic) - 2 * dogs[1].at<float>(ir, ic);
            Dcr = ((dogs[1].at<float>(ir + 1, ic + 1) + dogs[1].at<float>(ir + 1, ic - 1)) - (dogs[1].at<float>(ir + 1, ic + 1) + dogs[1].at<float>(ir - 1, ic + 1))) * 0.25f;
            Dcs = ((dogs[0].at<float>(ir, ic + 1) + dogs[2].at<float>(ir, ic + 1)) - (dogs[0].at<float>(ir, ic - 1) + dogs[2].at<float>(ir, ic + 1))) * 0.25f;
            Drs = ((dogs[0].at<float>(ir + 1, ic) + dogs[2].at<float>(ir + 1, ic)) - (dogs[0].at<float>(ir - 1, ic) + dogs[2].at<float>(ir + 1, ic))) * 0.25f;

            //构造hessian矩阵
            Eigen::Matrix<float, 3, 3> H;
            H << Dcc, Dcr, Dcs, Dcr, Drr, Drs, Dcs, Drs, Dss;
            float detH = H.determinant();
            if (MATH_EPSILON_EQ(detH, 0.0f, 1e-15f)) {
                //行列式为0
                num_singular++;
                delta_c = delta_r = delta_s = 0.0f;
                break;
            }
            Eigen::Matrix<float, 3, 3> H_inv;
            H_inv = H.inverse();
            Eigen::Vector3f b(-Dc, -Dr, -Ds);
            Eigen::Vector3f delta = H_inv * b;
            delta_c = delta[0];
            delta_r = delta[1];
            delta_s = delta[2];

            //如果 |delta| < 0.6f 说明当前的极值点是正确的，不需要变 d = 0
            int dc = (delta_c >0.6f && ic < w - 2) * 1 + (delta_c < -0.6f && ic > 1) * -1;
            int dr = (delta_r > 0.6f && ir < h - 2) * 1 + (delta_r < -0.6f && ir > 1) * -1;
            if(dc != 0 || dr !=0){
                ic += dc;
                ir += dr;
                continue;
            }
            break;
        }
        float val = dogs[1].at<float>(ic,ir) + 0.5f * (Dc*delta_c + Dr * delta_r + Ds * delta_s);

        //去除边缘点
        float hessian_trace = Dcc+Drr;
        float hessian_del = Dcc*Drr - Dcr*Dcr;
        float score = powf(hessian_trace,2) / hessian_del;
        float score_thres =powf(_options.edge_ratio_threshold+1,2) / _options.edge_ratio_threshold;

        kp.col = (float)ic+delta_c;
        kp.row = (float)ir+delta_r;
        kp.sample = (float)ic+delta_s;

        if(fabs(val) < _options.contrast_threshold
            || score < score_thres
            || score < 0.0f
            || fabs(delta_c) > 1.5f || fabs(delta_r) > 1.5f || fabs(delta_s) > 1.0f
            || kp.col < 0.0f || kp.col > w-1
            || kp.row < 0.0f || kp.row > h-1
            || kp.sample < -1.0f || kp.sample > _options.num_samples_per_octave
            )
            continue;

        _keypoints[num_keypoints++] = kp;
    }
    _keypoints.resize(num_keypoints);
    if (_options.debug_output && num_singular > 0)
    {
        std::cout << "SIFT: Warning: " << num_singular
                  << " singular matrices detected!" << std::endl;
    }

}