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
    if (_options.debug_output) {
        std::cout << "SIFT: Retained " << _keypoints.size() << " stable " << std::endl;
    }
    // 清除dog
    for (int i = 0; i < _octaves.size(); i++) {
        _octaves[i].dog.clear();
    }
    if (_options.verbose_output) {
        std::cout << "SIFT: Generating keypoint descriptors" << std::endl;
    }
    descriptor_generation();

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
            int dc = (delta_c > 0.6f && ic < w - 2) * 1 + (delta_c < -0.6f && ic > 1) * -1;
            int dr = (delta_r > 0.6f && ir < h - 2) * 1 + (delta_r < -0.6f && ir > 1) * -1;
            if (dc != 0 || dr != 0) {
                ic += dc;
                ir += dr;
                continue;
            }
            break;
        }
        float val = dogs[1].at<float>(ic, ir) + 0.5f * (Dc * delta_c + Dr * delta_r + Ds * delta_s);

        //去除边缘点
        float hessian_trace = Dcc + Drr;
        float hessian_del = Dcc * Drr - Dcr * Dcr;
        float score = powf(hessian_trace, 2) / hessian_del;
        float score_thres = powf(_options.edge_ratio_threshold + 1, 2) / _options.edge_ratio_threshold;

        kp.col = (float) ic + delta_c;
        kp.row = (float) ir + delta_r;
        kp.sample = (float) is + delta_s;

        if (fabs(val) < _options.contrast_threshold
            || score < score_thres
            || score < 0.0f
            || fabs(delta_c) > 1.5f || fabs(delta_r) > 1.5f || fabs(delta_s) > 1.0f
            || kp.col < 0.0f || kp.col > w - 1
            || kp.row < 0.0f || kp.row > h - 1
            || kp.sample < -1.0f || kp.sample > _options.num_samples_per_octave
                )
            continue;

        _keypoints[num_keypoints++] = kp;
    }
    _keypoints.resize(num_keypoints);
    if (_options.debug_output && num_singular > 0) {
        std::cout << "SIFT: Warning: " << num_singular
                  << " singular matrices detected!" << std::endl;
    }
}

void Sift::descriptor_generation() {
    if (_octaves.empty()) {
        throw std::runtime_error("Octaves not available");
    }
    if (_keypoints.empty()) return;
    _descriptors.clear();
    _descriptors.reserve(_keypoints.size() * 3 / 2);

    ///计算每个octave中每个高斯空间的梯度大小和梯度方向
    int octave_index = _keypoints[0].octave;
    Octave *octave = &_octaves[octave_index - _options.min_octave];
    generate_grad_ori_images(octave);
    for (int i = 0; i < _keypoints.size(); i++) {
        const Keypoint &kp = _keypoints[i];
        if (kp.octave > octave_index) {
            //当前关键点属于另一个octave
            octave->grad.clear();
            octave->ori.clear();
            octave_index = kp.octave;
            octave = &_octaves[octave_index - _options.min_octave];
            generate_grad_ori_images(octave);
        } else if (kp.octave < octave_index) throw std::runtime_error("Decreasing octave index");
        //统计直方图，找到特征主方向
        std::vector<float> orientations;
        orientations.reserve(8);
        orientation_assignment(kp, octave, orientations);
        for (int j = 0; j < orientations.size(); j++) {
            Descriptor desc;
            const float scale_factor = std::pow(2.0f, kp.octave);//octave变大尺度缩小两倍，特征描述时需要还原回来
            desc.x = scale_factor * (kp.col + 0.5f) - 0.5f;
            desc.y = scale_factor * (kp.row + 0.5f) - 0.5f;
            desc.scale = keypoint_absolute_scale(kp);
            desc.orientation = orientations[j];
            if (descriptor_assignment(kp, desc, octave)) _descriptors.push_back(desc);
        }

    }

}

/**
 *  计算octave中的梯度和方向
 * @param octave
 */

void Sift::generate_grad_ori_images(Sift::Octave *octave) {
    octave->grad.clear();
    octave->grad.reserve(octave->img.size());
    octave->ori.clear();
    octave->ori.reserve(octave->img.size());

    const int width = octave->img[0].cols;
    const int height = octave->img[0].rows;

    for (int i = 0; i < octave->img.size(); i++) {
        cv::Mat img = octave->img[i];
        cv::Mat grad = cv::Mat::zeros(cv::Size(width, height), CV_32F);
        cv::Mat ori = cv::Mat::zeros(cv::Size(width, height), CV_32F);
        ///opencv中 cv::Size(宽（列），高（行）)  cv::Mat.at(行，列)
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float dx = (img.at<float>(y, x + 1) - img.at<float>(y, x - 1)) * 0.5f;
                float dy = (img.at<float>(y + 1, x) - img.at<float>(y - 1, x)) * 0.5f;

                //梯度方向
                float atan2f = std::atan2(dy, dx);
                grad.at<float>(y, x) = std::sqrt(dx * dx + dy * dy);
                ori.at<float>(y, x) = atan2f < 0.0 ? atan2f + CV_PI * 2.0f : atan2f;
            }
        }
        octave->grad.push_back(grad);
        octave->ori.push_back(ori);
    }

}

/**
 * 找到主方向
 * @param kp
 * @param octave
 * @param orientations
 */
void Sift::orientation_assignment(const Sift::Keypoint &kp, const Sift::Octave *octave, std::vector<float> &orientations) {
    const int nbins = 36;//将2pi分为36个区域
    const float nbinsf = 36.0f;
    float hist[nbins] = {}; //36-bin 直方图
    const int ix = static_cast<int>(kp.col + 0.5f); //+0.5 是为了做到四舍五入
    const int iy = static_cast<int>(kp.row + 0.5f);
    const int is = static_cast<int>(std::round(kp.sample));
    const float sigma = keypoint_relative_scale(kp);
    cv::Mat grad(octave->grad[is + 1]);
    cv::Mat ori(octave->ori[is + 1]);
    const int width = grad.cols;
    const int height = grad.rows;

    float sigma_factor = 1.5f;
    int window = static_cast<int>(sigma * sigma_factor * 3.0f);
    if (ix < window || ix + window >= width || iy < window || iy + window >= height) {
        return;
    }
    const float dxf = kp.col - static_cast<float>(ix);
    const float dyf = kp.row - static_cast<float>(iy);
    const float maxdist = (window * window) + 0.5f;

    for (int dy = -window; dy <= window; dy++) {
        for (int dx = -window; dx <= window; dx++) {
            float dist = powf(dx - dxf, 2) + powf(dy - dyf, 2);
            if (dist > maxdist) continue; // 内切圆内有效
            float gm = grad.at<float>(iy + dy, ix + dx);
            float go = ori.at<float>(iy + dy, ix + dx);
            float gaussian_sigma = sigma * sigma_factor;
            int gaussian_window = int(gaussian_sigma * 6) % 2 == 0 ? int(gaussian_sigma * 6) + 1 : int(gaussian_sigma * 6);
            cv::Mat gaussian_kernel = cv::getGaussianKernel(gaussian_window, gaussian_sigma, CV_32F);
            cv::Mat gaussian_kernel2D = gaussian_kernel * gaussian_kernel.t();
            int gaussian_center = static_cast<int>(gaussian_window / 2) + 1;
            float weight = gaussian_kernel2D.at<float>(gaussian_center + dy, gaussian_center + dx);
            int bin = static_cast<int>(nbinsf * go / (2.0f * CV_PI));
            if (bin < 0) bin = 0;
            else if (bin > nbins - 1) bin = nbins - 1;
            hist[bin] += gm * weight;

        }
    }

    // Smooth histogram
    for (int i = 0; i < 6; i++) {
        float first = hist[0];
        float prev = hist[nbins - 1];
        for (int j = 0; j < nbins - 1; j++) {
            float current = hist[j];
            hist[j] = (prev + current + hist[j + 1]) / 3.0f;
            prev = current;
        }
        hist[nbins - 1] = (prev + hist[nbins - 1] + first) / 3.0f;
    }

    //选出主方向
    float maxh = *std::max_element(hist, hist + nbins);
    for (int i = 0; i < nbins; i++) {
        float h0 = hist[(i + nbins - 1) % nbins];
        float h1 = hist[i];
        float h2 = hist[(i + 1) % nbins];
        //局部最大值
        if (h1 <= 0.8f * maxh || h1 <= h0 || h1 <= h2) {
            continue;
        }
        /*
         * f(x) = ax^2 + bx + c, f(-1) = h0, f(0) = h1, f(1) = h2
         * --> a = 1/2 (h0 - 2h1 + h2), b = 1/2 (h2 - h0), c = h1.
         * x = f'(x) = 2ax + b = 0 --> x = -1/2 * (h2 - h0) / (h0 - 2h1 + h2)
         */
        float x = -0.5f * (h2 - h0) / (h0 - 2.0f * h1 + h2);
        float o = 2.0 * CV_PI * (x + float(i) + 0.5f) / nbinsf;
        orientations.push_back(o);
    }
}


/*
 * scale = sigma0 * 2^(octave + (s+1) / S)
 */
float Sift::keypoint_relative_scale(const Sift::Keypoint &kp) const {
    return _options.base_blur_sigma * std::pow(2.0f, (kp.sample + 1.0f) / _options.num_samples_per_octave);
}

float Sift::keypoint_absolute_scale(const Keypoint &kp) const {
    return _options.base_blur_sigma * std::pow(2.0f, kp.octave + (kp.sample + 1.0f) / _options.num_samples_per_octave);
}

bool Sift::descriptor_assignment(const Sift::Keypoint &kp, Sift::Descriptor &desc, const Sift::Octave *octave) {
    const int PXB = 4; // Pixel bins with 4*4 bins
    const int OHB = 8;// Orientation histogram with 8 bins

    const int ix = static_cast<int>(kp.col + 0.5f);
    const int iy = static_cast<int>(kp.row + 0.5f);
    const int is = static_cast<int>(std::round(kp.sample));
    const float dxf = kp.col - static_cast<float>(ix);
    const float dyf = kp.row - static_cast<float>(iy);
    const float sigma = keypoint_relative_scale(kp);

    cv::Mat grad(octave->grad[is + 1]);
    cv::Mat ori(octave->ori[is + 1]);
    const int width = grad.cols;
    const int height = grad.rows;
    memset(desc.data, 0, sizeof desc.data);

    const float sino = std::sin(0);
    const float coso = std::cos(0);

    /*
     * Compute window size.
     * Each spacial bin has an extension of 3 * sigma (sigma is the scale
     * of the keypoint). For interpolation we need another half bin at
     * both ends in each dimension. And since the window can be arbitrarily
     * rotated, we need to multiply with sqrt(2). The window size is:
     * 2W = sqrt(2) * 3 * sigma * (PXB + 1).
     */
    const float binsize = 3.0f * sigma; // 每个bin的大小
    int win = sqrt(2) * binsize * (float) (PXB + 1) * 0.5f;
    if (ix < win || ix + win >= width || iy < win || iy + win >= height) return false;
    for (int dy = -win; dy <= win; dy++) {
        for (int dx = -win; dx <= win; dx++) {
            float const mod = grad.at<float>(iy + dy, ix + dx);
            float const angle = ori.at<float>(iy + dy, ix + dx);
            float theta = angle - desc.orientation; // 当前像素梯度方向与主方向差值
            if (theta < 0.0f) theta += 2.0f * CV_PI;

            const float winx = (float) dx - dxf;
            const float winy = (float) dy - dyf;

            float binoff = (float) (PXB - 1) / 2.0f;
            float binx = (coso * winx + sino * winy) / binsize + binoff;
            float biny = (-sino * winx + coso * winy) / binsize + binoff;
            float bint = theta * (float) OHB / (2.0f * CV_PI) - 0.5f;
            float gaussian_sigma = 0.5 * (float) PXB;
            int gaussian_window = int(gaussian_sigma * 6) % 2 == 0 ? int(gaussian_sigma * 6) + 1 : int(gaussian_sigma * 6);
            cv::Mat gaussian_kernel = cv::getGaussianKernel(gaussian_window, gaussian_sigma, CV_32F);
            cv::Mat gaussian_kernel2D = gaussian_kernel * gaussian_kernel.t();
            int gaussian_center = static_cast<int>(gaussian_window / 2) + 1;

            float gaussian_weight = gaussian_kernel2D.at<float>(gaussian_center + (biny - binoff), gaussian_center + (binx - binoff));
            float contrib = mod * gaussian_weight;

            /*
           * Distribute values into bins (using trilinear interpolation).
           * Each sample is inserted into 8 bins. Some of these bins may
           * not exist, because the sample is outside the keypoint window.
           */
            int bxi[2] = {(int) std::floor(binx), (int) std::floor(binx) + 1};
            int byi[2] = {(int) std::floor(biny), (int) std::floor(biny) + 1};
            int bti[2] = {(int) std::floor(bint), (int) std::floor(bint) + 1};

            float weights[3][2] = {
                    {(float) bxi[1] - binx, 1.0f - ((float) bxi[1] - binx)},
                    {(float) byi[1] - biny, 1.0f - ((float) byi[1] - biny)},
                    {(float) bti[1] - bint, 1.0f - ((float) bti[1] - bint)}
            };
            //0-(OHB-1) 循环
            if (bti[0] < 0) bti[0] += OHB;
            if (bti[1] >= 0) bti[1] -= OHB;

            const int xstride = OHB;
            const int ystride = OHB * PXB;
            for (int y = 0; y < 2; y++) {
                for (int x = 0; x < 2; x++) {
                    for (int t = 0; t < 2; t++) {
                        if (bxi[x] < 0 || bxi[x] >= PXB || byi[y] < 0 || byi[y] >= PXB) continue;
                        int idx = bti[t] + bxi[x] * xstride + byi[y] * ystride;
                        std::cout << "idx: " << idx << std::endl;
                        desc.data[idx] += contrib * weights[0][x] * weights[1][y] * weights[2][t];
                    }
                }
            }
        }
    }


    //normalize
    float init = 0;
    for (float i : desc.data) {
        init = init + i * i;
    }
    for (float & i : desc.data) {
        i /= std::sqrt(init);
    }

    /* Truncate descriptor values to 0.2. */
    for (float & i : desc.data) {
        i = std::min(i, 0.2f);
    }

    //normalize
    for (float i : desc.data) {
        init = init + i * i;
    }
    for (float & i : desc.data) {
        i /= std::sqrt(init);
    }

    return true;
}
