// Lucyte Created on: 17.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "refinement.hpp"

#ifdef _MSC_VER
#define X_Query_perf_counter() _Query_perf_counter()
#define X_Query_perf_frequency() _Query_perf_frequency()
#else
#define X_Query_perf_counter() 0
#define X_Query_perf_frequency() 1
#endif

const int SEED = 0x13337;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

float frrand(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad * 2.f - rad;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<float>& errors, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<float>& lastFrameErrors) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    for (int i = 0; i < keyPoints.size(); i++) {
        if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
            const float sx = 2.f;
            const float sy = 4.f;
            if (distance >= MAXVARIANCEINPIXELS) {
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) - sx * cosf(a), keyPoints[i].y - sy * cosf(a) + sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) + sx * cosf(a), keyPoints[i].y - sy * cosf(a) - sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
            else {
                cv::circle(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), (int)floorf(MAXVARIANCEINPIXELS), cv::Scalar(255, 255, 255));
            }
        }
    }
    imshow(windowName, mat);
    return mat;
}

std::vector<cv::Mat> mipMaps(const cv::Mat& mat) {
    cv::Mat k;
    cv::cvtColor(mat, k, cv::COLOR_RGB2GRAY);
    int width = k.cols;
    int height = k.rows;
    std::vector<cv::Mat> mipmaps;
    while (k.cols > 4 && k.rows > 4) {
        cv::Mat clone = k.clone();
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size((int)floorf((float)k.cols * MIPSCALE), (int)floorf((float)k.rows * MIPSCALE)), 0.f, 0.f, cv::INTER_AREA);
    }
    return mipmaps;
}

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

int main(int argc, char** argv)
{
    defaultDescriptorShape;
    srand(SEED);

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    int mipLevels = (int)floorf(MIPEND * (float)mipmaps1.size());

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    keyPoints.resize(KEYPOINTCOUNT);
    errors.resize(KEYPOINTCOUNT);
    for (int i = 0; i < (int)keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2((float)mipmaps1[0].cols);
        keyPoints[i].y = frrand2((float)mipmaps1[0].rows);
    }

    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());
    for (int i = mipLevels-1; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptors[i].resize(keyPoints.size());
        for (int j = 0; j < keyPoints.size(); ++j) {
            sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps1[i].data, descriptorScale, width, height, mipScale);
        }
    }

    long long t0 = X_Query_perf_counter();;
    long long t00 = X_Query_perf_counter();;
    long long fr = X_Query_perf_frequency();
    for (int steps = firstFrame+1; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;
        t00 = X_Query_perf_counter();
        for (int v = 0; v < (CHECKVARIANCE ? 2 : 1); v++) {
#pragma omp parallel for num_threads(64)
            for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                //KeyPoint kp = { mipmaps2[0].cols * 0.5f, mipmaps2[0].rows * 0.5f };
                KeyPoint kp = keyPoints[j];
                for (int i = mipLevels-1; i >= 0; i--) {
                    const int width = mipmaps2[i].cols;
                    const int height = mipmaps2[i].rows;
                    for (int k = 0; k < STEPCOUNT; k++) {
                        const float mipScale = powf(MIPSCALE, float(i));
                        float descriptorScale = 1.f / mipScale;
                        const float step = STEPSIZE * descriptorScale;
                        descriptorScale *= (1.f + frrand(SCALEINVARIANCE));
                        const float angle = frrand(ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                        kp = refineKeyPoint(BOOLSTEPPING, kp, searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, angle, step, width, height, mipScale);
                    }
                }
                switch (v) {
                case 0: keyPoints[j] = kp; 
                case 1: errors[j] = sqrt((kp.x - keyPoints[j].x) * (kp.x - keyPoints[j].x) + (kp.y - keyPoints[j].y) * (kp.y - keyPoints[j].y)); keyPoints[j].x = (kp.x + keyPoints[j].x) * 0.5f; keyPoints[j].y = (kp.y + keyPoints[j].y) * 0.5f; break;
                }
            }
        }
        long long t1 = X_Query_perf_counter();

        cv::Mat v =output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27) 
            break;

        long long t2 = X_Query_perf_counter();
        printf("Frame: %d, KeyPoints: %d, Overall seconds: %f, Feature refinement seconds: %f\n", steps, validKeyPoints, double(t2 - t0) / fr, double(t1 - t00) / fr);
        t0 = X_Query_perf_counter();

        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint &k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width-RIGHT) || (k.y < TOP) || (k.y >= height-BOTTOM) || (errors[j] >= MAXVARIANCEINPIXELS)) {
                    k.x = frrand2((float)width);
                    k.y = frrand2((float)height);
                    errors[j] = 0;
                }
            }
        }

        const bool resample = true;
        if (resample) {
            for (int i = mipLevels-1; i >= 0; i--) {
                const float mipScale = powf(MIPSCALE, float(i));
                const float descriptorScale = 1.f / mipScale;
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
#pragma omp parallel for num_threads(64)
                for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                    if ((!RESAMPLEONVARIANCE)||(errors[j] < RESAMPLEONVARIANCERADIUS))
                        sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, width, height, mipScale);
                }
            }
        }
    }

    if (outputVideo) video.release();

    return 0;
}
