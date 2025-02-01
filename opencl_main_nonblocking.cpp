// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "opencl_refinement.hpp"

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
const float MIPEND = 1;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2[2];
std::vector<std::vector<Descriptor>> searchForDescriptors;

const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / (float)0x10000;
}

float frrand(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad * 2.f - rad;
}

float frrand2(float rad) {
    return ((float)(rand() % RAND_MAX) / RAND_MAX) * rad;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, const std::vector<KeyPoint>& keyPoints, const std::vector<float>& errors, const std::vector<KeyPoint>& lastFrameKeyPoints, const std::vector<float>& lastFrameErrors) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    for (int i = 0; i < keyPoints.size(); i++) {
        if (errors[i] != 0 && lastFrameErrors[i] != 0 && errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS) {
            validKeyPoints++;
            float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
            const float sx = 2.f;
            const float sy = 4.f;
            if (distance >= MAXVARIANCEINPIXELS) {
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
            else {
                cv::circle(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), MAXVARIANCEINPIXELS, cv::Scalar(255, 255, 255));
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
        cv::resize(k, k, cv::Size(k.cols * MIPSCALE, k.rows * MIPSCALE), 0.f, 0.f, cv::INTER_AREA);
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
    int thisBuffer = 0, lastBuffer = thisBuffer ^ 1;

    srand(SEED);
    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL(thisBuffer);
    uploadDescriptorShape_openCL(lastBuffer);

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(thisBuffer, 0, mipmaps1);
    uploadMipMaps_openCL(lastBuffer, 0, mipmaps1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    keyPoints.resize(KEYPOINTCOUNT);
    errors.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2(mipmaps1[0].cols);
        keyPoints[i].y = frrand2(mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(thisBuffer, 0, keyPoints);
    uploadKeyPoints_openCL(lastBuffer, 0, keyPoints);

    searchForDescriptors.resize(mipmaps1.size());
    for (int i = mipEnd; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        sampleDescriptors_openCL(thisBuffer, 0,0,0, i,keyPoints.size(), descriptorScale, width, height, mipScale);
        sampleDescriptors_openCL(lastBuffer, 0,0,0, i, keyPoints.size(), descriptorScale, width, height, mipScale);
        sampleDescriptors_openCL_waitfor(thisBuffer, 0, i, searchForDescriptors);
        sampleDescriptors_openCL_waitfor(lastBuffer, 0, i, searchForDescriptors);
        uploadDescriptors_openCL(thisBuffer, 0, i, searchForDescriptors);
        uploadDescriptors_openCL(lastBuffer, 0, i, searchForDescriptors);
    }

    long long t0 = X_Query_perf_counter();;
    long long t2 = X_Query_perf_counter();;
    long long t3 = X_Query_perf_counter();;
    long long t00 = X_Query_perf_counter();;
    long long fr = X_Query_perf_frequency();
    cv::Mat mat2[2];
    bool first = true;
    refineKeyPoints_openCL(thisBuffer, 0,0,0, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
    std::vector<KeyPoint> lastlastFrameKeyPoints;
    std::vector<float> lastlastFrameErrors;
    std::vector<std::vector<Descriptor>> resampledDescriptors;
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        thisBuffer ^= 1;
        lastBuffer = thisBuffer ^ 1;
        lastlastFrameKeyPoints = keyPoints;
        lastlastFrameErrors = errors;
        mat2[thisBuffer] = loadImage(steps);
        mipmaps2[thisBuffer] = mipMaps(mat2[thisBuffer]);
        uploadMipMaps_openCL(thisBuffer, 0, mipmaps2[thisBuffer]);
        t00 = X_Query_perf_counter();
        if (!resampledDescriptors.empty()) {
            for (int i = mipEnd; i >= 0; i--) {
                sampleDescriptors_openCL_waitfor(thisBuffer, 0, i, resampledDescriptors);
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS)) {
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
                    }
                }
                uploadDescriptors_openCL(thisBuffer, 0, i, searchForDescriptors);
            }
        }
        refineKeyPoints_openCL(thisBuffer, 0,0,0, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        refineKeyPoints_openCL_waitfor(lastBuffer, 0, keyPoints, errors);
        long long t1 = X_Query_perf_counter();

        first = false;

        if (!mat2[lastBuffer].empty()) {
            cv::Mat v = output("keypoints", mat2[lastBuffer], keyPoints, errors, lastlastFrameKeyPoints, lastlastFrameErrors);
            if (outputVideo) video.write(v);
        }
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27) 
            break;

        long long t4 = X_Query_perf_counter();
        printf("Frame: %d, Keypoints: %d, Overall seconds: %f, Feature refinement seconds: %f, Feature add seconds:%f\n", steps, validKeyPoints, double(t4 - t0) / fr, double(t1 - t00) / fr, double(t3 - t2) / fr);
        t0 = X_Query_perf_counter();

        const bool readd = true;
        if (readd && !(mipmaps2[lastBuffer].empty())) {
            const int width = mipmaps2[lastBuffer][0].cols;
            const int height = mipmaps2[lastBuffer][0].rows;
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint &k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if (lastlastFrameErrors[j] == 0) {
                    k.x = lastlastFrameKeyPoints[j].x;
                    k.y = lastlastFrameKeyPoints[j].y;
                } else
                if ((k.x < LEFT) || (k.x >= width-RIGHT) || (k.y < TOP) || (k.y >= height-BOTTOM) || errors[j] >= MAXVARIANCEINPIXELS) {
                    k.x = frrand2(width);
                    k.y = frrand2(height);
                    errors[j] = 0;
                }
            }
        }
        
        uploadKeyPoints_openCL(lastBuffer, 0, keyPoints);

        t2 = X_Query_perf_counter();
        const bool resample = true;
        if (resample && (!mipmaps2[thisBuffer].empty())) {
            resampledDescriptors.resize(mipmaps2[thisBuffer].size());
            for (int i = mipEnd; i >= 0; i--) {
                const float mipScale = powf(MIPSCALE, float(i));
                const float descriptorScale = 1.f / mipScale;
                const int width = mipmaps2[thisBuffer][i].cols;
                const int height = mipmaps2[thisBuffer][i].rows;
                sampleDescriptors_openCL(lastBuffer, 0,0,0, i, keyPoints.size(), descriptorScale, width, height, mipScale);
            }
        }
        t3 = X_Query_perf_counter();
    }

    if (outputVideo) video.release();

    return 0;
}
