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
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) - sx * cosf(a), keyPoints[i].y - sy * cosf(a) + sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(keyPoints[i].x - sy * sinf(a) + sx * cosf(a), keyPoints[i].y - sy * cosf(a) - sx * sinf(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), cv::Point2f(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
            }
            else {
                cv::circle(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), (int)(floorf(MAXVARIANCEINPIXELS)), cv::Scalar(255, 255, 255));
            }
        }
    }
    imshow(windowName, mat);
    return mat;
}

std::vector<MipMap> mipMaps(const std::vector<cv::Mat>& m) { std::vector<MipMap> r; r.resize(m.size()); for (int i = 0; i < r.size(); i++) { r[i].width = m[i].cols; r[i].height = m[i].rows; r[i].data = m[i].data; } return r; }
std::vector<cv::Mat> mipMaps(const cv::Mat& mat) {
    cv::Mat k;
    cv::cvtColor(mat, k, cv::COLOR_RGB2GRAY);
    int width = k.cols;
    int height = k.rows;
    std::vector<cv::Mat> mipmaps;
    while (k.cols > 4 && k.rows > 4) {
        cv::Mat clone = k.clone();
        mipmaps.push_back(clone);
        cv::resize(k, k, cv::Size((int)(floorf((float)k.cols * MIPSCALE)), (int)(floorf((float)k.rows * MIPSCALE))), 0.f, 0.f, cv::INTER_AREA);
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

    initOpenCL();
    srand(SEED);

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(thisBuffer, 0, mipMaps(mipmaps1));
    uploadMipMaps_openCL(lastBuffer, 0, mipMaps(mipmaps1));
    int mipLevels = (int)(floorf(MIPEND * float(mipmaps1.size())));

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    keyPoints.resize(KEYPOINTCOUNT);
    errors.resize(KEYPOINTCOUNT);
    for (int i = 0; i < keyPoints.size(); ++i) {
        keyPoints[i].x = frrand2((float)mipmaps1[0].cols);
        keyPoints[i].y = frrand2((float)mipmaps1[0].rows);
    }
    uploadKeyPoints_openCL(thisBuffer, 0, keyPoints);
    uploadKeyPoints_openCL(lastBuffer, 0, keyPoints);

    sampleDescriptors_openCL(thisBuffer, 0, 0, 0, mipLevels, 1.f, MIPSCALE);
    sampleDescriptors_openCL(lastBuffer, 0, 0, 0, mipLevels, 1.f, MIPSCALE);
    sampleDescriptors_openCL_waitfor(thisBuffer, 0, 0, 0, mipLevels, searchForDescriptors);
    sampleDescriptors_openCL_waitfor(lastBuffer, 0, 0, 0, mipLevels, searchForDescriptors);
    uploadDescriptors_openCL(thisBuffer, 0, mipLevels, searchForDescriptors);
    uploadDescriptors_openCL(lastBuffer, 0, mipLevels, searchForDescriptors);

    long long t0 = X_Query_perf_counter();;
    long long t2 = X_Query_perf_counter();;
    long long t3 = X_Query_perf_counter();;
    long long t00 = X_Query_perf_counter();;
    long long fr = X_Query_perf_frequency();
    cv::Mat mat2[2];
    bool first = true;
    refineKeyPoints_openCL(thisBuffer, 0,0,0, (int)keyPoints.size(), mipLevels, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
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
        uploadMipMaps_openCL(thisBuffer, 0, mipMaps(mipmaps2[thisBuffer]));
        t00 = X_Query_perf_counter();
        {
            sampleDescriptors_openCL_waitfor(thisBuffer, 0, 0, 0, mipLevels, resampledDescriptors);
            for (int i = mipLevels-1; i >= 0; i--)
                for (int j = (int)keyPoints.size() - 1; j >= 0; j--) 
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS))
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
            uploadDescriptors_openCL(thisBuffer, 0, mipLevels, searchForDescriptors);
        }
        refineKeyPoints_openCL(thisBuffer, 0,0,0, (int)keyPoints.size(), mipLevels, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
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
            for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
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
                    k.x = frrand2((float)width);
                    k.y = frrand2((float)height);
                    errors[j] = 0;
                }
            }
        }
        
        uploadKeyPoints_openCL(lastBuffer, 0, keyPoints);

        t2 = X_Query_perf_counter();
        const bool resample = true;
        if (resample && (!mipmaps2[thisBuffer].empty())) {
            sampleDescriptors_openCL(lastBuffer, 0, 0, 0, mipLevels, 1.f, MIPSCALE);
        }
        t3 = X_Query_perf_counter();
    }

    if (outputVideo) video.release();

    return 0;
}
