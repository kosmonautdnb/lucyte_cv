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
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;
const int DESCRIPTIVITYSTEPS = 10;
const float OUTLIERPERCENTAGE = 20.f;
const float OUTLIEREXPAND = 200.f;

std::vector<cv::Mat> mipmaps1;
std::vector<cv::Mat> mipmaps2;

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

float nonOutlierLength(std::vector<KeyPoint>& keyPoints, std::vector<KeyPoint>& lastFrameKeyPoints, const float outlierPercent = OUTLIERPERCENTAGE, const float expand = OUTLIEREXPAND) {
    float maxLength = 0.f;
    if (!keyPoints.empty()) {
        std::vector<float> lengths; lengths.resize(keyPoints.size());
        for (int i = 0; i < keyPoints.size(); i++) {
            float dx = keyPoints[i].x - lastFrameKeyPoints[i].x;
            float dy = keyPoints[i].y - lastFrameKeyPoints[i].y;
            lengths[i] = sqrt(dx * dx + dy * dy);
        }
        std::sort(lengths.begin(), lengths.end(), [](const float& a, const float& b)->bool {return a < b; });
        const int cut = int(lengths.size() * (1.0 - outlierPercent * 0.01f));
        maxLength = lengths[cut] + expand;
    }
    return maxLength;
}

int validKeyPoints = 0;
cv::Mat output(const std::string& windowName, const cv::Mat& image, std::vector<KeyPoint>& keyPoints, std::vector<float>& errors, std::vector<KeyPoint>& lastFrameKeyPoints, std::vector<float>& lastFrameErrors) {
    cv::Mat mat = image.clone();
    validKeyPoints = 0;
    const float maxLength = nonOutlierLength(keyPoints, lastFrameKeyPoints);
    for (int i = 0; i < keyPoints.size(); i++) {
        if (errors[i] < MAXVARIANCEINPIXELS && lastFrameErrors[i] < MAXVARIANCEINPIXELS) {
            float distanceX = lastFrameKeyPoints[i].x - keyPoints[i].x;
            float distanceY = lastFrameKeyPoints[i].y - keyPoints[i].y;
            float distance = sqrtf(distanceX * distanceX + distanceY * distanceY);
            if (distance < maxLength) {
                validKeyPoints++;
                const float a = atan2(keyPoints[i].x - lastFrameKeyPoints[i].x, keyPoints[i].y - lastFrameKeyPoints[i].y);
                const float sx = 2.f;
                const float sy = 4.f;
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) - sx * cos(a), keyPoints[i].y - sy * cos(a) + sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(keyPoints[i].x - sy * sin(a) + sx * cos(a), keyPoints[i].y - sy * cos(a) - sx * sin(a)), cv::Scalar(255, 255, 255));
                cv::line(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), cv::Point(lastFrameKeyPoints[i].x, lastFrameKeyPoints[i].y), cv::Scalar(255, 255, 255));
                cv::circle(mat, cv::Point(keyPoints[i].x, keyPoints[i].y), 2.0, cv::Scalar(255, 255, 255));
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

float descriptivity(std::vector<cv::Mat>& mipMaps, const KeyPoint& k, const int mipEnd) {
    Descriptor d;
    float h = 0;
    for (int i = mipEnd; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipMaps[i].cols;
        const int height = mipMaps[i].rows;
        h += sampleDescriptor(k, d, mipMaps[i].data, descriptorScale, width, height, mipScale)*mipScale;
    }
    h /= float(mipEnd + 1);
    return h;
}

cv::Mat loadImage(int frame) {
    char buffer[2000];
    snprintf(buffer, 2000, fileNames, frame);
    return cv::imread(buffer, cv::IMREAD_COLOR);
}

int main(int argc, char** argv)
{
    srand(SEED);
    defaultDescriptorShape(DESCRIPTORSCALE);
    initOpenCL();
    uploadDescriptorShape_openCL();

    cv::Mat mat1 = loadImage(firstFrame);
    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), outputVideoFrameRate / double(frameStep), cv::Size(mat1.cols, mat1.rows), true);
    mipmaps1 = mipMaps(mat1);
    uploadMipMaps_openCL(mipmaps1);
    int mipEnd = MIPEND * (mipmaps1.size() - 1);

    std::vector<KeyPoint> keyPoints;
    std::vector<float> errors;
    keyPoints.resize(KEYPOINTCOUNT);
    errors.resize(KEYPOINTCOUNT);
#pragma omp parallel for num_threads(64)
    for (int i = 0; i < keyPoints.size(); ++i) {
        float dBest = -1;
        KeyPoint kHere, kBest;
        for (int t = 0; t < DESCRIPTIVITYSTEPS; t++) {
            kHere.x = frrand2(mipmaps1[0].cols);
            kHere.y = frrand2(mipmaps1[0].rows);
            float d = descriptivity(mipmaps1, kHere, mipEnd);
            if (d > dBest) {
                dBest = d;
                kBest = kHere;
            }
        }
        keyPoints[i] = kBest;
    }
    uploadKeyPoints_openCL(keyPoints);

    std::vector<std::vector<Descriptor>> searchForDescriptors;
    searchForDescriptors.resize(mipmaps1.size());
    for (int i = mipEnd; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipmaps1[i].cols;
        const int height = mipmaps1[i].rows;
        searchForDescriptors[i].resize(keyPoints.size());
        sampleDescriptors_openCL(i,searchForDescriptors, descriptorScale, width, height, mipScale);
        uploadDescriptors_openCL(i, searchForDescriptors);
    }

    long long t0 = X_Query_perf_counter();;
    long long t2 = X_Query_perf_counter();;
    long long t3 = X_Query_perf_counter();;
    long long t00 = X_Query_perf_counter();;
    long long fr = X_Query_perf_frequency();
    int readded = 0;
    for (int steps = firstFrame; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps); mipmaps2 = mipMaps(mat2);  uploadMipMaps_openCL(mipmaps2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;

        t00 = X_Query_perf_counter();
        refineKeyPoints_openCL(keyPoints, errors, mipEnd, STEPCOUNT, BOOLSTEPPING, MIPSCALE, STEPSIZE, SCALEINVARIANCE, ROTATIONINVARIANCE);
        long long t1 = X_Query_perf_counter();

        cv::Mat v = output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("(OpenCL) Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT) + ", readd " + std::to_string(readded));
        readded = 0;
        if (cv::waitKey(1) == 27) 
            break;

        long long t4 = X_Query_perf_counter();
        printf("Frame: %d, Keypoint: %d, Overall seconds: %f, Feature refinement seconds: %f, Feature add seconds:%f\n", steps, validKeyPoints, double(t4 - t0) / fr, double(t1 - t00) / fr, double(t3 - t2) / fr);
        t0 = X_Query_perf_counter();

        t2 = X_Query_perf_counter();
        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            static int rlk = 0;
            for (int j = keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint &k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM) || errors[j] >= MAXVARIANCEINPIXELS) {
                    float dBest = -1;
                    KeyPoint kHere, kBest;
#pragma omp parallel for num_threads(64)
                    for (int t = 0; t < DESCRIPTIVITYSTEPS; t++) {
                        kHere.x = randomLike(rlk)*mipmaps1[0].cols;
                        kHere.y = randomLike(rlk*3+11231)*mipmaps1[0].rows;
                        rlk += 123;
                        float d = descriptivity(mipmaps1, kHere, mipEnd);
                        if (d > dBest) {
                            dBest = d;
                            kBest = kHere;
                        }
                    }
                    k = kBest;
                    readded++;
                    errors[j] = 0;
                }
            }
            t3 = X_Query_perf_counter();
        }

        uploadKeyPoints_openCL(keyPoints);

        const bool resample = true;
        if (resample) {
            std::vector<std::vector<Descriptor>> resampledDescriptors;
            resampledDescriptors.resize(mipmaps2.size());
            for (int i = mipEnd; i >= 0; i--) {
                const float mipScale = powf(MIPSCALE, float(i));
                const float descriptorScale = 1.f / mipScale;
                const int width = mipmaps2[i].cols;
                const int height = mipmaps2[i].rows;
                resampledDescriptors[i].resize(keyPoints.size());
                sampleDescriptors_openCL(i,resampledDescriptors, descriptorScale, width, height, mipScale);
                for (int j = keyPoints.size() - 1; j >= 0; j--) {
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS)) {
                        searchForDescriptors[i][j] = resampledDescriptors[i][j];
                    }
                }
                uploadDescriptors_openCL(i, searchForDescriptors);
            }
        }
    }

    if (outputVideo) video.release();

    return 0;
}
