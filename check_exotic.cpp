// Lucyte Created on: 21.01.2025 by Stefan Mader
#include "config.hpp"
#include "refinement.hpp"
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

const int SEED = 0x13337;
const bool CHECKVARIANCE = true;
const float MAXVARIANCEINPIXELS = 1.0;
const float MIPEND = 1.0;
const bool RESAMPLEONVARIANCE = true;
const float RESAMPLEONVARIANCERADIUS = 1.f;

#define FORWARD_BITCHECK  if (b1 && (!b2))
#define BACKWARD_BITCHECK if ((!b1) && b2)
#define _EA -=
#define _EB +=
#define STEPSIGNEDPOWER(__i0ad, __i1ad) (abs(__i0ad) > abs(__i1ad) ? sign(__i0ad) : 0)

#define COMBINESTEP_FEATUREREFINE\
    FORWARD_BITCHECK {\
        movementX _EA STEPSIGNEDPOWER(i0adx, i1adx);\
        movementY _EA STEPSIGNEDPOWER(i0ady, i1ady);\
        movementX _EB (float)signcmp(i1adx, i0adx);\
        movementY _EB (float)signcmp(i1ady, i0ady);\
    }\
    BACKWARD_BITCHECK {\
        movementX _EA signcmp(i1adx, i0adx);\
        movementY _EA signcmp(i1ady, i0ady);\
        movementX _EB (float)STEPSIGNEDPOWER(i0adx, i1adx);\
        movementY _EB (float)STEPSIGNEDPOWER(i0ady, i1ady);\
    }

#define SILVESBIFTFACTORX (fabsf( (float)(i0adx - i1adx == 0 ? 1 : i0adx - i1adx) )/256.f)
#define SILVESBIFTFACTORY (fabsf( (float)(i0ady - i1ady == 0 ? 1 : i0ady - i1ady) )/256.f)
#define SILVESBIFT(__partx, __party, __eforma, __eformb, __forward)\
    FORWARD_BITCHECK {\
        if (!(__forward)) {movementX __eforma SILVESBIFTFACTORX; movementY __eforma SILVESBIFTFACTORY;}\
        movementX _EA (float)(__partx);\
        movementY _EA (float)(__party);\
        if (__forward) {movementX __eforma SILVESBIFTFACTORX; movementY __eforma SILVESBIFTFACTORY;}\
    }\
    BACKWARD_BITCHECK {\
        if (__forward) {movementX __eformb SILVESBIFTFACTORX; movementY __eformb SILVESBIFTFACTORY;}\
        movementX _EB (float)(__partx);\
        movementY _EB (float)(__party);\
        if (!(__forward)) {movementX __eformb SILVESBIFTFACTORX; movementY __eformb SILVESBIFTFACTORY;}\
    }
#define SILVESBIFT_FEATUREREFINE SILVESBIFT(signcmp(i0adx,i1adx), signcmp(i0ady,i1ady), *=, /=, true)

#define REFINEMENT_TEMPLATE(__NAME__,__FUNCTION__) KeyPoint __NAME__(const KeyPoint& kp, const Descriptor& toSearch, const unsigned char* s, const float descriptorScale, const float angle, const float step, const int width, const int height, const float mipScale) {\
    KeyPoint kp2 = kp;\
    kp2.x *= mipScale;\
    kp2.y *= mipScale;\
    const float descriptorSize = mipScale * descriptorScale;\
    const float sina = sinf(angle);\
    const float cosa = cosf(angle);\
    const float sinad = sina * descriptorSize;\
    const float cosad = cosa * descriptorSize;\
    const float gdxx = cosa; const float gdxy = sina;\
    const float gdyx = -sina; const float gdyy = cosa;\
    float movementX = 0;\
    float movementY = 0;\
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {\
        const float d1x = kp2.x + (cosad * descriptorsX1[b] + sinad * descriptorsY1[b]);\
        const float d1y = kp2.y + (-sinad * descriptorsX1[b] + cosad * descriptorsY1[b]);\
        const float d2x = kp2.x + (cosad * descriptorsX2[b] + sinad * descriptorsY2[b]);\
        const float d2y = kp2.y + (-sinad * descriptorsX2[b] + cosad * descriptorsY2[b]);\
        const int l1 = tex(s, d1x, d1y, width, height);\
        const int l2 = tex(s, d2x, d2y, width, height);\
        int b1 = (toSearch.bits[b / 32] >> (b & 31)) & 1;\
        int b2 = (l2 < l1 ? 1 : 0);\
        if (b2 != b1) {\
            const int i0adx = tex(s, d1x - gdxx, d1y - gdxy, width, height) - tex(s, d1x + gdxx, d1y + gdxy, width, height);\
            const int i0ady = tex(s, d1x - gdyx, d1y - gdyy, width, height) - tex(s, d1x + gdyx, d1y + gdyy, width, height);\
            const int i1adx = tex(s, d2x - gdxx, d2y - gdxy, width, height) - tex(s, d2x + gdxx, d2y + gdxy, width, height);\
            const int i1ady = tex(s, d2x - gdyx, d2y - gdyy, width, height) - tex(s, d2x + gdyx, d2y + gdyy, width, height);\
            __FUNCTION__\
        }\
    }\
    float l = movementX * movementX + movementY * movementY;\
    if (l > 0.f) {\
        l = sqrtf(l);\
        movementX /= l;\
        movementY /= l;\
    }\
    const float stepSize = descriptorScale * step;\
    KeyPoint r;\
    r.x = kp.x + (cosa * movementX + sina * movementY) * stepSize;\
    r.y = kp.y + (-sina * movementX + cosa * movementY) * stepSize;\
    const bool clipping = false; if (clipping) {\
        const float w = floorf(float(width) / mipScale);\
        const float h = floorf(float(height) / mipScale);\
        if (r.x < 0) r.x = 0;\
        if (r.y < 0) r.y = 0;\
        if (r.x >= w - 1) r.x = w - 1;\
        if (r.y >= h - 1) r.y = h - 1;\
    }\
    return r;\
}

inline int signcmp(int f1, int f2) {
    return f1 > f2 ? 1 : -1;
}

inline int sign(int f) {
    return f < 0 ? -1 : f > 0 ? 1 : 0;
}

REFINEMENT_TEMPLATE(refineKeyPoint_combinestep, COMBINESTEP_FEATUREREFINE)
REFINEMENT_TEMPLATE(refineKeyPoint_silvesbift, SILVESBIFT_FEATUREREFINE)
#define REFINEMENTFUNCTION refineKeyPoint_combinestep

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
                cv::circle(mat, cv::Point2f(keyPoints[i].x, keyPoints[i].y), (int)(floorf(MAXVARIANCEINPIXELS)), cv::Scalar(255, 255, 255));
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
    for (int i = 0; i < keyPoints.size(); ++i) {
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

    for (int steps = firstFrame+1; steps <= lastFrame; steps += frameStep) {
        cv::Mat mat2 = loadImage(steps);
        mipmaps2 = mipMaps(mat2);
        std::vector<KeyPoint> lastFrameKeyPoints = keyPoints;
        std::vector<float> lastFrameErrors = errors;
        for (int v = 0; v < (CHECKVARIANCE ? 2 : 1); v++) {
#pragma omp parallel for num_threads(32)
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
                        kp = REFINEMENTFUNCTION(kp, searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, angle, step, width, height, mipScale);
                    }
                }
                switch (v) {
                case 0: keyPoints[j] = kp;
                case 1: errors[j] = sqrt((kp.x - keyPoints[j].x) * (kp.x - keyPoints[j].x) + (kp.y - keyPoints[j].y) * (kp.y - keyPoints[j].y)); keyPoints[j].x = (kp.x + keyPoints[j].x) * 0.5f; keyPoints[j].y = (kp.y + keyPoints[j].y) * 0.5f; break;
                }
            }
        }
        cv::Mat v = output("keypoints", mat2, keyPoints, errors, lastFrameKeyPoints, lastFrameErrors);
        if (outputVideo) video.write(v);
        cv::setWindowTitle("keypoints", std::string("Frame ") + std::to_string(steps - firstFrame) + " of " + std::to_string(lastFrame - firstFrame) + ", Keypoints " + std::to_string(validKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT));
        if (cv::waitKey(1) == 27)
            break;
        const bool readd = true;
        if (readd) {
            const int width = mipmaps2[0].cols;
            const int height = mipmaps2[0].rows;
            for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                KeyPoint& k = keyPoints[j];
                const float LEFT = 10;
                const float RIGHT = 10;
                const float TOP = 10;
                const float BOTTOM = 10;
                if ((k.x < LEFT) || (k.x >= width - RIGHT) || (k.y < TOP) || (k.y >= height - BOTTOM) || (errors[j] >= MAXVARIANCEINPIXELS)) {
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
#pragma omp parallel for num_threads(32)
                for (int j = (int)keyPoints.size() - 1; j >= 0; j--) {
                    if ((!RESAMPLEONVARIANCE) || (errors[j] < RESAMPLEONVARIANCERADIUS))
                        sampleDescriptor(keyPoints[j], searchForDescriptors[i][j], mipmaps2[i].data, descriptorScale, width, height, mipScale);
                }
            }
        }
    }

    if (outputVideo) video.release();

    return 0;
}
