// Lucyte Created on: 19.01.2025 by Stefan Mader
#include "config.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "refinement.hpp"

const int SEED = 0x13337;
const float MAXVARIANCEINPIXELS = 1.f;
const float MARGIN = 0.1f;
const double OUTLIERDISTANCE = 20;
const double RANDOMX = 50;
const double RANDOMY = 50;
int randomLikeIndex = 0;

const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / 0x10000;
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

cv::Mat drawPoly(cv::Mat& canvas, const cv::Mat& image, const float x0, const float y0, const float x1, const float y1, const float x2, const float y2, const float x3, const float y3) {
    cv::Point2f source[4];
    cv::Point2f dest[4];
    const float sx1 = (float)image.cols;
    const float sy1 = (float)image.rows;
    source[0] = { 0,0 };
    source[1] = { sx1,0 };
    source[2] = { sx1,sy1 };
    source[3] = { 0,sy1 };
    dest[0] = { x0, y0 };
    dest[1] = { x1, y1 };
    dest[2] = { x2, y2 };
    dest[3] = { x3, y3 };
    cv::Mat mat = cv::getPerspectiveTransform(source, dest);
    cv::warpPerspective(image, canvas, mat, cv::Size(canvas.cols, canvas.rows), cv::INTER_AREA);
    return mat;
}

KeyPoint getPointInPoly(float x, float y, const cv::Mat& perspective) {
    float xc = float(x * perspective.at<double>(0, 0) + y * perspective.at<double>(0, 1) + perspective.at<double>(0, 2));
    float yc = float(x * perspective.at<double>(1, 0) + y * perspective.at<double>(1, 1) + perspective.at<double>(1, 2));
    float zc = float(x * perspective.at<double>(2, 0) + y * perspective.at<double>(2, 1) + perspective.at<double>(2, 2));
    KeyPoint r;
    r.x = xc / zc;
    r.y = yc / zc;
    return r;
}

std::pair<KeyPoint,float> trackPoint(const std::vector<cv::Mat> & mipmapsSource, const std::vector<cv::Mat> &mipmapsDest, const KeyPoint &keyPointSource) {

    std::vector<Descriptor> searchForDescriptors;
    searchForDescriptors.resize(mipmapsSource.size());
    for (int i = (int)mipmapsSource.size()-1; i >= 0; i--) {
        const float mipScale = powf(MIPSCALE, float(i));
        const float descriptorScale = 1.f / mipScale;
        const int width = mipmapsSource[i].cols;
        const int height = mipmapsSource[i].rows;
        sampleDescriptor(keyPointSource, searchForDescriptors[i], mipmapsSource[i].data, descriptorScale, width, height, mipScale);
    }

    KeyPoint keyPoint;
    float error;
    for (int v = 0; v < 2; v++) {
//        KeyPoint kp = { float(mipmapsDest[0].cols) * 0.5f, float(mipmapsDest[0].rows) * 0.5f };
        KeyPoint kp = keyPointSource;
        for (int i = (int)mipmapsDest.size() - 1; i >= 0; i--) {
            const int width = mipmapsDest[i].cols;
            const int height = mipmapsDest[i].rows;
            for (int k = 0; k < STEPCOUNT; k++) {
                const float mipScale = powf(MIPSCALE, float(i));
                float descriptorScale = 1.f / mipScale;
                const float step = STEPSIZE * descriptorScale;
                descriptorScale *= 1.f + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.f - SCALEINVARIANCE;
                const float angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.f - ROTATIONINVARIANCE) / 360.f * 2 * 3.1415927f;
                kp = refineKeyPoint(kp, searchForDescriptors[i], mipmapsDest[i].data, descriptorScale, angle, step, width, height, mipScale);
            }
        }
        switch (v) {
        case 0: keyPoint = kp;
        case 1: error = sqrt((kp.x - keyPoint.x) * (kp.x - keyPoint.x) + (kp.y - keyPoint.y) * (kp.y - keyPoint.y)); keyPoint.x = (kp.x + keyPoint.x) * 0.5f; keyPoint.y = (kp.y + keyPoint.y) * 0.5f; break;
        }
    }
    return { keyPoint, error };
}

cv::Mat testFrame(const cv::Mat &image) {
    cv::Mat leftCanvas;
    cv::Mat rightCanvas;
    cv::Mat trackCanvas;
    cv::Mat kltLeftCanvas;
    cv::Mat kltRightCanvas;
    cv::Mat kltTrackCanvas;
    const int sx = 512;
    const int sy = 512;
    leftCanvas.create(cv::Size(sx, sy), CV_8UC3);
    rightCanvas.create(cv::Size(sx, sy), CV_8UC3);
    trackCanvas.create(cv::Size(sx, sy), CV_8UC3);
    kltLeftCanvas.create(cv::Size(sx, sy), CV_8UC3);
    kltRightCanvas.create(cv::Size(sx, sy), CV_8UC3);
    kltTrackCanvas.create(cv::Size(sx, sy), CV_8UC3);
    const float kx = RANDOMX;
    const float ky = RANDOMY;
    KeyPoint d[4] = {
        {randomLike(randomLikeIndex++) * kx,randomLike(randomLikeIndex++) * ky},
        {sx - randomLike(randomLikeIndex++) * kx,randomLike(randomLikeIndex++) * ky},
        {sx - randomLike(randomLikeIndex++) * kx,sy - randomLike(randomLikeIndex++) * ky},
        {randomLike(randomLikeIndex++) * kx,sy - randomLike(randomLikeIndex++) * ky},
    };
    cv::Mat baseImage = leftCanvas.clone(); cv::resize(image,baseImage,cv::Size(sx,sy));

    drawPoly(kltLeftCanvas, baseImage, 0, 0, sx, 0, sx, sy, 0, sy);
    drawPoly(kltRightCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);
    drawPoly(kltTrackCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);
    cv::Mat leftMat = drawPoly(leftCanvas, baseImage, 0, 0, sx, 0, sx, sy, 0, sy);
    cv::Mat rightMat = drawPoly(rightCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);
    cv::Mat trackMat = drawPoly(trackCanvas, baseImage, d[0].x, d[0].y, d[1].x, d[1].y, d[2].x, d[2].y, d[3].x, d[3].y);

    std::vector<KeyPoint> keyPointsSource;
    std::vector<KeyPoint> keyPointsGroundTruth;
    keyPointsSource.resize(KEYPOINTCOUNT);
    keyPointsGroundTruth.resize(KEYPOINTCOUNT);
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        keyPointsSource[i].x = randomLike(i*2+1234) * baseImage.cols * (1.f - MARGIN * 2.f) + MARGIN * baseImage.cols;
        keyPointsSource[i].y = randomLike(i*5+1234) * baseImage.rows * (1.f - MARGIN * 2.f) + MARGIN * baseImage.rows;
        KeyPoint k = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, leftMat);
        cv::circle(leftCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(0, 0, 255));
        cv::circle(kltLeftCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(0, 0, 255));
    }
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        KeyPoint k = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, rightMat);
        keyPointsGroundTruth[i] = k;
        cv::circle(rightCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(0, 0, 255));
        cv::circle(kltRightCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(0, 0, 255));
    }

    std::vector<cv::Mat> mipmapsSource = mipMaps(baseImage);
    std::vector<cv::Mat> mipmapsDest = mipMaps(trackCanvas);

    int validLucyteKeyPoints = 0;
    double lucyteError = 0;
    int lucyteErrors = 0;
#pragma omp parallel for num_threads(32)
    for (int i = 0; i < KEYPOINTCOUNT; i++) {
        std::pair<KeyPoint, float> kpv = trackPoint(mipmapsSource, mipmapsDest, keyPointsSource[i]);
        KeyPoint k = kpv.first;
        float variance = kpv.second;
        if (variance < MAXVARIANCEINPIXELS) {
            cv::circle(trackCanvas, cv::Point2f(k.x, k.y), 3, cv::Scalar(0, 0, 255), -1);
            KeyPoint k1 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, leftMat);
            cv::circle(leftCanvas, cv::Point2f(k1.x, k1.y), 3, cv::Scalar(0, 0, 255), -1);
            KeyPoint k2 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, rightMat);
            cv::circle(rightCanvas, cv::Point2f(k2.x, k2.y), 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(rightCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(255, 0, 0), -1);
            validLucyteKeyPoints++;
            double pixelErrorX = k2.x - k.x;
            double pixelErrorY = k2.y - k.y;
            double distance = sqrt(pixelErrorX * pixelErrorX + pixelErrorY * pixelErrorY);
            if (distance < OUTLIERDISTANCE) {
                cv::line(rightCanvas, cv::Point2f(k.x, k.y), cv::Point2f(k2.x, k2.y), cv::Scalar(255, 255, 255), 1);
                lucyteError += distance;
                lucyteErrors++;
            }
        }
    }
    lucyteError /= double(lucyteErrors);

    std::vector<cv::Point2f> cvKeyPoints;
    std::vector<cv::Point2f> cvResultKeyPoints;
    cv::Mat status;
    std::vector<float> err;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT)+(cv::TermCriteria::EPS), 10, 0.03);
    for (int i = 0; i < keyPointsSource.size(); i++) cvKeyPoints.push_back(cv::Point2f(keyPointsSource[i].x, keyPointsSource[i].y));
    cv::Mat m1, m2;
    cv::cvtColor(baseImage, m1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(kltTrackCanvas, m2, cv::COLOR_RGB2GRAY);
    cv::calcOpticalFlowPyrLK(m1, m2, cvKeyPoints, cvResultKeyPoints, status, err, cv::Size(15, 15), 2, criteria);

    int validKLTKeyPoints = 0;
    double kltError = 0;
    int kltErrors = 0;
    for (int i = 0; i < cvResultKeyPoints.size(); i++) {
        KeyPoint k;
        k.x = cvResultKeyPoints[i].x;
        k.y = cvResultKeyPoints[i].y;
        float variance = err[i];
        if (variance < 10.0) {
            validKLTKeyPoints++;
            cv::circle(kltTrackCanvas, cv::Point2f(k.x, k.y), 3, cv::Scalar(0, 0, 255), -1);
            KeyPoint k1 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, leftMat);
            cv::circle(kltLeftCanvas, cv::Point2f(k1.x, k1.y), 3, cv::Scalar(0, 0, 255), -1);
            KeyPoint k2 = getPointInPoly(keyPointsSource[i].x, keyPointsSource[i].y, rightMat);
            cv::circle(kltRightCanvas, cv::Point2f(k2.x, k2.y), 3, cv::Scalar(0, 0, 255), -1);
            cv::circle(kltRightCanvas, cv::Point2f(k.x, k.y), 2, cv::Scalar(255, 0, 0), -1);
            double pixelErrorX = k2.x - k.x;
            double pixelErrorY = k2.y - k.y;
            double distance = sqrt(pixelErrorX * pixelErrorX + pixelErrorY * pixelErrorY);
            if (distance < OUTLIERDISTANCE) {
                cv::line(kltRightCanvas, cv::Point2f(k.x, k.y), cv::Point2f(k2.x, k2.y), cv::Scalar(255, 255, 255), 1);
                kltError += distance;
                kltErrors++;
            }
        }
    }
    kltError /= double(kltErrors);

    std::string lucyteText = "Lucyte avg. pixel error:" + std::to_string(lucyteError);
    std::string kltText = "KLT avg. pixel error:" + std::to_string(kltError);
    for (int i = 0; i < 2; i++) {
        cv::Scalar color = i ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 0);
        const int font = cv::HersheyFonts::FONT_HERSHEY_DUPLEX;
        const float fontScale = 0.7f;
        const float yp = 30.f - (float)i * 1.f;
        const float xp = 5.f;
        cv::putText(leftCanvas, std::to_string(validLucyteKeyPoints) + " keypoints of " + std::to_string(KEYPOINTCOUNT), cv::Point2f(xp, yp), font, fontScale, color);
        cv::putText(rightCanvas, "GroundTruth and Lucyte", cv::Point2f(xp, yp), font, fontScale, color);
        cv::putText(trackCanvas, lucyteText, cv::Point2f(xp, yp), font, fontScale, color);
        cv::putText(kltLeftCanvas, std::to_string(validKLTKeyPoints) + " keypoints of " + std::to_string(KEYPOINTCOUNT), cv::Point2f(xp, yp), font, fontScale, color);
        cv::putText(kltRightCanvas, "GroundTruth and KLT", cv::Point2f(xp, yp), font, fontScale, color);
        cv::putText(kltTrackCanvas, kltText, cv::Point2f(xp, yp), font, fontScale, color);
    }

    cv::Mat kltCanvas;
    cv::Mat canvas;
    cv::hconcat(kltLeftCanvas, kltRightCanvas, kltCanvas);
    cv::hconcat(kltCanvas, kltTrackCanvas, kltCanvas);
    cv::hconcat( leftCanvas, rightCanvas, canvas);
    cv::hconcat( canvas, trackCanvas, canvas);
    cv::vconcat(canvas, kltCanvas, canvas);
    imshow("canvas", canvas);
    std::string message = "Lucyte keypoints " + std::to_string(validLucyteKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT);
    message += ", Lucity average displace in pixels : " + std::to_string(lucyteError);
    message += ", KLT keypoints " + std::to_string(validKLTKeyPoints) + " of " + std::to_string(KEYPOINTCOUNT);
    message += ", KLT average displace in pixels : " + std::to_string(kltError);
    cv::setWindowTitle("canvas",  message);
    return canvas;
}

int main(int argc, char** argv)
{
    defaultDescriptorShape;
    srand(SEED);

    cv::VideoWriter video;
    if (outputVideo) video = cv::VideoWriter(outputBenchmarkVideoFileName, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(512 * 3, 512 * 2), true);

    cv::Mat benchImage = cv::imread(benchmarkPictureFileName, cv::IMREAD_COLOR);
    while (cv::waitKey(100) != 27) {
        cv::Mat mat = testFrame(benchImage);
        if (outputVideo) for (int i = 0; i < 5; i++) video.write(mat);
        cv::waitKey(1);
    }

    if (outputVideo) video.release();

    return 0;
}
