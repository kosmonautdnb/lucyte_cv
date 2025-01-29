// Lucyte Created on: 21.01.2025 by Stefan Mader
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "refinement.hpp"
#include "opencl_refinement.hpp"

void initOpenCL_nonblocking();
void uploadMipMaps_openCL(const int bufferId, const std::vector<cv::Mat>& mipMaps);
void uploadMipMaps_openCL_waitfor(const int bufferId);
void uploadKeyPoints_openCL(const int bufferId, const std::vector<KeyPoint> &keyPoints);
void uploadDescriptorShape_openCL(const int bufferId);
void uploadDescriptors_openCL(const int bufferId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openCL(const int bufferId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale);
void refineKeyPoints_openCL(const int bufferId, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);

