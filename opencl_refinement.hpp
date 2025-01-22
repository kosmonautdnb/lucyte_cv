// Created on: 21.01.2025 by Stefan Mader
#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "refinement.hpp"

void initOpenCL();
void uploadMipMaps_openCL(const std::vector<cv::Mat>& mipMaps);
void uploadKeyPoints_openCL(const std::vector<KeyPoint> &keyPoints);
void uploadDescriptorShape_openCL();
void uploadDescriptors_openCL(const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openCL(const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale);
void refineKeyPoints_openCL(std::vector<KeyPoint>& destKeyPoints, std::vector<KeyPoint>& destVariancePoints, const int mipEnd, const int STEPCOUNT, const int stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE);

