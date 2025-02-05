// Lucyte Created on: 03.02.2025 by Stefan Mader
#pragma once
#include "refinement.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>

extern GLFWwindow* window;
extern int window_width;
extern int window_height;

void initOpenGL();
void uploadDescriptorShape_openGL();
void uploadMipMaps_openGL(const int mipmapsId, const std::vector<cv::Mat>& mipMaps);
void uploadKeyPoints_openGL(const int keyPointsId, const std::vector<KeyPoint>& keyPoints);
void uploadDescriptors_openGL(const int descriptorsId, const int mipEnd, const std::vector<std::vector<Descriptor>>& sourceMips);
void sampleDescriptors_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipEnd, std::vector<std::vector<Descriptor>>& destMips, const float DESCRIPTORSCALE, const float MIPSCALE);
void refineKeyPoints_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int keyPointCount, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors);
