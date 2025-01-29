// Lucyte Created on: 24.01.2025 by Stefan Mader
#pragma once
#include "constants.hpp"

extern float descriptorsX1[DESCRIPTORSIZE];
extern float descriptorsY1[DESCRIPTORSIZE];
extern float descriptorsX2[DESCRIPTORSIZE];
extern float descriptorsY2[DESCRIPTORSIZE];

void defaultDescriptorShape38(const float rad);
void defaultDescriptorShape64(const float rad);
void defaultDescriptorShapeSpiral(const float rad, const int descriptorSize);
void defaultDescriptorShapeCrosses(const float rad, const int descriptorSize);
