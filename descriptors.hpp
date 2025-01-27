// Lucyte Created on: 24.01.2025 by Stefan Mader
#pragma once

const int MAXDESCRIPTORSIZE = 1024;
extern float descriptorsX1[MAXDESCRIPTORSIZE];
extern float descriptorsY1[MAXDESCRIPTORSIZE];
extern float descriptorsX2[MAXDESCRIPTORSIZE];
extern float descriptorsY2[MAXDESCRIPTORSIZE];

void defaultDescriptorShape38(const float rad);
void defaultDescriptorShape64(const float rad);
void defaultDescriptorShapeSpiral(const float rad, const int descriptorSize);
