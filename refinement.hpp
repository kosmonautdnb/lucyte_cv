// Created on: 17.01.2025 by Stefan Mader
#pragma once
#include <stdint.h>
#include <string.h>

const int DESCRIPTORSIZE = 32 * 2;
extern float descriptorsX1[DESCRIPTORSIZE];
extern float descriptorsY1[DESCRIPTORSIZE];
extern float descriptorsX2[DESCRIPTORSIZE];
extern float descriptorsY2[DESCRIPTORSIZE];

class KeyPoint {
public:
    float x, y;
};

class Descriptor
{
public:
    static const int uint32count = (DESCRIPTORSIZE + 31) / 32;
    uint32_t bits[uint32count];
    uint32_t valid[uint32count];
    void clear()
    {
        memset(bits, 0, sizeof(bits));
        memset(valid, 0, sizeof(valid));
    }
};

int tex(const unsigned char* s, const float x, const float y, int width, int height);
void sampleDescriptor(KeyPoint& kp, Descriptor& d, const unsigned char* s, float descriptorScale, int width, int height);
void refineKeyPoint(bool stepping, KeyPoint& kp, const Descriptor& toSearch, Descriptor& current, const unsigned char* s, float descriptorScale, float angle, float step, int width, int height);
