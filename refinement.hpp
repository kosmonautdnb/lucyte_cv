// Lucyte Created on: 17.01.2025 by Stefan Mader
#pragma once
#include <stdint.h>
#include <string.h>
#include "descriptors.hpp"

class KeyPoint {
public:
    float x, y;
};

class Descriptor
{
public:
    static const int uint32count = (MAXDESCRIPTORSIZE + 31) / 32;
    uint32_t bits[uint32count];
    uint32_t valid[uint32count];
    void clear()
    {
        memset(bits, 0, sizeof(bits));
        memset(valid, 0, sizeof(valid));
    }
};

int tex(const unsigned char* s, const float x, const float y, const int width, const int height);
float sampleDescriptor(const KeyPoint& kp, Descriptor& d, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale);
KeyPoint refineKeyPoint(const bool stepping, const KeyPoint& kp, const Descriptor& toSearch, const Descriptor& current, const unsigned char* s, const float descriptorScale, const float angle, const float step, const int width, const int height, const float mipScale);
