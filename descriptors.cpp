// Lucyte Created on: 24.01.2025 by Stefan Mader
#include "descriptors.hpp"
#include <math.h>

float descriptorsX1[MAXDESCRIPTORSIZE];
float descriptorsY1[MAXDESCRIPTORSIZE];
float descriptorsX2[MAXDESCRIPTORSIZE];
float descriptorsY2[MAXDESCRIPTORSIZE];

static const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / 0x10000;
}

static void cross(const int index, const double centerx, const double centery, const double size) {
    float a = randomLike(index) * 2.f * 3.14159f;
    descriptorsX1[index * 2 + 0] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 0] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 0] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 0] = centery - sinf(a) * size;
    a += 3.14159f * 0.5 * 0.75;
    descriptorsX1[index * 2 + 1] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 1] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 1] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 1] = centery - sinf(a) * size;
}

static void cross2(const int index, const double centerx, const double centery, const double size) {
    float a = 0;
    descriptorsX1[index * 2 + 0] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 0] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 0] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 0] = centery - sinf(a) * size;
    a += 3.14159f * 0.5;
    descriptorsX1[index * 2 + 1] = centerx + cosf(a) * size * 1.5f;
    descriptorsY1[index * 2 + 1] = centery + sinf(a) * size * 1.5f;
    descriptorsX2[index * 2 + 1] = centerx - cosf(a) * size;
    descriptorsY2[index * 2 + 1] = centery - sinf(a) * size;
}

void defaultDescriptorShape64(const float rad) {
    int i = 0;
    cross(i, 0, 0, rad);
    i++;
    cross(i, 0 - rad * 0.3, 0 - rad * 0.3, rad * 0.6);
    i++;
    cross(i, 0 + rad * 0.3, 0 + rad * 0.3, rad * 0.6);
    i++;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            cross(i, -rad * 0.5 + rad * x, -rad * 0.5 + rad * y, rad * 0.5);
            i++;
        }
    }
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            cross(i, -rad * 0.75 + rad * 0.5 * x, -rad * 0.75 + rad * 0.5 * y, rad * 0.25);
            i++;
        }
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            cross(i, -rad * 0.25 + rad * 0.25 * x, -rad * 0.25 + rad * 0.25 * y, rad * 0.3);
            i++;
        }
    }
    // 6 + 8 + 32 + 18
}

void defaultDescriptorShape38(const float rad) {
    int i = 0;
    cross2(i, 0, 0, rad);
    i++;
    cross2(i, 0 - rad * 0.3, 0 - rad * 0.3, rad * 0.6);
    i++;
    cross2(i, 0 + rad * 0.3, 0 + rad * 0.3, rad * 0.6);
    i++;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            cross(i, -rad * 0.75 + rad * 0.5 * x, -rad * 0.75 + rad * 0.5 * y, rad * 0.25);
            i++;
        }
    }
    // 6 + 32
}

void defaultDescriptorShapeSpiral(const float rad, int descriptorSize) {
    for (int i = 0; i < descriptorSize; i += 2) {
        const float ri = (float)randomLike(i*13 + 1234);
        const float a1 = ri * 2 * 3.14159f * 2.5 + 3.14159*0.5;
        const float a2 = ri * 2 * 3.14159f * 4.75;
        const float rad1 = ri * 0.5 * rad;
        const float rad2 = (1-ri*0.75) * rad;
        descriptorsX1[i] = sin(a1) * rad1;
        descriptorsY1[i] = cos(a1) * rad1;
        descriptorsX2[i] = sin(a2) * rad2;
        descriptorsY2[i] = cos(a2) * rad2;
        descriptorsX1[i+1] = cos(a1) * rad1;
        descriptorsY1[i+1] = -sin(a1) * rad1;
        descriptorsX2[i+1] = cos(a2) * rad2;
        descriptorsY2[i+1] = -sin(a2) * rad2;
    }
}
