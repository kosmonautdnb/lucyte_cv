// Lucyte Created on: 24.01.2025 by Stefan Mader
#include "descriptors.hpp"
#include <math.h>

float descriptorsX1[DESCRIPTORSIZE] = { 0 };
float descriptorsY1[DESCRIPTORSIZE] = { 0 };
float descriptorsX2[DESCRIPTORSIZE] = { 0 };
float descriptorsY2[DESCRIPTORSIZE] = { 0 };

static const float randomLike(const int index) {
    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);
    b = b ^ (b << 8) ^ (b * 23);
    b >>= 3;
    return (float)(b & 0xffff) / 0x10000;
}

static void cross(const int index, const float centerx, const float centery, const float size) {
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

static void cross2(const int index, const float centerx, const float centery, const float size) {
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
    cross(i, 0 - rad * 0.3f, 0 - rad * 0.3f, rad * 0.6f);
    i++;
    cross(i, 0 + rad * 0.3f, 0 + rad * 0.3f, rad * 0.6f);
    i++;
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            cross(i, -rad * 0.5f + rad * x, -rad * 0.5f + rad * y, rad * 0.5f);
            i++;
        }
    }
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            cross(i, -rad * 0.75f + rad * 0.5f * x, -rad * 0.75f + rad * 0.5f * y, rad * 0.25f);
            i++;
        }
    }
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            cross(i, -rad * 0.25f + rad * 0.25f * x, -rad * 0.25f + rad * 0.25f * y, rad * 0.3f);
            i++;
        }
    }
    // 6 + 8 + 32 + 18
}

void defaultDescriptorShape38(const float rad) {
    int i = 0;
    cross2(i, 0, 0, rad);
    i++;
    cross2(i, 0 - rad * 0.3f, 0 - rad * 0.3f, rad * 0.6f);
    i++;
    cross2(i, 0 + rad * 0.3f, 0 + rad * 0.3f, rad * 0.6f);
    i++;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            cross(i, -rad * 0.75f + rad * 0.5f * x, -rad * 0.75f + rad * 0.5f * y, rad * 0.25f);
            i++;
        }
    }
    // 6 + 32
}

void defaultDescriptorShapeSpiral(const float rad, int descriptorSize) {
    for (int i = 0; i < descriptorSize; i += 2) {
        const float ri = (float)randomLike(i*13 + 1234);
        const float a1 = ri * 2 * 3.14159f * 2.5f + 3.14159f*0.5f;
        const float a2 = ri * 2 * 3.14159f * 4.75f;
        const float rad1 = ri * 0.5f * rad;
        const float rad2 = (1-ri*0.75f) * rad;
        descriptorsX1[i] = sinf(a1) * rad1;
        descriptorsY1[i] = cosf(a1) * rad1;
        descriptorsX2[i] = sinf(a2) * rad2;
        descriptorsY2[i] = cosf(a2) * rad2;
        descriptorsX1[i+1] = cosf(a1) * rad1;
        descriptorsY1[i+1] = -sinf(a1) * rad1;
        descriptorsX2[i+1] = cosf(a2) * rad2;
        descriptorsY2[i+1] = -sinf(a2) * rad2;
    }
}

void defaultDescriptorShapeCrosses(const float rad, int descriptorSize) {
    for (int i = 0; i < descriptorSize/2; i++) {
        float xp = (randomLike(i * 113 + 1212)*2.f-1.f)*rad;
        float yp = (randomLike(i * 237 + 212)*2.f-1.f)*rad;
        float r1 = rad - fabsf(xp);
        float r2 = rad - fabsf(yp);
        cross(i, xp, yp, r1 < r2 ? r1 : r2);
    }
}
