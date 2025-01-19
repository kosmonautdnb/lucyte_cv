// Created on: 17.01.2025 by Stefan Mader
#include "refinement.hpp"
#include <math.h>
#include <omp.h>

float descriptorsX1[DESCRIPTORSIZE];
float descriptorsY1[DESCRIPTORSIZE];
float descriptorsX2[DESCRIPTORSIZE];
float descriptorsY2[DESCRIPTORSIZE];
const bool ONLYVALID = false;

int tex(const unsigned char* s, const float x, const float y, const int width, const int height) {
    const int xr8 = int(floorf(x * 1024.f));
    const int yr8 = int(floorf(y * 1024.f));
    const int xr = xr8 >> 10;
    const int yr = yr8 >> 10;
    if ((unsigned int)xr >= (width - 1) || (unsigned int)yr >= (height - 1))
        return -abs(xr) - abs(yr); // invalid but put some value in so that the compares always have the same result here
    s += xr + yr * width;
    const int xn = xr8 & 1023;
    const int yn = yr8 & 1023;
    const unsigned short a = *((unsigned short*)(s));
    const unsigned short b = *((unsigned short*)(s + width));
    const unsigned char a0 = a & 255;
    const unsigned char a1 = a >> 8;
    const unsigned char b0 = b & 255;
    const unsigned char b1 = b >> 8;
    const int xc1 = (((a1 - a0) * xn) >> 9) + (a0 << 1);
    const int xc2 = (((b1 - b0) * xn) >> 9) + (b0 << 1);
    return (((xc2 - xc1) * yn) >> 9) + (xc1 << 1); // 10 bit
}

void sampleDescriptor(const KeyPoint& kp, Descriptor& d, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale) {
    KeyPoint kp2 = kp;
    kp2.x *= mipScale;
    kp2.y *= mipScale;
    const float descriptorSize = mipScale * descriptorScale;
    d.clear();
#pragma omp parallel for num_threads(32)
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        const int l1 = tex(s, kp2.x + (descriptorSize * descriptorsX1[b]), kp2.y + (descriptorSize * descriptorsY1[b]), width, height);
        const int l2 = tex(s, kp2.x + (descriptorSize * descriptorsX2[b]), kp2.y + (descriptorSize * descriptorsY2[b]), width, height);
        d.bits[b / 32]  |= (l2 < l1 ? 1 : 0) << (b & 31);
        d.valid[b / 32] |= ((l1 < 0 || l2 < 0) ? 0 : 1) << (b & 31);
    }
}

KeyPoint refineKeyPoint(const bool stepping, const KeyPoint& kp, const Descriptor& toSearch, const Descriptor& current, const unsigned char* s, const float descriptorScale, const float angle, const float step, const int width, const int height, const const float mipScale) {
    KeyPoint kp2 = kp;
    kp2.x *= mipScale;
    kp2.y *= mipScale;
    const float descriptorSize = mipScale * descriptorScale;
    const float sina = sinf(angle);
    const float cosa = cosf(angle);
    const float sinad = sina * descriptorSize;
    const float cosad = cosa * descriptorSize;
    const float gdxx = cosa; const float gdxy = sina;
    const float gdyx = -sina; const float gdyy = cosa;
    float xa = 0;
    float ya = 0;
#pragma omp parallel for num_threads(32)
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        if (ONLYVALID) {
            int v1 = (toSearch.valid[b / 32] >> (b & 31)) & 1;
            int v2 = (current.valid[b / 32] >> (b & 31)) & 1;
            if ((v1 != 1 || v2 != 1)) {
                continue;
            }
        }
        int b1 = (toSearch.bits[b / 32] >> (b & 31)) & 1;
        int b2 = (current.bits[b / 32] >> (b & 31)) & 1;
        if (b2 != b1) {
            const float d1x = (cosad * descriptorsX1[b] + sinad * descriptorsY1[b]);
            const float d1y = (-sinad * descriptorsX1[b] + cosad * descriptorsY1[b]);
            const float d2x = (cosad * descriptorsX2[b] + sinad * descriptorsY2[b]);
            const float d2y = (-sinad * descriptorsX2[b] + cosad * descriptorsY2[b]);
            // go in direction of the gradient to get smaller values
            const int g1x = tex(s, kp2.x + d1x - gdxx, kp2.y + d1y - gdxy, width, height) - tex(s, kp2.x + d1x + gdxx, kp2.y + d1y + gdxy, width, height);
            const int g1y = tex(s, kp2.x + d1x - gdyx, kp2.y + d1y - gdyy, width, height) - tex(s, kp2.x + d1x + gdyx, kp2.y + d1y + gdyy, width, height);
            // go in direction of the gradient to get smaller values
            const int g2x = tex(s, kp2.x + d2x - gdxx, kp2.y + d2y - gdxy, width, height) - tex(s, kp2.x + d2x + gdxx, kp2.y + d2y + gdxy, width, height);
            const int g2y = tex(s, kp2.x + d2x - gdyx, kp2.y + d2y - gdyy, width, height) - tex(s, kp2.x + d2x + gdyx, kp2.y + d2y + gdyy, width, height);
            //g2 should be bigger g1 for bit to be set
            // ideally g2 should be bigger and g1 should be smaller
            int gx = g2x - g1x;
            int gy = g2y - g1y;
            if (b2 != 0) {
                gx = -gx;
                gy = -gy;
            }
            if (stepping) {
                gx = (gx == 0) ? 0 : (gx > 0) ? 1 : -1;
                gy = (gy == 0) ? 0 : (gy > 0) ? 1 : -1;
            }
            const float dx = d2x - d1x;
            const float dy = d2y - d1y;
            const float l = sqrtf(dx*dx+dy*dy);
            xa += float(gx) * l;
            ya += float(gy) * l;
        }
    }
    float l = xa * xa + ya * ya;
    if (l > 0.f) {
        l = sqrtf(l);
        xa /= l;
        ya /= l;
    }
    float stepSize = descriptorScale * step;
    KeyPoint r;
    r.x = kp.x + (cosa * xa + sina * ya) * stepSize;
    r.y = kp.y + (-sina * xa + cosa * ya) * stepSize;
    const bool clipping = false;
    if (clipping) {
        int w = floorf(width / mipScale);
        int h = floorf(height / mipScale);
        if (r.x < 0) r.x = 0;
        if (r.y < 0) r.y = 0;
        if (r.x >= w - 1) r.x = w - 1;
        if (r.y >= h - 1) r.y = h - 1;
    }
    return r;
}

