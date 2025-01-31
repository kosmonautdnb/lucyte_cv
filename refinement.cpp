// Lucyte Created on: 17.01.2025 by Stefan Mader
#include "refinement.hpp"
#include <math.h>
#include <omp.h>

int texf(const unsigned char* s, const int x, const int y, const int width, const int height) {
    int xr = x >> 10;
    int yr = y >> 10;
    if ((unsigned int)xr >= (width - 1) || (unsigned int)yr >= (height - 1)) {
        if (xr < 0) xr = 0;
        if (yr < 0) yr = 0;
        if (xr >= (width - 1)) xr = width - 2;
        if (yr >= (height - 1)) yr = height - 2;
    }
    s += xr + yr * width;
    const unsigned short a = *((unsigned short*)(s));
    const unsigned short b = *((unsigned short*)(s + width));
    const int a0 = a & 255;
    const int a1 = a >> 8;
    const int b0 = b & 255;
    const int b1 = b >> 8;
    const int xc1 = (((a1 - a0) * (x & 1023)) >> 9) + (a0 << 1);
    const int xc2 = (((b1 - b0) * (x & 1023)) >> 9) + (b0 << 1);
    return (((xc2 - xc1) * (y & 1023)) >> 9) + (xc1 << 1); // 10 bit
}

int tex(const unsigned char* s, const float x, const float y, const int width, const int height) {
    return texf(s, int(floorf(x * 1024.f)), int(floorf(y * 1024.f)),width,height);
}

float sampleDescriptor(const KeyPoint& kp, Descriptor& d, const unsigned char* s, const float descriptorScale, const int width, const int height, const float mipScale) {
    KeyPoint kp2 = kp;
    kp2.x *= mipScale;
    kp2.y *= mipScale;
    const float descriptorSize = mipScale * descriptorScale * 1024.f;
    d.clear();
    int h = 0;
    int kp2x = int(floorf(kp2.x * 1024.f));
    int kp2y = int(floorf(kp2.y * 1024.f));
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        const int l1 = texf(s, kp2x + int(descriptorSize * descriptorsX1[b]), kp2y + int(descriptorSize * descriptorsY1[b]), width, height);
        const int l2 = texf(s, kp2x + int(descriptorSize * descriptorsX2[b]), kp2y + int(descriptorSize * descriptorsY2[b]), width, height);
        h += abs(l2 - l1);
        d.bits[b / 32] |= (l2 < l1 ? 1 : 0) << (b & 31);
    }
    return float(h) / float(DESCRIPTORSIZE * 1024);
}

KeyPoint refineKeyPoint(const bool stepping, const KeyPoint& kp, const Descriptor& toSearch, const unsigned char* s, const float descriptorScale, const float angle, const float step, const int width, const int height, const float mipScale) {
    const int kp2x = floorf(kp.x * mipScale * 1024.f);
    const int kp2y = floorf(kp.y * mipScale * 1024.f);
    const float descriptorSize = mipScale * descriptorScale;
    const float sina = sinf(angle);
    const float cosa = cosf(angle);
    const float sinad = (sina * descriptorSize * 1024.f);
    const float cosad = (cosa * descriptorSize * 1024.f);
    const int gdxx = int(cosa * 1024.f); const int gdxy = int(sina * 1024.f);
    const int gdyx = int(-sina * 1024.f); const int gdyy = int(cosa * 1024.f);
    float xa = 0;
    float ya = 0;
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        const int d1x = kp2x + int(cosad * descriptorsX1[b] + sinad * descriptorsY1[b]);
        const int d1y = kp2y + int(-sinad * descriptorsX1[b] + cosad * descriptorsY1[b]);
        const int d2x = kp2x + int(cosad * descriptorsX2[b] + sinad * descriptorsY2[b]);
        const int d2y = kp2y + int(-sinad * descriptorsX2[b] + cosad * descriptorsY2[b]);
        const int l1 = texf(s, d1x, d1y, width, height);
        const int l2 = texf(s, d2x, d2y, width, height);
        const int b1 = (toSearch.bits[b>>5] >> (b & 31)) & 1;
        const int b2 = (l2 < l1 ? 1 : 0);
        if (b2 != b1) {
            // get x and y gradient at both points
            int gx = texf(s, d2x - gdxx, d2y - gdxy, width, height) - texf(s, d2x + gdxx, d2y + gdxy, width, height);
            int gy = texf(s, d2x - gdyx, d2y - gdyy, width, height) - texf(s, d2x + gdyx, d2y + gdyy, width, height);
            gx -= texf(s, d1x - gdxx, d1y - gdxy, width, height) - texf(s, d1x + gdxx, d1y + gdxy, width, height);
            gy -= texf(s, d1x - gdyx, d1y - gdyy, width, height) - texf(s, d1x + gdyx, d1y + gdyy, width, height);
            // go in correct direction by the gradients for the bits to flip from (b2 != b1) to (b2 == b1)
            if (b2 != 0) {
                gx = -gx;
                gy = -gy;
            }
            if (stepping) {
                gx = (gx == 0) ? 0 : (gx > 0) ? 1 : -1;
                gy = (gy == 0) ? 0 : (gy > 0) ? 1 : -1;
            }
            const int dx = (d2x - d1x);
            const int dy = (d2y - d1y);
            const float l = sqrtf((float)(dx*dx+dy*dy)/(1024.f*1024.f));
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
    const float stepSize = descriptorScale * step;
    KeyPoint r;
    r.x = kp.x + (cosa * xa + sina * ya) * stepSize;
    r.y = kp.y + (-sina * xa + cosa * ya) * stepSize;
    const bool clipping = false; if (clipping) {
        const int w = int(floorf(float(width) / mipScale));
        const int h = int(floorf(float(height) / mipScale));
        if (r.x < 0) r.x = 0;
        if (r.y < 0) r.y = 0;
        if (r.x >= w - 1) r.x = w - 1;
        if (r.y >= h - 1) r.y = h - 1;
    }
    return r;
}

