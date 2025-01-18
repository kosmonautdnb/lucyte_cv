// Created on: 17.01.2025 by Stefan Mader
#include "refinement.hpp"
#include <math.h>
#include <omp.h>

float descriptorsX1[DESCRIPTORSIZE];
float descriptorsY1[DESCRIPTORSIZE];
float descriptorsX2[DESCRIPTORSIZE];
float descriptorsY2[DESCRIPTORSIZE];
const bool ONLYVALID = false;

int tex(const unsigned char* s, const float x, const float y, int width, int height) {
    const int xr8 = int(floor(x * 1024.f));
    const int yr8 = int(floor(y * 1024.f));
    const int xr = xr8 >> 10;
    const int yr = yr8 >> 10;
    if ((unsigned int)xr >= (width - 1) || (unsigned int)yr >= (height - 1))
        return -abs(xr) - abs(yr); // invalid but put some value in maybe it helps
    s += xr + yr * width;
    const int xn = xr8 & 1023;
    const int yn = yr8 & 1023;
    const unsigned short a = *((unsigned short*)(s));
    const unsigned short b = *((unsigned short*)(s + width));
    const unsigned char a0 = *((unsigned char*)&a);
    const unsigned char a1 = *((unsigned char*)&a + 1);
    const unsigned char b0 = *((unsigned char*)&b);
    const unsigned char b1 = *((unsigned char*)&b + 1);
    const int xc1 = (((a1 - a0) * xn) >> 9) + (a0 << 1);
    const int xc2 = (((b1 - b0) * xn) >> 9) + (b0 << 1);
    return (((xc2 - xc1) * yn) >> 9) + (xc1 << 1); // 10 bit
}

void sampleDescriptor(KeyPoint& kp, Descriptor& d, const unsigned char* s, float descriptorScale, int width, int height) {
    d.clear();
    uint32_t l = 0;
    uint32_t v = 0;
#pragma omp parallel for
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        const int l1 = tex(s, kp.x + (descriptorScale * descriptorsX1[b]), kp.y + (descriptorScale * descriptorsY1[b]), width, height);
        const int l2 = tex(s, kp.x + (descriptorScale * descriptorsX2[b]), kp.y + (descriptorScale * descriptorsY2[b]), width, height);
        l |= (l2 < l1 ? 1 : 0) << (b & 31);
        v |= ((l1 < 0 || l2 < 0) ? 0 : 1) << (b & 31);
        if ((b & 31) == 31) {
            d.bits[b / 32] = l;
            d.valid[b / 32] = v;
            l = 0;
            v = 0;
        }
    }
}

KeyPoint refineKeyPoint(bool stepping, const KeyPoint& kp, const Descriptor& toSearch, const Descriptor& current, const unsigned char* s, float descriptorScale, float angle, float step, int width, int height) {
    const float sina = sin(angle) * descriptorScale;
    const float cosa = cos(angle) * descriptorScale;
    const float gdxx = cosa; const float gdxy = sina;
    const float gdyx = -sina; const float gdyy = cosa;
    double xa = 0;
    double ya = 0;
#pragma omp parallel for
    for (int b = 0; b < DESCRIPTORSIZE; ++b) {
        int v1 = (toSearch.valid[b / 32] >> (b & 31)) & 1;
        int v2 = (current.valid[b / 32] >> (b & 31)) & 1;
        if (ONLYVALID && (v1 != 1 || v2 != 1))
            continue;
        int b1 = (toSearch.bits[b / 32] >> (b & 31)) & 1;
        int b2 = (current.bits[b / 32] >> (b & 31)) & 1;
        if (b2 != b1) {
            const float d1x = cosa * descriptorsX1[b] + sina * descriptorsY1[b];
            const float d1y = -sina * descriptorsX1[b] + cosa * descriptorsY1[b];
            const float d2x = cosa * descriptorsX2[b] + sina * descriptorsY2[b];
            const float d2y = -sina * descriptorsX2[b] + cosa * descriptorsY2[b];
            // go in direction of the gradient to get smaller values
            int g1x = tex(s, kp.x + d1x - gdxx, kp.y + d1y - gdxy, width, height) - tex(s, kp.x + d1x + gdxx, kp.y + d1y + gdxy, width, height);
            int g1y = tex(s, kp.x + d1x - gdyx, kp.y + d1y - gdyy, width, height) - tex(s, kp.x + d1x + gdyx, kp.y + d1y + gdyy, width, height);
            // go in direction of the gradient to get smaller values
            int g2x = tex(s, kp.x + d2x - gdxx, kp.y + d2y - gdxy, width, height) - tex(s, kp.x + d2x + gdxx, kp.y + d2y + gdxy, width, height);
            int g2y = tex(s, kp.x + d2x - gdyx, kp.y + d2y - gdyy, width, height) - tex(s, kp.x + d2x + gdyx, kp.y + d2y + gdyy, width, height);
            //g2 should be bigger g1 for bit to be set
            // ideally g2 should be bigger and g1 should be smaller
            int gx = g2x - g1x;
            int gy = g2y - g1y;
            if (b2 != 0) {
                gx *= -1;
                gy *= -1;
            }
            if (stepping) {
                gx = (gx == 0) ? 0 : (gx > 0) ? 1 : -1;
                gy = (gy == 0) ? 0 : (gy > 0) ? 1 : -1;
            }
            double l = sqrt(double((d2x - d1x) * (d2x - d1x) + (d2y - d1y) * (d2y - d1y)));
            xa += double(gx) * l;
            ya += double(gy) * l;
        }
    }
    double l = xa * xa + ya * ya;
    if (l > 0.0) {
        l = sqrt(l);
        xa /= l;
        ya /= l;
    }
    KeyPoint r;
    r.x = kp.x + (cosa * xa + sina * ya) * step;
    r.y = kp.y + (-sina * xa + cosa * ya) * step;
    return r;
}

