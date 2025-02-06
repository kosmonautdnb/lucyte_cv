// Lucyte Created on: 03.02.2025 by Stefan Mader
#include "constants.hpp"
#include "opengl_refinement.hpp"
#include <string>

#if __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__
#include <GL/glew.h>
#include <GLFW/glfw3.h>
GLFWwindow* window = NULL;
int window_width;
int window_height;
void glNewFrame() {
    glfwSwapBuffers(window);
    glfwPollEvents();
    glfwGetFramebufferSize(window, &window_width, &window_height);
    glViewport(0, 0, window_width, window_height);
    glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}
bool glIsCancelled() {
    return glfwWindowShouldClose(window);
}
#endif // __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__

#if __CURRENT_PLATFORM__ == __PLATFORM_ANDROID__
#include <EGL/egl.h>
#include <GLES/gl.h>
#include <GLES3/gl3.h>
#endif // __CURRENT_PLATFORM__ == __PLATFORM_ANDROID__

static const int MAXMIPMAPS = 32;
static const int MIPMAPIDCOUNT = 8;
static const int KEYPOINTIDCOUNT = 2;
static const int DESCRIPTORIDCOUNT = 2;
static const int maxTextureSize = 1024;

static GLuint openGLDescriptorShape[4] = { 0 };
static int openGLDescriptorShapeTextureWidth = 0;
static int openGLDescriptorShapeTextureHeight = 0;
static GLuint openGLMipMaps[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLMipMapsTextureWidth[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLMipMapsTextureHeight[MIPMAPIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLMipMapCount[MIPMAPIDCOUNT] = { 0 };
static GLuint openGLKeyPointsX[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLKeyPointsY[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureWidth[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureHeight[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointCount[KEYPOINTIDCOUNT] = { 0 };
static std::pair<GLuint, GLuint> openGLDescriptorRenderTarget[KEYPOINTIDCOUNT][MIPMAPIDCOUNT] = { {{0,0}} };
static GLuint openGLDescriptorRenderTargetWidth[KEYPOINTIDCOUNT][MIPMAPIDCOUNT] = { {0} };
static GLuint openGLDescriptorRenderTargetHeight[KEYPOINTIDCOUNT][MIPMAPIDCOUNT] = { {0} };
static std::pair<GLuint, GLuint> openGLKeyPointsRenderTarget[KEYPOINTIDCOUNT] = { {0,0} };
static GLuint openGLKeyPointsRenderTargetWidth[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLKeyPointsRenderTargetHeight[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLDescriptors[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLDescriptorsTextureWidth[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static int openGLDescriptorsTextureHeight[DESCRIPTORIDCOUNT][MAXMIPMAPS] = { {0} };
static GLuint openGLFragmentShader_sampleDescriptors = 0;
static GLuint openGLFragmentShader_refineKeyPoints = 0;
static GLuint openGLFragmentShader_displayMipMap = 0;
static GLuint openGLVertexShader_basic = 0;
static GLuint openGLProgram_sampleDescriptors = 0;
static GLuint openGLProgram_refineKeyPoints = 0;
static GLuint openGLProgram_displayMipMap = 0;
#define SHARED_SHADER_FUNCTIONS ""\
"vec2 t2d_nearest(sampler2D t1, sampler2D t2, int x, int y, int width, int height)\n"\
"{\n"\
"   vec2 k = vec2((float(x)+0.25)/float(width),(float(y)+0.25)/float(height));\n"\
"   return vec2(textureLod(t1,k,0.0).x,textureLod(t2,k,0.0).x);\n"\
"}\n"\
"vec2 descriptors1(int i) {\n"\
"   return t2d_nearest(descriptor1x, descriptor1y, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"}\n"\
"vec2 descriptors2(int i) {\n"\
"   return t2d_nearest(descriptor2x, descriptor2y, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"}\n"

const std::string sampleDescriptorProgram = ""
"in vec2 p;\n"
"out uvec4 frag_color;\n"
"uniform int texWidthDescriptorShape;\n"
"uniform int texHeightDescriptorShape;\n"
"uniform int texWidthMipMap;\n"
"uniform int texHeightMipMap;\n"
"uniform int texWidthKeyPoints;\n"
"uniform int texHeightKeyPoints;\n"
"uniform int DESCRIPTORSIZE;\n"
"uniform float descriptorScale;\n"
"uniform float mipScale;\n"
"uniform sampler2D descriptor1x;\n"
"uniform sampler2D descriptor1y;\n"
"uniform sampler2D descriptor2x;\n"
"uniform sampler2D descriptor2y;\n"
"uniform sampler2D mipMap;\n"
"uniform sampler2D keyPointsx;\n"
"uniform sampler2D keyPointsy;\n"
SHARED_SHADER_FUNCTIONS
"void main()\n"
"{\n"
"       int txx = int(floor(p.x+0.25));\n"
"       int txy = int(floor(p.y+0.25));\n"
"       float descriptorSize = descriptorScale * mipScale;\n"
"       vec2 kp = t2d_nearest(keyPointsx,keyPointsy,txx,txy,texWidthKeyPoints,texHeightKeyPoints);\n"
"       kp *= mipScale;\n"
"       vec2 sc = vec2(1.0/float(texWidthMipMap),1.0/float(texHeightMipMap));\n"
"       unsigned int b[4]; b[0]=unsigned int(0); b[1]=unsigned int(0); b[2]=unsigned int(0); b[3]=unsigned int(0);\n"
"       for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
"       {\n"
"           float l1 = textureLod(mipMap, (kp + vec2(descriptors1x[i],descriptors1y[i]) * descriptorSize) * sc, 0.0).x;\n"
"           float l2 = textureLod(mipMap, (kp + vec2(descriptors2x[i],descriptors2y[i]) * descriptorSize) * sc, 0.0).x;\n"
"           b[i>>5] += (l2 < l1) ? unsigned int(1)<<unsigne int(i & 31) : unsigned int(0);\n"
"       }\n"
"       frag_color = uvec4(b[0],b[1],b[2],b[3]);\n"
"}\n"
"\n";
const std::string refineKeyPointsProgram = ""\
"in vec2 p;\n"
"out vec4 frag_color;\n"
"uniform sampler2D mipMaps_0;\n"
"uniform sampler2D mipMaps_1;\n"
"uniform sampler2D mipMaps_2;\n"
"uniform sampler2D mipMaps_3;\n"
"uniform sampler2D mipMaps_4;\n"
"uniform sampler2D mipMaps_5;\n"
"uniform sampler2D mipMaps_6;\n"
"uniform sampler2D mipMaps_7;\n"
"uniform sampler2D mipMaps_8;\n"
"uniform sampler2D mipMaps_9;\n"
"uniform sampler2D mipMaps_10;\n"
"uniform sampler2D mipMaps_11;\n"
"uniform sampler2D mipMaps_12;\n"
"uniform sampler2D mipMaps_13;\n"
"uniform sampler2D mipMaps_14;\n"
"uniform sampler2D mipMaps_15;\n"
"uniform sampler2D mipMaps_16;\n"
"uniform sampler2D mipMaps_17;\n"
"uniform sampler2D mipMaps_18;\n"
"uniform sampler2D mipMaps_19;\n"
"uniform sampler2D mipMaps_20;\n"
"uniform sampler2D mipMaps_21;\n"
"uniform sampler2D mipMaps_22;\n"
"uniform sampler2D mipMaps_23;\n"
"uniform sampler2D mipMaps_24;\n"
"uniform sampler2D mipMaps_25;\n"
"uniform sampler2D mipMaps_26;\n"
"uniform sampler2D mipMaps_27;\n"
"uniform sampler2D mipMaps_28;\n"
"uniform sampler2D mipMaps_29;\n"
"uniform sampler2D mipMaps_30;\n"
"uniform sampler2D mipMaps_31;\n"
"uniform usampler2D descriptors_0;\n"
"uniform usampler2D descriptors_1;\n"
"uniform usampler2D descriptors_2;\n"
"uniform usampler2D descriptors_3;\n"
"uniform usampler2D descriptors_4;\n"
"uniform usampler2D descriptors_5;\n"
"uniform usampler2D descriptors_6;\n"
"uniform usampler2D descriptors_7;\n"
"uniform usampler2D descriptors_8;\n"
"uniform usampler2D descriptors_9;\n"
"uniform usampler2D descriptors_10;\n"
"uniform usampler2D descriptors_11;\n"
"uniform usampler2D descriptors_12;\n"
"uniform usampler2D descriptors_13;\n"
"uniform usampler2D descriptors_14;\n"
"uniform usampler2D descriptors_15;\n"
"uniform usampler2D descriptors_16;\n"
"uniform usampler2D descriptors_17;\n"
"uniform usampler2D descriptors_18;\n"
"uniform usampler2D descriptors_19;\n"
"uniform usampler2D descriptors_20;\n"
"uniform usampler2D descriptors_21;\n"
"uniform usampler2D descriptors_22;\n"
"uniform usampler2D descriptors_23;\n"
"uniform usampler2D descriptors_24;\n"
"uniform usampler2D descriptors_25;\n"
"uniform usampler2D descriptors_26;\n"
"uniform usampler2D descriptors_27;\n"
"uniform usampler2D descriptors_28;\n"
"uniform usampler2D descriptors_29;\n"
"uniform usampler2D descriptors_30;\n"
"uniform usampler2D descriptors_31;\n"
"uniform int mipMapWidth[32];\n"
"uniform int mipMapHeight[32];\n"
"uniform int texWidthDescriptorShape;\n"
"uniform int texHeightDescriptorShape;\n"
"uniform int texWidthDescriptors;\n"
"uniform int texHeightDescriptors;\n"
"uniform int DESCRIPTORSIZE;\n"
"uniform int stepping;\n"
"uniform int mipEnd;\n"
"uniform int STEPCOUNT;\n"
"uniform float MIPSCALE;\n"
"uniform float STEPSIZE;\n"
"uniform float SCALEINVARIANCE;\n"
"uniform float ROTATIONINVARIANCE;\n"
"uniform sampler2D descriptor1x;\n"
"uniform sampler2D descriptor1y;\n"
"uniform sampler2D descriptor2x;\n"
"uniform sampler2D descriptor2y;\n"
"uniform int texWidthKeyPoints;\n"
"uniform int texHeightKeyPoints;\n"
"uniform sampler2D keyPointsx;\n"
"uniform sampler2D keyPointsy;\n"
"float mipScale;\n"
"float descriptorScale;\n"
"float angle;\n"
"float step;\n"
SHARED_SHADER_FUNCTIONS
"float randomLike(int index) {\n"
"    int b = index ^ (index * 11) ^ (index / 17) ^ (index >> 16) ^ (index * 1877) ^ (index * 8332) ^ (index * 173);\n"
"    b = b ^ (b << 8) ^ (b * 23);\n"
"    b >>= 3;\n"
"    return float(b & 0xffff) / float(0x10000);\n"
"}\n"
"vec2 refineKeyPoint(sampler2D s, vec2 kp, int mipMap, uvec4 descriptor) {\n"
"   vec2 sc = vec2(1.0/float(mipMapWidth[0]),1.0/float(mipMapHeight[0]));\n"
"   float sina = sin(angle);\n"
"   float cosa = cos(angle);\n"
"   vec2 cosid = vec2(cosa, sina) * descriptorScale;\n"
"   vec2 nsicd = vec2(-sina, cosa) * descriptorScale;\n"
"   vec2 gdx = vec2(cosa,sina) / mipScale * sc;\n"
"   vec2 gdy = vec2(-sina,cosa) / mipScale * sc;\n"
"   vec2 xya = vec2(0.0,0.0);\n"
"   unsigned int dBits[4];\n"
"   dBits[0] = descriptor.x;"
"   dBits[1] = descriptor.y;"
"   dBits[2] = descriptor.z;"
"   dBits[3] = descriptor.w;"
"   for (int b = 0; b < DESCRIPTORSIZE; b++) {\n"
//"       vec2 dp1 = descriptors1(b);\n"
//"       vec2 dp2 = descriptors2(b);\n"
"       vec2 dp1 = vec2(descriptors1x[b],descriptors1y[b]);\n"
"       vec2 dp2 = vec2(descriptors2x[b],descriptors2y[b]);\n"
"       vec2 d1 = (kp + vec2(dot(cosid,dp1), dot(nsicd,dp1)))*sc;\n"
"       vec2 d2 = (kp + vec2(dot(cosid,dp2), dot(nsicd,dp2)))*sc;\n"
"       float l1 = textureLod(s, d1, 0.0).x;\n"
"       float l2 = textureLod(s, d2, 0.0).x;\n"
"       unsigned int b2 = l2 < l1 ? unsigned int(1) : unsigned int(0);\n"
"       if (b2 != ((dBits[b>>5]>>unsigned int(b & 31)) & unsigned int(1)) ) {\n"
"           vec2 gr = vec2( textureLod(s, d2 - gdx, 0.0).x - textureLod(s, d2 + gdx, 0.0).x , textureLod(s, d2 - gdy, 0.0).x - textureLod(s, d2 + gdy, 0.0).x )\n"
"                    -vec2( textureLod(s, d1 - gdx, 0.0).x - textureLod(s, d1 + gdx, 0.0).x , textureLod(s, d1 - gdy, 0.0).x - textureLod(s, d1 + gdy, 0.0).x );\n"
"           if (b2 != unsigned int(0)) gr = -gr;\n"
"           if (stepping != 0) gr = vec2((gr.x == 0.0) ? 0.0 : (gr.x > 0.0) ? 1.0 : -1.0, (gr.y == 0.0) ? 0.0 : (gr.y > 0.0) ? 1.0 : -1.0);\n"
"           xya += gr * distance(d2,d1);\n"
"       }\n"
"   }\n"
"   float l = length(xya);\n"
"   if (l != 0.0) xya *= step / mipScale / l;\n"
"   vec2 r = kp + vec2(cosa * xya.x + sina * xya.y, -sina * xya.x + cosa * xya.y);\n"
"   const bool clipping = false; if (clipping) {\n"
"       float w = float(mipMapWidth[0]);\n"
"       float h = float(mipMapHeight[0]);\n"
"       if (r.x < 0.0) r.x = 0.0;\n"
"       if (r.y < 0.0) r.y = 0.0;\n"
"       if (r.x >= w - 1.0) r.x = w - 1.0;\n"
"       if (r.y >= h - 1.0) r.y = h - 1.0;\n"
"   }\n"
"   return r;\n"
"}\n"
"void main()\n"
"{\n"
"   int txx = int(floor(p.x+0.25));\n"
"   int txy = int(floor(p.y+0.25));\n"
"   int descriptorNr = txx + txy * texWidthKeyPoints;\n"
"   int descriptorNrX = descriptorNr % texWidthDescriptors;\n"
"   int descriptorNrY = descriptorNr / texWidthDescriptors;\n"
"   vec2 descriptorNrXY = vec2(float(descriptorNrX) + 0.25,float(descriptorNrY) + 0.25) / vec2(float(texWidthDescriptors),float(texHeightDescriptors));\n"
"   for(int v = 0; v < 2; v++) {\n"
"       vec2 kp = t2d_nearest(keyPointsx,keyPointsy,txx,txy,texWidthKeyPoints,texHeightKeyPoints);\n"
"       for(int i = mipEnd; i >= 0; i--) {\n"
"               uvec4 descriptorNeeded;\n"
"               switch(i) {\n"
"                   case  0:descriptorNeeded = textureLod( descriptors_0, descriptorNrXY, 0.0); break;\n"
"                   case  1:descriptorNeeded = textureLod( descriptors_1, descriptorNrXY, 0.0); break;\n"
"                   case  2:descriptorNeeded = textureLod( descriptors_2, descriptorNrXY, 0.0); break;\n"
"                   case  3:descriptorNeeded = textureLod( descriptors_3, descriptorNrXY, 0.0); break;\n"
"                   case  4:descriptorNeeded = textureLod( descriptors_4, descriptorNrXY, 0.0); break;\n"
"                   case  5:descriptorNeeded = textureLod( descriptors_5, descriptorNrXY, 0.0); break;\n"
"                   case  6:descriptorNeeded = textureLod( descriptors_6, descriptorNrXY, 0.0); break;\n"
"                   case  7:descriptorNeeded = textureLod( descriptors_7, descriptorNrXY, 0.0); break;\n"
"                   case  8:descriptorNeeded = textureLod( descriptors_8, descriptorNrXY, 0.0); break;\n"
"                   case  9:descriptorNeeded = textureLod( descriptors_9, descriptorNrXY, 0.0); break;\n"
"                   case 10:descriptorNeeded = textureLod(descriptors_10, descriptorNrXY, 0.0); break;\n"
"                   case 11:descriptorNeeded = textureLod(descriptors_11, descriptorNrXY, 0.0); break;\n"
"                   case 12:descriptorNeeded = textureLod(descriptors_12, descriptorNrXY, 0.0); break;\n"
"                   case 13:descriptorNeeded = textureLod(descriptors_13, descriptorNrXY, 0.0); break;\n"
"                   case 14:descriptorNeeded = textureLod(descriptors_14, descriptorNrXY, 0.0); break;\n"
"                   case 15:descriptorNeeded = textureLod(descriptors_15, descriptorNrXY, 0.0); break;\n"
"                   case 16:descriptorNeeded = textureLod(descriptors_16, descriptorNrXY, 0.0); break;\n"
"                   case 17:descriptorNeeded = textureLod(descriptors_17, descriptorNrXY, 0.0); break;\n"
"                   case 18:descriptorNeeded = textureLod(descriptors_18, descriptorNrXY, 0.0); break;\n"
"                   case 19:descriptorNeeded = textureLod(descriptors_19, descriptorNrXY, 0.0); break;\n"
"                   case 20:descriptorNeeded = textureLod(descriptors_20, descriptorNrXY, 0.0); break;\n"
"                   case 21:descriptorNeeded = textureLod(descriptors_21, descriptorNrXY, 0.0); break;\n"
"                   case 22:descriptorNeeded = textureLod(descriptors_22, descriptorNrXY, 0.0); break;\n"
"                   case 23:descriptorNeeded = textureLod(descriptors_23, descriptorNrXY, 0.0); break;\n"
"                   case 24:descriptorNeeded = textureLod(descriptors_24, descriptorNrXY, 0.0); break;\n"
"                   case 25:descriptorNeeded = textureLod(descriptors_25, descriptorNrXY, 0.0); break;\n"
"                   case 26:descriptorNeeded = textureLod(descriptors_26, descriptorNrXY, 0.0); break;\n"
"                   case 27:descriptorNeeded = textureLod(descriptors_27, descriptorNrXY, 0.0); break;\n"
"                   case 28:descriptorNeeded = textureLod(descriptors_28, descriptorNrXY, 0.0); break;\n"
"                   case 29:descriptorNeeded = textureLod(descriptors_29, descriptorNrXY, 0.0); break;\n"
"                   case 30:descriptorNeeded = textureLod(descriptors_30, descriptorNrXY, 0.0); break;\n"
"                   case 31:descriptorNeeded = textureLod(descriptors_31, descriptorNrXY, 0.0); break;\n"
"               }\n"
"           for(int k = 0; k < STEPCOUNT; k++) {\n"
"               mipScale = pow(MIPSCALE, float(i));\n"
"               descriptorScale = 1.0 / mipScale;\n"
"               step = STEPSIZE * descriptorScale;\n"
"               descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.0 - SCALEINVARIANCE;\n"
"               angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.0 - ROTATIONINVARIANCE) / 360.0 * 2.0 * 3.1415927;\n"
"               switch(i) {\n"
"                   case  0: kp = refineKeyPoint( mipMaps_0, kp, i, descriptorNeeded); break;\n"
"                   case  1: kp = refineKeyPoint( mipMaps_1, kp, i, descriptorNeeded); break;\n"
"                   case  2: kp = refineKeyPoint( mipMaps_2, kp, i, descriptorNeeded); break;\n"
"                   case  3: kp = refineKeyPoint( mipMaps_3, kp, i, descriptorNeeded); break;\n"
"                   case  4: kp = refineKeyPoint( mipMaps_4, kp, i, descriptorNeeded); break;\n"
"                   case  5: kp = refineKeyPoint( mipMaps_5, kp, i, descriptorNeeded); break;\n"
"                   case  6: kp = refineKeyPoint( mipMaps_6, kp, i, descriptorNeeded); break;\n"
"                   case  7: kp = refineKeyPoint( mipMaps_7, kp, i, descriptorNeeded); break;\n"
"                   case  8: kp = refineKeyPoint( mipMaps_8, kp, i, descriptorNeeded); break;\n"
"                   case  9: kp = refineKeyPoint( mipMaps_9, kp, i, descriptorNeeded); break;\n"
"                   case 10: kp = refineKeyPoint(mipMaps_10, kp, i, descriptorNeeded); break;\n"
"                   case 11: kp = refineKeyPoint(mipMaps_11, kp, i, descriptorNeeded); break;\n"
"                   case 12: kp = refineKeyPoint(mipMaps_12, kp, i, descriptorNeeded); break;\n"
"                   case 13: kp = refineKeyPoint(mipMaps_13, kp, i, descriptorNeeded); break;\n"
"                   case 14: kp = refineKeyPoint(mipMaps_14, kp, i, descriptorNeeded); break;\n"
"                   case 15: kp = refineKeyPoint(mipMaps_15, kp, i, descriptorNeeded); break;\n"
"                   case 16: kp = refineKeyPoint(mipMaps_16, kp, i, descriptorNeeded); break;\n"
"                   case 17: kp = refineKeyPoint(mipMaps_17, kp, i, descriptorNeeded); break;\n"
"                   case 18: kp = refineKeyPoint(mipMaps_18, kp, i, descriptorNeeded); break;\n"
"                   case 19: kp = refineKeyPoint(mipMaps_19, kp, i, descriptorNeeded); break;\n"
"                   case 20: kp = refineKeyPoint(mipMaps_20, kp, i, descriptorNeeded); break;\n"
"                   case 21: kp = refineKeyPoint(mipMaps_21, kp, i, descriptorNeeded); break;\n"
"                   case 22: kp = refineKeyPoint(mipMaps_22, kp, i, descriptorNeeded); break;\n"
"                   case 23: kp = refineKeyPoint(mipMaps_23, kp, i, descriptorNeeded); break;\n"
"                   case 24: kp = refineKeyPoint(mipMaps_24, kp, i, descriptorNeeded); break;\n"
"                   case 25: kp = refineKeyPoint(mipMaps_25, kp, i, descriptorNeeded); break;\n"
"                   case 26: kp = refineKeyPoint(mipMaps_26, kp, i, descriptorNeeded); break;\n"
"                   case 27: kp = refineKeyPoint(mipMaps_27, kp, i, descriptorNeeded); break;\n"
"                   case 28: kp = refineKeyPoint(mipMaps_28, kp, i, descriptorNeeded); break;\n"
"                   case 29: kp = refineKeyPoint(mipMaps_29, kp, i, descriptorNeeded); break;\n"
"                   case 30: kp = refineKeyPoint(mipMaps_30, kp, i, descriptorNeeded); break;\n"
"                   case 31: kp = refineKeyPoint(mipMaps_31, kp, i, descriptorNeeded); break;\n"
"               }\n"
"           }\n"
"       }\n"
"       switch(v) {\n"
"           case 0: frag_color.xy = kp; break;\n"
"           case 1: frag_color.zw = kp; break;\n"
"       }\n"
"   vec2 sc = vec2(1.0/float(mipMapWidth[0]),1.0/float(mipMapHeight[0]));\n"
"   }\n"
"}\n"
"\n";
const std::string displayMipMapProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
"in vec2 p;\n"
"out vec4 frag_color;\n"
"uniform sampler2D mipMaps_0;\n"
"uniform sampler2D mipMaps_1;\n"
"uniform sampler2D mipMaps_2;\n"
"uniform sampler2D mipMaps_3;\n"
"uniform sampler2D mipMaps_4;\n"
"uniform sampler2D mipMaps_5;\n"
"uniform sampler2D mipMaps_6;\n"
"uniform sampler2D mipMaps_7;\n"
"uniform sampler2D mipMaps_8;\n"
"uniform sampler2D mipMaps_9;\n"
"uniform sampler2D mipMaps_10;\n"
"uniform sampler2D mipMaps_11;\n"
"uniform sampler2D mipMaps_12;\n"
"uniform sampler2D mipMaps_13;\n"
"uniform sampler2D mipMaps_14;\n"
"uniform sampler2D mipMaps_15;\n"
"uniform sampler2D mipMaps_16;\n"
"uniform sampler2D mipMaps_17;\n"
"uniform sampler2D mipMaps_18;\n"
"uniform sampler2D mipMaps_19;\n"
"uniform sampler2D mipMaps_20;\n"
"uniform sampler2D mipMaps_21;\n"
"uniform sampler2D mipMaps_22;\n"
"uniform sampler2D mipMaps_23;\n"
"uniform sampler2D mipMaps_24;\n"
"uniform sampler2D mipMaps_25;\n"
"uniform sampler2D mipMaps_26;\n"
"uniform sampler2D mipMaps_27;\n"
"uniform sampler2D mipMaps_28;\n"
"uniform sampler2D mipMaps_29;\n"
"uniform sampler2D mipMaps_30;\n"
"uniform sampler2D mipMaps_31;\n"
"uniform int mipMapWidth[32];\n"
"uniform int mipMapHeight[32];\n"
"void main()\n"
"{\n"
"   int i = 0;\n"
"   vec2 sc = vec2(1.0/float(mipMapWidth[i]),1.0/float(mipMapHeight[i]));\n"
"   vec2 xy = p * sc;\n"
"   float mip;\n"
"   switch(i) {\n"
"       case  0:mip = textureLod( mipMaps_0, xy, 0.0).x; break;\n"
"       case  1:mip = textureLod( mipMaps_1, xy, 0.0).x; break;\n"
"       case  2:mip = textureLod( mipMaps_2, xy, 0.0).x; break;\n"
"       case  3:mip = textureLod( mipMaps_3, xy, 0.0).x; break;\n"
"       case  4:mip = textureLod( mipMaps_4, xy, 0.0).x; break;\n"
"       case  5:mip = textureLod( mipMaps_5, xy, 0.0).x; break;\n"
"       case  6:mip = textureLod( mipMaps_6, xy, 0.0).x; break;\n"
"       case  7:mip = textureLod( mipMaps_7, xy, 0.0).x; break;\n"
"       case  8:mip = textureLod( mipMaps_8, xy, 0.0).x; break;\n"
"       case  9:mip = textureLod( mipMaps_9, xy, 0.0).x; break;\n"
"       case 10:mip = textureLod(mipMaps_10, xy, 0.0).x; break;\n"
"       case 11:mip = textureLod(mipMaps_11, xy, 0.0).x; break;\n"
"       case 12:mip = textureLod(mipMaps_12, xy, 0.0).x; break;\n"
"       case 13:mip = textureLod(mipMaps_13, xy, 0.0).x; break;\n"
"       case 14:mip = textureLod(mipMaps_14, xy, 0.0).x; break;\n"
"       case 15:mip = textureLod(mipMaps_15, xy, 0.0).x; break;\n"
"       case 16:mip = textureLod(mipMaps_16, xy, 0.0).x; break;\n"
"       case 17:mip = textureLod(mipMaps_17, xy, 0.0).x; break;\n"
"       case 18:mip = textureLod(mipMaps_18, xy, 0.0).x; break;\n"
"       case 19:mip = textureLod(mipMaps_19, xy, 0.0).x; break;\n"
"       case 20:mip = textureLod(mipMaps_20, xy, 0.0).x; break;\n"
"       case 21:mip = textureLod(mipMaps_21, xy, 0.0).x; break;\n"
"       case 22:mip = textureLod(mipMaps_22, xy, 0.0).x; break;\n"
"       case 23:mip = textureLod(mipMaps_23, xy, 0.0).x; break;\n"
"       case 24:mip = textureLod(mipMaps_24, xy, 0.0).x; break;\n"
"       case 25:mip = textureLod(mipMaps_25, xy, 0.0).x; break;\n"
"       case 26:mip = textureLod(mipMaps_26, xy, 0.0).x; break;\n"
"       case 27:mip = textureLod(mipMaps_27, xy, 0.0).x; break;\n"
"       case 28:mip = textureLod(mipMaps_28, xy, 0.0).x; break;\n"
"       case 29:mip = textureLod(mipMaps_29, xy, 0.0).x; break;\n"
"       case 30:mip = textureLod(mipMaps_30, xy, 0.0).x; break;\n"
"       case 31:mip = textureLod(mipMaps_31, xy, 0.0).x; break;\n"
"   }\n"
"   frag_color.x = mip;"
"   frag_color.y = mip;"
"   frag_color.z = mip;"
"   frag_color.w = mip;"
"}\n"
"\n";
const std::string basicProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
"layout(location = 0) in vec2 pos;\n"
"uniform int texWidthKeyPoints;\n"
"uniform int texHeightKeyPoints;\n"
"out vec2 p;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(pos.xy, 0.0, 1.0);\n"
"    p = ((pos.xy + 1.0) * 0.5) * vec2(float(texWidthKeyPoints),float(texHeightKeyPoints));\n"
"}\n"
"\n";

void checkGLError();
void checkShader(GLuint shader);
void checkProgram(GLuint program);
GLuint pixelShader(const char* shaderProgram, unsigned int size);
GLuint vertexShader(const char* shaderProgram, unsigned int size);
GLuint program(GLuint vertexShader, GLuint pixelShader);
void createUint324RenderTarget(std::pair<GLuint, GLuint>& dest, int width, int height, bool nearest);
void createFloat4RenderTarget(std::pair<GLuint, GLuint>& dest, int width, int height, bool nearest);
void upload2DFloatTexture(const GLuint& tex, const float* data, unsigned int width, unsigned int height, bool nearest);
void upload2DUnsignedCharTexture(const GLuint& tex, const unsigned char* data, unsigned int width, unsigned int height, bool nearest);
void upload1DFloatTexture(const GLuint& tex, const float* data, unsigned int count, int& width, int& height, bool nearest);
void upload2Duint324Texture(const GLuint& tex, const unsigned int* data, unsigned int width, unsigned int height, bool nearest);
void upload1Duint324Texture(const GLuint& tex, const unsigned int* data, unsigned int count, int& width, int& height, bool nearest);

#if __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__
void glfwErrorCallback(int error_code, const char* description) {
    lcLog("Error(%d): %s\n", error_code, description);
}
#endif // __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__

void initOpenGL() {
    defaultDescriptorShape;

#if __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        exit(0);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    window = glfwCreateWindow(1920, 1080, "Lucyte", NULL, NULL);
    if (window == NULL)
        exit(0);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glewInit();
#endif // __CURRENT_PLATFORM__ == __PLATFORM_WINDOWS__

    std::string descriptors1x = "float descriptors1x[" + std::to_string(DESCRIPTORSIZE) + "] = float [](";
    std::string descriptors1y = "float descriptors1y[" + std::to_string(DESCRIPTORSIZE) + "] = float [](";
    std::string descriptors2x = "float descriptors2x[" + std::to_string(DESCRIPTORSIZE) + "] = float [](";
    std::string descriptors2y = "float descriptors2y[" + std::to_string(DESCRIPTORSIZE) + "] = float [](";
    for (int i = 0; i < DESCRIPTORSIZE; i++) {
        if (i != 0) {
            descriptors1x += ",";
            descriptors1y += ",";
            descriptors2x += ",";
            descriptors2y += ",";
        }
        descriptors1x += std::to_string(descriptorsX1[i]);
        descriptors1y += std::to_string(descriptorsY1[i]);
        descriptors2x += std::to_string(descriptorsX2[i]);
        descriptors2y += std::to_string(descriptorsY2[i]);
    }
    descriptors1x += ");\n";
    descriptors1y += ");\n";
    descriptors2x += ");\n";
    descriptors2y += ");\n";

    std::string refineKeys = "#version 300 es\nprecision highp float;precision highp int;\n" + descriptors1x + descriptors1y + descriptors2x + descriptors2y + refineKeyPointsProgram;
    std::string sampleDescriptors = "#version 300 es\nprecision highp float;precision highp int;\n" + descriptors1x + descriptors1y + descriptors2x + descriptors2y + sampleDescriptorProgram;

    openGLFragmentShader_sampleDescriptors = pixelShader(sampleDescriptors.c_str(), (unsigned int)sampleDescriptors.length());
    openGLFragmentShader_refineKeyPoints = pixelShader(refineKeys.c_str(), (unsigned int)refineKeys.length());
    openGLFragmentShader_displayMipMap = pixelShader(displayMipMapProgram.c_str(), (unsigned int)displayMipMapProgram.length());
    openGLVertexShader_basic = vertexShader(basicProgram.c_str(), (unsigned int)basicProgram.length());
    openGLProgram_sampleDescriptors = program(openGLVertexShader_basic, openGLFragmentShader_sampleDescriptors);
    openGLProgram_refineKeyPoints = program(openGLVertexShader_basic, openGLFragmentShader_refineKeyPoints);
    openGLProgram_displayMipMap = program(openGLVertexShader_basic, openGLFragmentShader_displayMipMap);
    for (int i = 0; i < MIPMAPIDCOUNT; i++) glGenTextures(MAXMIPMAPS, openGLMipMaps[i]);
    glGenTextures(4, openGLDescriptorShape);
    for (int i = 0; i < KEYPOINTIDCOUNT; i++) {
        glGenTextures(1, &(openGLKeyPointsX[i]));
        glGenTextures(1, &(openGLKeyPointsY[i]));
    }
    for (int j = 0; j < MAXMIPMAPS; j++) {
        for (int i = 0; i < DESCRIPTORIDCOUNT; i++) {
            glGenTextures(1, &(openGLDescriptors[i][j]));
        }
    }
    uploadDescriptorShape_openGL();
}

void checkGLError() {
    int error = (int)glGetError();
    if (error != GL_NO_ERROR) {
        lcLog("OpenGL error(%d)\n", error);
    }
}

void checkShader(GLuint shader) {
    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled); checkGLError();
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength); checkGLError();
        std::vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]); checkGLError();
        lcLog("%s", &(errorLog[0]));
        glDeleteShader(shader); checkGLError();
        exit(0);
    }
}

void checkProgram(GLuint program) {
    GLint isCompiled = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isCompiled); checkGLError();
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength); checkGLError();
        std::vector<char> errorLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &errorLog[0]); checkGLError();
        lcLog("%s", &(errorLog[0]));
        glDeleteProgram(program); checkGLError();
        exit(0);
    }
}

GLuint pixelShader(const char *shaderProgram, unsigned int size) {
    GLint length[] = { (GLint)size };
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint vertexShader(const char* shaderProgram, unsigned int size) {
    GLint length[] = { (GLint)size };
    GLuint shader = glCreateShader(GL_VERTEX_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint program(const GLuint vertexShader, GLuint pixelShader) {
    GLuint program = glCreateProgram(); checkGLError();
    glAttachShader(program, vertexShader); checkGLError();
    glAttachShader(program, pixelShader); checkGLError();
    glLinkProgram(program); checkGLError();
    checkProgram(program);
    return program;
}

void createUint324RenderTarget(std::pair<GLuint, GLuint> &dest, int width, int height, bool nearest) {
    glDeleteFramebuffers(1, &dest.first); checkGLError();
    glDeleteTextures(1, &dest.second); checkGLError();
    GLuint framebufferId = 0;
    glGenFramebuffers(1, &framebufferId); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferId); checkGLError();
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0); checkGLError();
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
    dest = { framebufferId, renderedTexture };
}

void createFloat4RenderTarget(std::pair<GLuint, GLuint>& dest, int width, int height, bool nearest) {
    glDeleteFramebuffers(1, &dest.first); checkGLError();
    glDeleteTextures(1, &dest.second); checkGLError();
    GLuint framebufferId = 0;
    glGenFramebuffers(1, &framebufferId); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferId); checkGLError();
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderedTexture, 0); checkGLError();
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
    dest = { framebufferId, renderedTexture };
}

void upload2DFloatTexture(const GLuint& tex, const float* data, unsigned int width, unsigned int height, bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (int)width, (int)height, 0, GL_RED, GL_FLOAT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload2DUnsignedCharTexture(const GLuint& tex, const unsigned char* data, unsigned int width, unsigned int height, bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, (int)width, (int)height, 0, GL_RED, GL_UNSIGNED_BYTE, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload1DFloatTexture(const GLuint& tex, const float* data, unsigned int count, int& width, int& height, bool nearest) {
    width = (int)count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (int)(count + maxTextureSize - 1) / maxTextureSize;
        float* data2 = new float[width * height];
        memcpy(data2, data, count * sizeof(float));
        upload2DFloatTexture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2DFloatTexture(tex, data, width, height, nearest);
}

void upload2Duint324Texture(const GLuint& tex, const unsigned int* data, unsigned int width, unsigned int height, bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, (int)width, (int)height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}


void upload1Duint324Texture(const GLuint& tex, const unsigned int* data, unsigned int count, int &width, int &height, bool nearest) {
    width = (int)count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (int)(count + maxTextureSize - 1) / maxTextureSize;
        unsigned int* data2 = new unsigned int[width * height * 4];
        memcpy(data2, data, count * sizeof(unsigned int) * 4);
        upload2Duint324Texture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2Duint324Texture(tex, data, width, height, nearest);
}

void uploadDescriptorShape_openGL() {
    upload1DFloatTexture(openGLDescriptorShape[0], descriptorsX1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[1], descriptorsY1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[2], descriptorsX2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[3], descriptorsY2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
}

void uploadMipMaps_openGL(int mipmapsId, const std::vector<MipMap>& mipMaps) {
    for (int i = 0; i < mipMaps.size(); i++) {
        openGLMipMapsTextureWidth[mipmapsId][i] = mipMaps[i].width;
        openGLMipMapsTextureHeight[mipmapsId][i] = mipMaps[i].height;
        upload2DUnsignedCharTexture(openGLMipMaps[mipmapsId][i], mipMaps[i].data, mipMaps[i].width, mipMaps[i].height, false);
    }
    openGLMipMapCount[mipmapsId] = (int)mipMaps.size();
}

static std::vector<float> kpx;
static std::vector<float> kpy;

void uploadKeyPoints_openGL(int keyPointsId, const std::vector<KeyPoint>& keyPoints) {
    kpx.resize(keyPoints.size());
    kpy.resize(keyPoints.size());
    for (int i = int(keyPoints.size()) - 1; i >= 0; i--) {
        kpx[i] = keyPoints[i].x;
        kpy[i] = keyPoints[i].y;
    }
    upload1DFloatTexture(openGLKeyPointsX[keyPointsId], &(kpx[0]), (unsigned int)keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    upload1DFloatTexture(openGLKeyPointsY[keyPointsId], &(kpy[0]), (unsigned int)keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    openGLKeyPointCount[keyPointsId] = (int)keyPoints.size();
}

void uploadDescriptors_openGL(int descriptorsId, int mipEnd, const std::vector<std::vector<Descriptor>>& sourceMips) {
    for (int i2 = 0; i2 <= mipEnd; i2++) {
        const std::vector<Descriptor>& descriptors = sourceMips[i2];
        std::vector<unsigned int> bits; bits.resize(4 * descriptors.size());
        for (int i = int(descriptors.size()) - 1; i >= 0; i--) {
            for (int j = 0; j < Descriptor::uint32count; j++) {
                bits[j + i * 4] = descriptors[i].bits[j];
            }
        }
        upload1Duint324Texture(openGLDescriptors[descriptorsId][i2], &(bits[0]), (int)bits.size() / 4, openGLDescriptorsTextureWidth[descriptorsId][i2], openGLDescriptorsTextureHeight[descriptorsId][i2], true);
    }
}

void fullScreenRect() {
    GLfloat model[] =
    {
        -1.f, -1.f, // lower left
        -1.f, 1.f, // upper left
        1.f, -1.f, // lower right
        1.f, 1.f  // upper right
    };
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, model);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void displayMipMap(int mipmapsId, int mipEnd) {
    GLint displayMipMap_location_mipMaps[32]; for (int i = 0; i < 32; i++) { displayMipMap_location_mipMaps[i] = glGetUniformLocation(openGLProgram_displayMipMap, ("mipMaps_" + std::to_string(i)).c_str()); checkGLError(); }
    GLint displayMipMap_location_mipMapWidth = glGetUniformLocation(openGLProgram_displayMipMap, "mipMapWidth"); checkGLError();
    GLint displayMipMap_location_mipMapHeight = glGetUniformLocation(openGLProgram_displayMipMap, "mipMapHeight"); checkGLError();
    GLint displayMipMap_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_displayMipMap, "texWidthKeyPoints"); checkGLError();
    GLint displayMipMap_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_displayMipMap, "texHeightKeyPoints"); checkGLError();
    glUseProgram(openGLProgram_displayMipMap); checkGLError();
    int j = 6 + 8;
    for (int i = 0; i <= mipEnd; i++) {
        glActiveTexture(GL_TEXTURE0 + j); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId][i]); checkGLError();
        glUniform1i(displayMipMap_location_mipMaps[i], j); checkGLError();
        j++;
    }
    std::vector<GLint> mipMapWidth;
    std::vector<GLint> mipMapHeight;
    for (int i = 0; i <= mipEnd; i++) {
        mipMapWidth.push_back(openGLMipMapsTextureWidth[mipmapsId][i]);
        mipMapHeight.push_back(openGLMipMapsTextureHeight[mipmapsId][i]);
    }
    glUniform1iv(displayMipMap_location_mipMapWidth, (GLint)mipMapWidth.size(), &(mipMapWidth[0])); checkGLError();
    glUniform1iv(displayMipMap_location_mipMapHeight, (GLint)mipMapHeight.size(), &(mipMapHeight[0])); checkGLError();
    glUniform1i(displayMipMap_location_texWidthKeyPoints, 1000); checkGLError();
    glUniform1i(displayMipMap_location_texHeightKeyPoints, 1000); checkGLError();
    glViewport(0, 0, 1000, 1000);
    fullScreenRect();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void sampleDescriptors_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int mipEnd, std::vector<std::vector<Descriptor>>& destMips, float DESCRIPTORSCALE2, float MIPSCALE) {

    GLint sampleDescriptors_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthDescriptorShape"); checkGLError();
    GLint sampleDescriptors_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightDescriptorShape"); checkGLError();
    GLint sampleDescriptors_location_texWidthMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthMipMap"); checkGLError();
    GLint sampleDescriptors_location_texHeightMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightMipMap"); checkGLError();
    GLint sampleDescriptors_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthKeyPoints"); checkGLError();
    GLint sampleDescriptors_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightKeyPoints"); checkGLError();
    GLint sampleDescriptors_location_DESCRIPTORSIZE = glGetUniformLocation(openGLProgram_sampleDescriptors, "DESCRIPTORSIZE"); checkGLError();
    GLint sampleDescriptors_location_descriptorScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptorScale"); checkGLError();
    GLint sampleDescriptors_location_mipScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipScale"); checkGLError();
    GLint sampleDescriptors_location_descriptor1x = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor1x"); checkGLError();
    GLint sampleDescriptors_location_descriptor1y = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor1y"); checkGLError();
    GLint sampleDescriptors_location_descriptor2x = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor2x"); checkGLError();
    GLint sampleDescriptors_location_descriptor2y = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor2y"); checkGLError();
    GLint sampleDescriptors_location_mipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipMap"); checkGLError();
    GLint sampleDescriptors_location_keyPointsx = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsx"); checkGLError();
    GLint sampleDescriptors_location_keyPointsy = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsy"); checkGLError();

    destMips.resize(openGLMipMapCount[mipmapsId]);
    for (int mipMap = 0; mipMap <= mipEnd; mipMap++) {
        std::vector<Descriptor>& descriptors = destMips[mipMap];
        destMips[mipMap].resize(openGLKeyPointCount[keyPointsId]);
        const float mipScale = powf(MIPSCALE, float(mipMap));
        const float descriptorScale = DESCRIPTORSCALE2 / mipScale;

        if (openGLKeyPointsTextureWidth[keyPointsId] != openGLDescriptorRenderTargetWidth[keyPointsId][mipMap] || openGLKeyPointsTextureHeight[keyPointsId] != openGLDescriptorRenderTargetHeight[keyPointsId][mipMap]) {
            openGLDescriptorRenderTargetWidth[keyPointsId][mipMap] = openGLKeyPointsTextureWidth[keyPointsId];
            openGLDescriptorRenderTargetHeight[keyPointsId][mipMap] = openGLKeyPointsTextureHeight[keyPointsId];
            createUint324RenderTarget(openGLDescriptorRenderTarget[keyPointsId][mipMap], (int)openGLDescriptorRenderTargetWidth[keyPointsId][mipMap], (int)openGLDescriptorRenderTargetHeight[keyPointsId][mipMap], true);
        }

        glUseProgram(openGLProgram_sampleDescriptors); checkGLError();
        glUniform1i(sampleDescriptors_location_texWidthDescriptorShape, openGLDescriptorShapeTextureWidth); checkGLError();
        glUniform1i(sampleDescriptors_location_texHeightDescriptorShape, openGLDescriptorShapeTextureHeight); checkGLError();
        glUniform1i(sampleDescriptors_location_texWidthMipMap, openGLMipMapsTextureWidth[mipmapsId][mipMap]); checkGLError();
        glUniform1i(sampleDescriptors_location_texHeightMipMap, openGLMipMapsTextureHeight[mipmapsId][mipMap]); checkGLError();
        glUniform1i(sampleDescriptors_location_texWidthKeyPoints, openGLKeyPointsTextureWidth[keyPointsId]); checkGLError();
        glUniform1i(sampleDescriptors_location_texHeightKeyPoints, openGLKeyPointsTextureHeight[keyPointsId]); checkGLError();
        glUniform1i(sampleDescriptors_location_DESCRIPTORSIZE, DESCRIPTORSIZE); checkGLError();
        glUniform1f(sampleDescriptors_location_descriptorScale, descriptorScale); checkGLError();
        glUniform1f(sampleDescriptors_location_mipScale, mipScale); checkGLError();

        glActiveTexture(GL_TEXTURE0); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLKeyPointsX[keyPointsId]); checkGLError();
        glUniform1i(sampleDescriptors_location_keyPointsx, 0); checkGLError();
        glActiveTexture(GL_TEXTURE1); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLKeyPointsY[keyPointsId]); checkGLError();
        glUniform1i(sampleDescriptors_location_keyPointsy, 1); checkGLError();
        glActiveTexture(GL_TEXTURE2); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[0]); checkGLError();
        glUniform1i(sampleDescriptors_location_descriptor1x, 2); checkGLError();
        glActiveTexture(GL_TEXTURE3); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[1]); checkGLError();
        glUniform1i(sampleDescriptors_location_descriptor1y, 3); checkGLError();
        glActiveTexture(GL_TEXTURE4); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[2]); checkGLError();
        glUniform1i(sampleDescriptors_location_descriptor2x, 4); checkGLError();
        glActiveTexture(GL_TEXTURE5); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[3]); checkGLError();
        glUniform1i(sampleDescriptors_location_descriptor2y, 5); checkGLError();
        glActiveTexture(GL_TEXTURE6); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId][mipMap]); checkGLError();
        glUniform1i(sampleDescriptors_location_mipMap, 6); checkGLError();

        glBindFramebuffer(GL_FRAMEBUFFER, openGLDescriptorRenderTarget[keyPointsId][mipMap].first); checkGLError();
        const int w = (int)openGLDescriptorRenderTargetWidth[keyPointsId][mipMap];
        const int h = (int)openGLDescriptorRenderTargetHeight[keyPointsId][mipMap];
        glViewport(0, 0, w, h);
        fullScreenRect();
    }
    for (int mipMap = 0; mipMap <= mipEnd; mipMap++) {
        std::vector<Descriptor>& descriptors = destMips[mipMap];
        glBindFramebuffer(GL_FRAMEBUFFER, openGLDescriptorRenderTarget[keyPointsId][mipMap].first); checkGLError();
        const int w = (int)openGLDescriptorRenderTargetWidth[keyPointsId][mipMap];
        const int h = (int)openGLDescriptorRenderTargetHeight[keyPointsId][mipMap];
        unsigned int* data = new unsigned int[4 * w * h];
        glReadPixels(0, 0, w, h, GL_RGBA_INTEGER, GL_UNSIGNED_INT, data); checkGLError();
        for (int i = 0; i < descriptors.size(); i++) {
            int x = i % w;
            int y = i / w;
            int texAdr = x + y * w;
            for (int j = 0; j < Descriptor::uint32count; ++j) {
                descriptors[i].bits[j] = data[texAdr * 4 + j];
            }
        }
        delete[] data;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void refineKeyPoints_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int keyPointCount, int mipEnd, int STEPCOUNT, bool stepping, float MIPSCALE, float STEPSIZE, float SCALEINVARIANCE, float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors) {

    if (openGLKeyPointsTextureWidth[keyPointsId] != openGLKeyPointsRenderTargetWidth[keyPointsId] || openGLKeyPointsTextureHeight[keyPointsId] != (int)openGLKeyPointsRenderTargetHeight[keyPointsId]) {
        openGLKeyPointsRenderTargetWidth[keyPointsId] = openGLKeyPointsTextureWidth[keyPointsId];
        openGLKeyPointsRenderTargetHeight[keyPointsId] = openGLKeyPointsTextureHeight[keyPointsId];
        createFloat4RenderTarget(openGLKeyPointsRenderTarget[keyPointsId], (int)openGLKeyPointsRenderTargetWidth[keyPointsId], (int)openGLKeyPointsRenderTargetHeight[keyPointsId], true);
    }

    // OpenGL is not working, yet.
    GLint refineKeyPoints_location_mipMaps[32]; for (int i = 0; i < 32; i++) { refineKeyPoints_location_mipMaps[i] = glGetUniformLocation(openGLProgram_refineKeyPoints, ("mipMaps_" + std::to_string(i)).c_str()); checkGLError(); }
    GLint refineKeyPoints_location_descriptors[32]; for (int i = 0; i < 32; i++) { refineKeyPoints_location_descriptors[i] = glGetUniformLocation(openGLProgram_refineKeyPoints, ("descriptors_" + std::to_string(i)).c_str()); checkGLError(); }
    GLint refineKeyPoints_location_mipMapWidth = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapWidth"); checkGLError();
    GLint refineKeyPoints_location_mipMapHeight = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapHeight"); checkGLError();
    GLint refineKeyPoints_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptorShape"); checkGLError();
    GLint refineKeyPoints_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptorShape"); checkGLError();
    GLint refineKeyPoints_location_texWidthDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptors"); checkGLError();
    GLint refineKeyPoints_location_texHeightDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptors"); checkGLError();
    GLint refineKeyPoints_location_DESCRIPTORSIZE = glGetUniformLocation(openGLProgram_refineKeyPoints, "DESCRIPTORSIZE"); checkGLError();
    GLint refineKeyPoints_location_stepping = glGetUniformLocation(openGLProgram_refineKeyPoints, "stepping"); checkGLError();
    GLint refineKeyPoints_location_mipEnd = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipEnd"); checkGLError();
    GLint refineKeyPoints_location_STEPCOUNT = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPCOUNT"); checkGLError();
    GLint refineKeyPoints_location_MIPSCALE = glGetUniformLocation(openGLProgram_refineKeyPoints, "MIPSCALE"); checkGLError();
    GLint refineKeyPoints_location_STEPSIZE = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPSIZE"); checkGLError();
    GLint refineKeyPoints_location_SCALEINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "SCALEINVARIANCE"); checkGLError();
    GLint refineKeyPoints_location_ROTATIONINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "ROTATIONINVARIANCE"); checkGLError();
    GLint refineKeyPoints_location_descriptor1x = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor1x"); checkGLError();
    GLint refineKeyPoints_location_descriptor1y = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor1y"); checkGLError();
    GLint refineKeyPoints_location_descriptor2x = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor2x"); checkGLError();
    GLint refineKeyPoints_location_descriptor2y = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor2y"); checkGLError();
    GLint refineKeyPoints_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthKeyPoints"); checkGLError();
    GLint refineKeyPoints_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightKeyPoints"); checkGLError();
    GLint refineKeyPoints_location_keyPointsx = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsx"); checkGLError();
    GLint refineKeyPoints_location_keyPointsy = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsy"); checkGLError();

    glUseProgram(openGLProgram_refineKeyPoints); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptorShape, openGLDescriptorShapeTextureWidth); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptorShape, openGLDescriptorShapeTextureHeight); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptors, openGLDescriptorsTextureWidth[descriptorsId][0]); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptors, openGLDescriptorsTextureHeight[descriptorsId][0]); checkGLError();
    glUniform1i(refineKeyPoints_location_DESCRIPTORSIZE, DESCRIPTORSIZE); checkGLError();
    glUniform1i(refineKeyPoints_location_stepping, stepping ? 1 : 0); checkGLError();
    glUniform1i(refineKeyPoints_location_mipEnd, mipEnd); checkGLError();
    glUniform1i(refineKeyPoints_location_STEPCOUNT, STEPCOUNT); checkGLError();
    glUniform1f(refineKeyPoints_location_MIPSCALE, MIPSCALE); checkGLError();
    glUniform1f(refineKeyPoints_location_STEPSIZE, STEPSIZE); checkGLError();
    glUniform1f(refineKeyPoints_location_SCALEINVARIANCE, SCALEINVARIANCE); checkGLError();
    glUniform1f(refineKeyPoints_location_ROTATIONINVARIANCE, ROTATIONINVARIANCE); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthKeyPoints, openGLKeyPointsTextureWidth[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightKeyPoints, openGLKeyPointsTextureHeight[keyPointsId]); checkGLError();

    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLKeyPointsX[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_keyPointsx, 0); checkGLError();
    glActiveTexture(GL_TEXTURE1); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLKeyPointsY[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_keyPointsy, 1); checkGLError();
    glActiveTexture(GL_TEXTURE2); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[0]); checkGLError();
    glUniform1i(refineKeyPoints_location_descriptor1x, 2); checkGLError();
    glActiveTexture(GL_TEXTURE3); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[1]); checkGLError();
    glUniform1i(refineKeyPoints_location_descriptor1y, 3); checkGLError();
    glActiveTexture(GL_TEXTURE4); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[2]); checkGLError();
    glUniform1i(refineKeyPoints_location_descriptor2x, 4); checkGLError();
    glActiveTexture(GL_TEXTURE5); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLDescriptorShape[3]); checkGLError();
    glUniform1i(refineKeyPoints_location_descriptor2y, 5); checkGLError();
    int j = 6;
    for (int i = 0; i <= mipEnd; i++) {
        glActiveTexture(GL_TEXTURE0 + j); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptors[descriptorsId][i]); checkGLError();
        glUniform1i(refineKeyPoints_location_descriptors[i], j); checkGLError();
        j++;
    }
    for (int i = 0; i <= mipEnd; i++) {
        glActiveTexture(GL_TEXTURE0 + j); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId][i]); checkGLError();
        glUniform1i(refineKeyPoints_location_mipMaps[i], j); checkGLError();
        j++;
    }
    std::vector<GLint> mipMapWidth;
    std::vector<GLint> mipMapHeight;
    for (int i = 0; i <= mipEnd; i++) {
        mipMapWidth.push_back(openGLMipMapsTextureWidth[mipmapsId][i]);
        mipMapHeight.push_back(openGLMipMapsTextureHeight[mipmapsId][i]);
    }
    glUniform1iv(refineKeyPoints_location_mipMapWidth, (GLint)mipMapWidth.size(), &(mipMapWidth[0])); checkGLError();
    glUniform1iv(refineKeyPoints_location_mipMapHeight, (GLint)mipMapHeight.size(), &(mipMapHeight[0])); checkGLError();

    glBindFramebuffer(GL_FRAMEBUFFER, openGLKeyPointsRenderTarget[keyPointsId].first); checkGLError();
    const int w = (int)openGLKeyPointsRenderTargetWidth[keyPointsId];
    const int h = (int)openGLKeyPointsRenderTargetHeight[keyPointsId];
    glViewport(0, 0, w, h);
    fullScreenRect();
    float* data = new float[4 * w * h];
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, data); checkGLError();
    for (int i = 0; i < keyPointCount; i++) {
        const int x = i % w;
        const int y = i / w;
        const int texAdr = x + y * w;
        destKeyPoints[i].x = (data[4 * texAdr + 0] + data[4 * texAdr + 2]) * 0.5f;
        destKeyPoints[i].y = (data[4 * texAdr + 1] + data[4 * texAdr + 3]) * 0.5f;
        const float dx = data[4 * texAdr + 0] - data[4 * texAdr + 2];
        const float dy = data[4 * texAdr + 1] - data[4 * texAdr + 3];
        destErrors[i] = sqrtf(dx * dx + dy * dy);
    }
    delete[] data;
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}
