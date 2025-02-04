// Lucyte Created on: 03.02.2025 by Stefan Mader
#include <GL/glew.h>
#include "opengl_refinement.hpp"

GLFWwindow* window = NULL;
int window_width;
int window_height;

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
static GLuint openGLKeyPointsX[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLKeyPointsY[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureWidth[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureHeight[KEYPOINTIDCOUNT] = { 0 };
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
static GLuint openGLVertexShader_basic = 0;
static GLuint openGLProgram_sampleDescriptors = 0;
static GLuint openGLProgram_refineKeyPoints = 0;
#define SHARED_SHADER_FUNCTIONS ""\
"float t2d_nearest(sampler2D t, int x, int y, int width, int height)\n"\
"{\n"\
"   return textureLod(t,vec2((float(x)+0.25)/float(width),(float(y)+0.25)/float(height)),0.0).x;\n"\
"}\n"\
"vec2 descriptors1(int i) {\n"\
"   float x = t2d_nearest(descriptor1x, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"   float y = t2d_nearest(descriptor1y, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"   return vec2(x,y);\n"\
"}\n"\
"vec2 descriptors2(int i) {\n"\
"   float x = t2d_nearest(descriptor2x, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"   float y = t2d_nearest(descriptor2y, i % texWidthDescriptorShape, i / texWidthDescriptorShape, texWidthDescriptorShape, texHeightDescriptorShape);\n"\
"   return vec2(x,y);\n"\
"}\n"

const std::string sampleDescriptorProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
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
"       vec2 kp = vec2(t2d_nearest(keyPointsx,txx,txy,texWidthKeyPoints,texHeightKeyPoints),t2d_nearest(keyPointsy,txx,txy,texWidthKeyPoints,texHeightKeyPoints));\n"
"       vec2 sc = vec2(1.0/float(texWidthMipMap),1.0/float(texHeightMipMap));\n"
"       unsigned int b[4]; b[0]=unsigned int(0); b[1]=unsigned int(0); b[2]=unsigned int(0); b[3]=unsigned int(0);\n"
"       for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
"       {\n"
"           float l1 = textureLod(mipMap, (kp + descriptors1(i) * descriptorSize) * sc, 0.0).x;\n"
"           float l2 = textureLod(mipMap, (kp + descriptors2(i) * descriptorSize) * sc, 0.0).x;\n"
"           b[i>>5] += (l2 < l1) ? unsigned int(1<<(i & 31)) : unsigned int(0);\n"
"       }\n"
"       frag_color = uvec4(b[0],b[1],b[2],b[3]);\n"
"}\n"
"\n";
const std::string refineKeyPointsProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
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
"uniform sampler2D descriptors_0;\n"
"uniform sampler2D descriptors_1;\n"
"uniform sampler2D descriptors_2;\n"
"uniform sampler2D descriptors_3;\n"
"uniform sampler2D descriptors_4;\n"
"uniform sampler2D descriptors_5;\n"
"uniform sampler2D descriptors_6;\n"
"uniform sampler2D descriptors_7;\n"
"uniform sampler2D descriptors_8;\n"
"uniform sampler2D descriptors_9;\n"
"uniform sampler2D descriptors_10;\n"
"uniform sampler2D descriptors_11;\n"
"uniform sampler2D descriptors_12;\n"
"uniform sampler2D descriptors_13;\n"
"uniform sampler2D descriptors_14;\n"
"uniform sampler2D descriptors_15;\n"
"uniform sampler2D descriptors_16;\n"
"uniform sampler2D descriptors_17;\n"
"uniform sampler2D descriptors_18;\n"
"uniform sampler2D descriptors_19;\n"
"uniform sampler2D descriptors_20;\n"
"uniform sampler2D descriptors_21;\n"
"uniform sampler2D descriptors_22;\n"
"uniform sampler2D descriptors_23;\n"
"uniform sampler2D descriptors_24;\n"
"uniform sampler2D descriptors_25;\n"
"uniform sampler2D descriptors_26;\n"
"uniform sampler2D descriptors_27;\n"
"uniform sampler2D descriptors_28;\n"
"uniform sampler2D descriptors_29;\n"
"uniform sampler2D descriptors_30;\n"
"uniform sampler2D descriptors_31;\n"
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
"unsigned int dBits[4];\n"
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
"vec2 refineKeyPoint(sampler2D s, vec2 kpxy, int mipMap, uvec4 descriptor) {\n"
"   vec2 kp = kpxy * mipScale;\n"
"   float descriptorSize = mipScale * descriptorScale;\n"
"   float sina = sin(angle);\n"
"   float cosa = cos(angle);\n"
"   vec2 cosid = vec2(cosa, sina) * descriptorSize;\n"
"   vec2 nsicd = vec2(-sina, cosa) * descriptorSize;\n"
"   vec2 gdx = vec2(cosa,sina);\n"
"   vec2 gdy = vec2(-sina,cosa);\n"
"   vec2 xya = vec2(0.0,0.0);\n"
"   vec2 sc = vec2(1.0/float(mipMapWidth[mipMap]),1.0/float(mipMapHeight[mipMap]));\n"
"   dBits[0] = descriptor.x;"
"   dBits[1] = descriptor.y;"
"   dBits[2] = descriptor.z;"
"   dBits[3] = descriptor.w;"
"   for (int b = 0; b < DESCRIPTORSIZE; ++b) {\n"
"       vec2 dp1 = descriptors1(b);\n"
"       vec2 dp2 = descriptors2(b);\n"
"       vec2 d1 = kp + vec2(dot(cosid,dp1), dot(nsicd,dp1));\n"
"       vec2 d2 = kp + vec2(dot(cosid,dp2), dot(nsicd,dp2));\n"
"       float l1 = textureLod(s, d1 * sc, 0.0).x;\n"
"       float l2 = textureLod(s, d2 * sc, 0.0).x;\n"
"       int b2 = (l2 < l1 ? 1 : 0);\n"
"       if (b2 != ( int(dBits[b>>5]>>(b & 31)) & 1 )) {\n"
"           vec2 gr = vec2( textureLod(s, (d2 - gdx) * sc, 0.0).x - textureLod(s, (d2 + gdx) * sc, 0.0).x , textureLod(s, (d2 - gdy) * sc, 0.0).x - textureLod(s, (d2 + gdy) * sc, 0.0).x )\n"
"                    -vec2( textureLod(s, (d1 - gdx) * sc, 0.0).x - textureLod(s, (d1 + gdx) * sc, 0.0).x , textureLod(s, (d1 - gdy) * sc, 0.0).x - textureLod(s, (d1 + gdy) * sc, 0.0).x );\n"
"           if (b2 != 0) gr = -gr;\n"
"           if (stepping != 0) gr = vec2((gr.x == 0.0) ? 0.0 : (gr.x > 0.0) ? 1.0 : -1.0, (gr.y == 0.0) ? 0.0 : (gr.y > 0.0) ? 1.0 : -1.0);\n"
"           xya += gr * length(d2 - d1);\n"
"       }\n"
"   }\n"
"   float l = length(xya);\n"
"   if (l != 0.0) xya *= step / mipScale / l;\n"
"   vec2 r = kpxy + vec2(cosa * xya.x + sina * xya.y, -sina * xya.x + cosa * xya.y);\n"
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
"       vec2 kp = vec2(t2d_nearest(keyPointsx,txx,txy,texWidthKeyPoints,texHeightKeyPoints),t2d_nearest(keyPointsy,txx,txy,texWidthKeyPoints,texHeightKeyPoints));\n"
"       for(int i = mipEnd; i >= 0; i--) {\n"
"               uvec4 descriptorNeeded;\n"
"               switch(i) {\n"
"                   case  0:descriptorNeeded = uvec4(textureLod( descriptors_0, descriptorNrXY, 0.0)); break;"
"                   case  1:descriptorNeeded = uvec4(textureLod( descriptors_1, descriptorNrXY, 0.0)); break;"
"                   case  2:descriptorNeeded = uvec4(textureLod( descriptors_2, descriptorNrXY, 0.0)); break;"
"                   case  3:descriptorNeeded = uvec4(textureLod( descriptors_3, descriptorNrXY, 0.0)); break;"
"                   case  4:descriptorNeeded = uvec4(textureLod( descriptors_4, descriptorNrXY, 0.0)); break;"
"                   case  5:descriptorNeeded = uvec4(textureLod( descriptors_5, descriptorNrXY, 0.0)); break;"
"                   case  6:descriptorNeeded = uvec4(textureLod( descriptors_6, descriptorNrXY, 0.0)); break;"
"                   case  7:descriptorNeeded = uvec4(textureLod( descriptors_7, descriptorNrXY, 0.0)); break;"
"                   case  8:descriptorNeeded = uvec4(textureLod( descriptors_8, descriptorNrXY, 0.0)); break;"
"                   case  9:descriptorNeeded = uvec4(textureLod( descriptors_9, descriptorNrXY, 0.0)); break;"
"                   case 10:descriptorNeeded = uvec4(textureLod(descriptors_10, descriptorNrXY, 0.0)); break;"
"                   case 11:descriptorNeeded = uvec4(textureLod(descriptors_11, descriptorNrXY, 0.0)); break;"
"                   case 12:descriptorNeeded = uvec4(textureLod(descriptors_12, descriptorNrXY, 0.0)); break;"
"                   case 13:descriptorNeeded = uvec4(textureLod(descriptors_13, descriptorNrXY, 0.0)); break;"
"                   case 14:descriptorNeeded = uvec4(textureLod(descriptors_14, descriptorNrXY, 0.0)); break;"
"                   case 15:descriptorNeeded = uvec4(textureLod(descriptors_15, descriptorNrXY, 0.0)); break;"
"                   case 16:descriptorNeeded = uvec4(textureLod(descriptors_16, descriptorNrXY, 0.0)); break;"
"                   case 17:descriptorNeeded = uvec4(textureLod(descriptors_17, descriptorNrXY, 0.0)); break;"
"                   case 18:descriptorNeeded = uvec4(textureLod(descriptors_18, descriptorNrXY, 0.0)); break;"
"                   case 19:descriptorNeeded = uvec4(textureLod(descriptors_19, descriptorNrXY, 0.0)); break;"
"                   case 20:descriptorNeeded = uvec4(textureLod(descriptors_20, descriptorNrXY, 0.0)); break;"
"                   case 21:descriptorNeeded = uvec4(textureLod(descriptors_21, descriptorNrXY, 0.0)); break;"
"                   case 22:descriptorNeeded = uvec4(textureLod(descriptors_22, descriptorNrXY, 0.0)); break;"
"                   case 23:descriptorNeeded = uvec4(textureLod(descriptors_23, descriptorNrXY, 0.0)); break;"
"                   case 24:descriptorNeeded = uvec4(textureLod(descriptors_24, descriptorNrXY, 0.0)); break;"
"                   case 25:descriptorNeeded = uvec4(textureLod(descriptors_25, descriptorNrXY, 0.0)); break;"
"                   case 26:descriptorNeeded = uvec4(textureLod(descriptors_26, descriptorNrXY, 0.0)); break;"
"                   case 27:descriptorNeeded = uvec4(textureLod(descriptors_27, descriptorNrXY, 0.0)); break;"
"                   case 28:descriptorNeeded = uvec4(textureLod(descriptors_28, descriptorNrXY, 0.0)); break;"
"                   case 29:descriptorNeeded = uvec4(textureLod(descriptors_29, descriptorNrXY, 0.0)); break;"
"                   case 30:descriptorNeeded = uvec4(textureLod(descriptors_30, descriptorNrXY, 0.0)); break;"
"                   case 31:descriptorNeeded = uvec4(textureLod(descriptors_31, descriptorNrXY, 0.0)); break;"
"               }\n"
"           for(int k = 0; k < STEPCOUNT; k++) {\n"
"               mipScale = pow(MIPSCALE, float(i));\n"
"               descriptorScale = 1.0 / mipScale;\n"
"               step = STEPSIZE * descriptorScale;\n"
"               descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.0 - SCALEINVARIANCE;\n"
"               angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.0 - ROTATIONINVARIANCE) / 360.0 * 2.0 * 3.1415927f;\n"
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
"   }\n"
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

void checkGLError() {
    int error = glGetError();
    if (error != GL_NO_ERROR) {
        printf("OpenGL error(%d): %s\n", error, glewGetErrorString(error));
        int debug = 1;
    }
}

void checkShader(GLuint shader) {
    GLint isCompiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &isCompiled); checkGLError();
    if (isCompiled == GL_FALSE) {
        GLint maxLength = 0; glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength); checkGLError();
        std::vector<char> errorLog(maxLength);
        glGetShaderInfoLog(shader, maxLength, &maxLength, &errorLog[0]); checkGLError();
        printf("%s", &(errorLog[0]));
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
        printf("%s", &(errorLog[0]));
        glDeleteProgram(program); checkGLError();
        exit(0);
    }
}

GLuint pixelShader(const char *shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint vertexShader(const char* shaderProgram, const unsigned int size) {
    int length[] = { size };
    GLuint shader = glCreateShader(GL_VERTEX_SHADER); checkGLError();
    glShaderSource(shader, 1, &shaderProgram, length); checkGLError();
    glCompileShader(shader); checkGLError();
    checkShader(shader);
    return shader;
}

GLuint program(const GLuint vertexShader, const GLuint pixelShader) {
    GLuint program = glCreateProgram(); checkGLError();
    glAttachShader(program, vertexShader); checkGLError();
    glAttachShader(program, pixelShader); checkGLError();
    glLinkProgram(program); checkGLError();
    checkProgram(program);
    return program;
}

void createUint32RenderTarget(std::pair<GLuint, GLuint> &dest, int width, int height, bool nearest) {
    glDeleteFramebuffers(1, &dest.first); checkGLError();
    glDeleteTextures(1, &dest.second);
    GLuint framebufferId = 0;
    glGenFramebuffers(1, &framebufferId); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferId); checkGLError();
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, width, height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 0); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0); checkGLError();
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
    dest = { framebufferId, renderedTexture };
}

void createFloatRenderTarget(std::pair<GLuint, GLuint>& dest, int width, int height, bool nearest) {
    glDeleteFramebuffers(1, &dest.first); checkGLError();
    glDeleteTextures(1, &dest.second);
    GLuint framebufferId = 0;
    glGenFramebuffers(1, &framebufferId); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferId); checkGLError();
    GLuint renderedTexture;
    glGenTextures(1, &renderedTexture);
    glBindTexture(GL_TEXTURE_2D, renderedTexture); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0); checkGLError();
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBuffers); checkGLError();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
    dest = { framebufferId, renderedTexture };
}

void glfwErrorCallback(int error_code, const char* description) {

}

void initOpenGL() {
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
    openGLFragmentShader_sampleDescriptors = pixelShader(sampleDescriptorProgram.c_str(), sampleDescriptorProgram.length());
    openGLFragmentShader_refineKeyPoints = pixelShader(refineKeyPointsProgram.c_str(), refineKeyPointsProgram.length());
    openGLVertexShader_basic = vertexShader(basicProgram.c_str(), basicProgram.length());
    openGLProgram_sampleDescriptors = program(openGLVertexShader_basic, openGLFragmentShader_sampleDescriptors);
    openGLProgram_refineKeyPoints = program(openGLVertexShader_basic, openGLFragmentShader_refineKeyPoints);
}

void upload2DFloatTexture(const GLuint& tex, const float* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload2DUnsignedCharTexture(const GLuint& tex, const float* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload1DFloatTexture(const GLuint& tex, const float* data, const unsigned int count, int& width, int& height, const bool nearest) {
    width = count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
        float* data2 = new float[width * height];
        memcpy(data2, data, count * sizeof(float));
        upload2DFloatTexture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2DFloatTexture(tex, data, width, height, nearest);
}

void upload2Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int width, const int height, const bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_INT, data); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void upload1Duint32Texture(const GLuint& tex, const unsigned int* data, const unsigned int count, int &width, int &height, const bool nearest) {
    width = count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (count + maxTextureSize - 1) / maxTextureSize;
        unsigned int* data2 = new unsigned int[width * height];
        memcpy(data2, data, count * sizeof(unsigned int));
        upload2Duint32Texture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2Duint32Texture(tex, data, width, height, nearest);
}

void uploadDescriptorShape_openGL() {
    glDeleteTextures(4, openGLDescriptorShape);
    glGenTextures(4, openGLDescriptorShape);
    upload1DFloatTexture(openGLDescriptorShape[0], descriptorsX1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[1], descriptorsY1, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[2], descriptorsX2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
    upload1DFloatTexture(openGLDescriptorShape[3], descriptorsY2, DESCRIPTORSIZE, openGLDescriptorShapeTextureWidth, openGLDescriptorShapeTextureHeight, true);
}

void uploadMipMaps_openGL(const int mipmapsId, const std::vector<cv::Mat>& mipMaps) {
    glDeleteTextures(MAXMIPMAPS, openGLMipMaps[mipmapsId]);
    glGenTextures(mipMaps.size(), openGLMipMaps[mipmapsId]);
    for (int i = 0; i < mipMaps.size(); i++) {
        openGLMipMapsTextureWidth[mipmapsId][i] = mipMaps[i].cols;
        openGLMipMapsTextureHeight[mipmapsId][i] = mipMaps[i].rows;
        upload2DUnsignedCharTexture(openGLMipMaps[mipmapsId][i], (float*)(mipMaps[i].data), mipMaps[i].cols, mipMaps[i].rows, false);
    }
}

void uploadKeyPoints_openGL(const int keyPointsId, const std::vector<KeyPoint>& keyPoints) {
    std::vector<float> kpx; kpx.resize(keyPoints.size());
    std::vector<float> kpy; kpy.resize(keyPoints.size());
    for (int i = int(keyPoints.size()) - 1; i >= 0; i--) {
        kpx[i] = keyPoints[i].x;
        kpy[i] = keyPoints[i].y;
    }
    glDeleteTextures(1, &(openGLKeyPointsX[keyPointsId]));
    glDeleteTextures(1, &(openGLKeyPointsY[keyPointsId]));
    glGenTextures(1, &(openGLKeyPointsX[keyPointsId]));
    glGenTextures(1, &(openGLKeyPointsY[keyPointsId]));
    
    upload1DFloatTexture(openGLKeyPointsX[keyPointsId], &(kpx[0]), keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    upload1DFloatTexture(openGLKeyPointsY[keyPointsId], &(kpy[0]), keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
}

void uploadDescriptors_openGL(const int descriptorsId, const int mipMap, const std::vector<std::vector<Descriptor>>& sourceMips) {
    const std::vector<Descriptor>& descriptors = sourceMips[mipMap];
    std::vector<unsigned int> bits; bits.resize(Descriptor::uint32count * descriptors.size());
    for (int i = int(descriptors.size()) - 1; i >= 0; i--) {
        for (int j = 0; j < Descriptor::uint32count; j++) {
            bits[j + i * Descriptor::uint32count] = descriptors[i].bits[j];
        }
    }
    glDeleteTextures(1, &(openGLDescriptors[descriptorsId][mipMap]));
    glGenTextures(1, &(openGLDescriptors[descriptorsId][mipMap]));
    upload1Duint32Texture(openGLDescriptors[descriptorsId][mipMap], &(bits[0]), bits.size(), openGLDescriptorsTextureWidth[descriptorsId][mipMap], openGLDescriptorsTextureHeight[descriptorsId][mipMap],true);
}

void sampleDescriptors_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int mipMap, std::vector<std::vector<Descriptor>>& destMips, const float descriptorScale, const int width, const int height, const float mipScale) {

    std::vector<Descriptor> &descriptors = destMips[mipMap];

    GLuint sampleDescriptors_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthDescriptorShape"); checkGLError();
    GLuint sampleDescriptors_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightDescriptorShape"); checkGLError();
    GLuint sampleDescriptors_location_texWidthMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthMipMap"); checkGLError();
    GLuint sampleDescriptors_location_texHeightMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightMipMap"); checkGLError();
    GLuint sampleDescriptors_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthKeyPoints"); checkGLError();
    GLuint sampleDescriptors_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightKeyPoints"); checkGLError();
    GLuint sampleDescriptors_location_DESCRIPTORSIZE = glGetUniformLocation(openGLProgram_sampleDescriptors, "DESCRIPTORSIZE"); checkGLError();
    GLuint sampleDescriptors_location_descriptorScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptorScale"); checkGLError();
    GLuint sampleDescriptors_location_mipScale = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipScale"); checkGLError();
    GLuint sampleDescriptors_location_descriptor1x = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor1x"); checkGLError();
    GLuint sampleDescriptors_location_descriptor1y = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor1y"); checkGLError();
    GLuint sampleDescriptors_location_descriptor2x = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor2x"); checkGLError();
    GLuint sampleDescriptors_location_descriptor2y = glGetUniformLocation(openGLProgram_sampleDescriptors, "descriptor2y"); checkGLError();
    GLuint sampleDescriptors_location_mipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipMap"); checkGLError();
    GLuint sampleDescriptors_location_keyPointsx = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsx"); checkGLError();
    GLuint sampleDescriptors_location_keyPointsy = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsy"); checkGLError();

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

    if (openGLKeyPointsTextureWidth[keyPointsId] != openGLDescriptorRenderTargetWidth[keyPointsId][mipMap] || openGLKeyPointsTextureHeight[keyPointsId] != openGLDescriptorRenderTargetHeight[keyPointsId][mipMap]) {
        openGLDescriptorRenderTargetWidth[keyPointsId][mipMap] = openGLKeyPointsTextureWidth[keyPointsId];
        openGLDescriptorRenderTargetHeight[keyPointsId][mipMap] = openGLKeyPointsTextureHeight[keyPointsId];
        createUint32RenderTarget(openGLDescriptorRenderTarget[keyPointsId][mipMap], openGLDescriptorRenderTargetWidth[keyPointsId][mipMap], openGLDescriptorRenderTargetHeight[keyPointsId][mipMap], true);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, openGLDescriptorRenderTarget[keyPointsId][mipMap].first); checkGLError();
    const int w = openGLDescriptorRenderTargetWidth[keyPointsId][mipMap];
    const int h = openGLDescriptorRenderTargetHeight[keyPointsId][mipMap];
    glViewport(0, 0, w, h);
    glRects(-1, -1, 1, 1);
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
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void refineKeyPoints_openGL(const int keyPointsId, const int descriptorsId, const int mipmapsId, const int keyPointCount, const int mipEnd, const int STEPCOUNT, const bool stepping, const float MIPSCALE, const float STEPSIZE, const float SCALEINVARIANCE, const float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors) {
    // OpenGL is not working, yet.
    GLuint refineKeyPoints_location_mipMaps[32]; for (int i = 0; i < 32; i++) { refineKeyPoints_location_mipMaps[i] = glGetUniformLocation(openGLProgram_refineKeyPoints, ("mipMaps_" + std::to_string(i)).c_str()); checkGLError(); }
    GLuint refineKeyPoints_location_descriptors[32]; for (int i = 0; i < 32; i++) { refineKeyPoints_location_descriptors[i] = glGetUniformLocation(openGLProgram_refineKeyPoints, ("descriptors_" + std::to_string(i)).c_str()); checkGLError(); }
    GLuint refineKeyPoints_location_mipMapWidth = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapWidth"); checkGLError();
    GLuint refineKeyPoints_location_mipMapHeight = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapHeight"); checkGLError();
    GLuint refineKeyPoints_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptorShape"); checkGLError();
    GLuint refineKeyPoints_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptorShape"); checkGLError();
    GLuint refineKeyPoints_location_texWidthDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptors"); checkGLError();
    GLuint refineKeyPoints_location_texHeightDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptors"); checkGLError();
    GLuint refineKeyPoints_location_DESCRIPTORSIZE = glGetUniformLocation(openGLProgram_refineKeyPoints, "DESCRIPTORSIZE"); checkGLError();
    GLuint refineKeyPoints_location_stepping = glGetUniformLocation(openGLProgram_refineKeyPoints, "stepping"); checkGLError();
    GLuint refineKeyPoints_location_mipEnd = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipEnd"); checkGLError();
    GLuint refineKeyPoints_location_STEPCOUNT = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPCOUNT"); checkGLError();
    GLuint refineKeyPoints_location_MIPSCALE = glGetUniformLocation(openGLProgram_refineKeyPoints, "MIPSCALE"); checkGLError();
    GLuint refineKeyPoints_location_STEPSIZE = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPSIZE"); checkGLError();
    GLuint refineKeyPoints_location_SCALEINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "SCALEINVARIANCE"); checkGLError();
    GLuint refineKeyPoints_location_ROTATIONINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "ROTATIONINVARIANCE"); checkGLError();
    GLuint refineKeyPoints_location_descriptor1x = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor1x"); checkGLError();
    GLuint refineKeyPoints_location_descriptor1y = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor1y"); checkGLError();
    GLuint refineKeyPoints_location_descriptor2x = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor2x"); checkGLError();
    GLuint refineKeyPoints_location_descriptor2y = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptor2y"); checkGLError();
    GLuint refineKeyPoints_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthKeyPoints"); checkGLError();
    GLuint refineKeyPoints_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightKeyPoints"); checkGLError();
    GLuint refineKeyPoints_location_keyPointsx = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsx"); checkGLError();
    GLuint refineKeyPoints_location_keyPointsy = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsy"); checkGLError();

    glUseProgram(openGLProgram_refineKeyPoints); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptorShape, openGLDescriptorShapeTextureWidth); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptorShape, openGLDescriptorShapeTextureHeight); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptors, openGLDescriptorRenderTargetWidth[keyPointsId][0]); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptors, openGLDescriptorRenderTargetHeight[keyPointsId][0]); checkGLError();
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
        glActiveTexture(GL_TEXTURE0+j); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLDescriptorRenderTarget[keyPointsId][i].second); checkGLError();
        glUniform1i(refineKeyPoints_location_descriptors[i], j);
        j++;
    }
    for (int i = 0; i <= mipEnd; i++) {
        glActiveTexture(GL_TEXTURE0 + j); checkGLError();
        glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId][i]); checkGLError();
        glUniform1i(refineKeyPoints_location_mipMaps[i], j);
        j++;
    }
    std::vector<GLint> mipMapWidth;
    std::vector<GLint> mipMapHeight;
    for (int i = 0; i <= mipEnd; i++) {
        mipMapWidth.push_back(openGLMipMapsTextureWidth[mipmapsId][i]);
        mipMapHeight.push_back(openGLMipMapsTextureHeight[mipmapsId][i]);
    }
    glUniform1iv(refineKeyPoints_location_mipMapWidth, mipMapWidth.size(), &(mipMapWidth[0])); checkGLError();
    glUniform1iv(refineKeyPoints_location_mipMapHeight, mipMapHeight.size(), &(mipMapHeight[0])); checkGLError();

    if (openGLKeyPointsTextureWidth[keyPointsId] != openGLKeyPointsRenderTargetWidth[keyPointsId]|| openGLKeyPointsTextureHeight[keyPointsId] != openGLKeyPointsRenderTargetHeight[keyPointsId]) {
        openGLKeyPointsRenderTargetWidth[keyPointsId] = openGLKeyPointsTextureWidth[keyPointsId];
        openGLKeyPointsRenderTargetHeight[keyPointsId] = openGLKeyPointsTextureHeight[keyPointsId];
        createFloatRenderTarget(openGLKeyPointsRenderTarget[keyPointsId], openGLKeyPointsRenderTargetWidth[keyPointsId], openGLKeyPointsRenderTargetHeight[keyPointsId], true);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, openGLKeyPointsRenderTarget[keyPointsId].first); checkGLError();
    const int w = openGLKeyPointsRenderTargetWidth[keyPointsId];
    const int h = openGLKeyPointsRenderTargetHeight[keyPointsId];
    glViewport(0, 0, w, h);
    glRects(-1, -1, 1, 1);
    float* data = new float[4 * w * h];
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, data); checkGLError();
    for (int i = 0; i < keyPointCount; i++) {
        int x = i % w;
        int y = i / w;
        int texAdr = x + y * w;
        destKeyPoints[i].x = (data[4 * texAdr + 0] + data[4 * texAdr + 2]) * 0.5;
        destKeyPoints[i].y = (data[4 * texAdr + 1] + data[4 * texAdr + 3]) * 0.5;
        const float dx = (data[4 * texAdr + 0] - data[4 * texAdr + 2]);
        const float dy = (data[4 * texAdr + 1] - data[4 * texAdr + 3]);
        destErrors[i] = sqrt(dx * dx + dy * dy);
    }
    delete[] data;
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}
