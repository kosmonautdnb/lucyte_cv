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

static const int MIPMAPIDCOUNT = 8;
static const int KEYPOINTIDCOUNT = 2;
static const int DESCRIPTORIDCOUNT = 2;
static const int maxTextureSize = 1024;

static GLuint openGLDescriptorShape[4] = { 0 };
static int openGLDescriptorShapeTextureWidth = 0;
static int openGLDescriptorShapeTextureHeight = 0;
static GLuint openGLMipMaps[MIPMAPIDCOUNT] = { 0 };
static int openGLMipMapsTextureWidth[MIPMAPIDCOUNT] = { 0 };
static int openGLMipMapsTextureHeight[MIPMAPIDCOUNT] = { 0 };
static GLuint openGLKeyPoints[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureWidth[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointsTextureHeight[KEYPOINTIDCOUNT] = { 0 };
static int openGLKeyPointCount[KEYPOINTIDCOUNT] = { 0 };
static std::pair<GLuint, GLuint> openGLDescriptorRenderTarget[DESCRIPTORIDCOUNT] = { {0,0} };
static GLuint openGLDescriptorRenderTargetWidth[DESCRIPTORIDCOUNT] = { 0 };
static GLuint openGLDescriptorRenderTargetHeight[DESCRIPTORIDCOUNT] = { 0 };
static std::pair<GLuint, GLuint> openGLKeyPointsRenderTarget[KEYPOINTIDCOUNT] = { {0,0} };
static GLuint openGLKeyPointsRenderTargetWidth[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLKeyPointsRenderTargetHeight[KEYPOINTIDCOUNT] = { 0 };
static GLuint openGLDescriptors[DESCRIPTORIDCOUNT] = { 0 };
static int openGLDescriptorsTextureWidth[DESCRIPTORIDCOUNT] = { 0 };
static int openGLDescriptorsTextureHeight[DESCRIPTORIDCOUNT] = { 0 };
static GLuint openGLFragmentShader_sampleDescriptors = 0;
static GLuint openGLFragmentShader_refineKeyPoints = 0;
static GLuint openGLFragmentShader_displayMipMap = 0;
static GLuint openGLVertexShader_basic = 0;
static GLuint openGLProgram_sampleDescriptors = 0;
static GLuint openGLProgram_refineKeyPoints = 0;
static GLuint openGLProgram_displayMipMap = 0;
#define SHARED_SHADER_FUNCTIONS ""\
"vec2 t2d_nearest(sampler2D t, int x, int y, int width, int height)\n"\
"{\n"\
"   vec2 k = vec2((float(x)+0.25)/float(width),(float(y)+0.25)/float(height));\n"\
"   return textureLod(t,k,0.0).xy;\n"\
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
                                            "uniform int texWidthKeyPoints2;\n"
                                            "uniform int texHeightKeyPoints2;\n"
                                            "uniform int mipLevels;\n"
                                            "uniform int keyPointCount;\n"
                                            "uniform float MIPSCALE;\n"
                                            "uniform float DESCRIPTORSCALE2;\n"
                                            "uniform sampler2D mipMap;\n"
                                            "uniform sampler2D keyPoints;\n"
                                            SHARED_SHADER_FUNCTIONS
                                            "void main()\n"
                                            "{\n"
                                            "       uint b[4]; b[0]=uint(0); b[1]=uint(0); b[2]=uint(0); b[3]=uint(0);\n"
                                            "       vec2 sc = vec2(1.0/float(texWidthMipMap),1.0/float(texHeightMipMap));\n"
                                            "       int txx = int(floor(p.x+0.25));\n"
                                            "       int txy = int(floor(p.y+0.25));\n"
                                            "       int t = txx + txy * texWidthKeyPoints;\n"
                                            "       if (t < keyPointCount*mipLevels) {"
                                            "           int keyPointId = t % keyPointCount;\n"
                                            "           int mipLevel = t / keyPointCount;\n"
                                            "           vec2 kp = t2d_nearest(keyPoints,keyPointId % texWidthKeyPoints2,keyPointId / texWidthKeyPoints2,texWidthKeyPoints2,texHeightKeyPoints2);\n"
                                            "           float mipScale = pow(MIPSCALE,float(mipLevel));\n"
                                            "           float descriptorScale = DESCRIPTORSCALE2/mipScale;\n"
                                            "           float m = float(mipLevel);\n"
                                            "           for(int i = 0; i < DESCRIPTORSIZE; i++)\n"
                                            "           {\n"
                                            "               float l1 = textureLod(mipMap, (kp + descriptors12[i].xy * descriptorScale) * sc, m).x;\n"
                                            "               float l2 = textureLod(mipMap, (kp + descriptors12[i].zw * descriptorScale) * sc, m).x;\n"
                                            "               b[i>>5] += (l2 < l1) ? uint(1)<<uint(i & 31) : uint(0);\n"
                                            "           }\n"
                                            "       }\n"
                                            "       frag_color = uvec4(b[0],b[1],b[2],b[3]);\n"
                                            "}\n"
                                            "\n";
const std::string refineKeyPointsProgram = ""\
"in vec2 p;\n"
                                           "out vec4 frag_color;\n"
                                           "uniform sampler2D mipMap;\n"
                                           "uniform int mipMapWidth;\n"
                                           "uniform int mipMapHeight;\n"
                                           "uniform usampler2D descriptors;\n"
                                           "uniform int texWidthDescriptorShape;\n"
                                           "uniform int texHeightDescriptorShape;\n"
                                           "uniform int texWidthDescriptors;\n"
                                           "uniform int texHeightDescriptors;\n"
                                           "uniform int mipLevels;\n"
                                           "uniform int STEPCOUNT;\n"
                                           "uniform float MIPSCALE;\n"
                                           "uniform float STEPSIZE;\n"
                                           "uniform float SCALEINVARIANCE;\n"
                                           "uniform float ROTATIONINVARIANCE;\n"
                                           "uniform int texWidthKeyPoints;\n"
                                           "uniform int texHeightKeyPoints;\n"
                                           "uniform sampler2D keyPoints;\n"
                                           "uniform int keyPointCount;\n"
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
                                           "vec2 refineKeyPoint(sampler2D s, vec2 kp, float m, uvec4 descriptor) {\n"
                                           "   vec2 sc = vec2(1.0/float(mipMapWidth),1.0/float(mipMapHeight));\n"
                                           "   vec2 kp2 = kp * sc;\n"
                                           "   float sina = sin(angle);\n"
                                           "   float cosa = cos(angle);\n"
                                           "   vec2 cosid = vec2(cosa, sina) * descriptorScale * sc.x;\n"
                                           "   vec2 nsicd = vec2(-sina, cosa) * descriptorScale * sc.y;\n"
                                           "   vec2 gdx = vec2(sc.x / mipScale,0.0);\n"
                                           "   vec2 gdy = vec2(0.0, sc.y / mipScale);\n"
                                           "   vec2 xya = vec2(0.0,0.0);\n"
                                           "   uint dBits[4];\n"
                                           "   dBits[0] = descriptor.x;\n"
                                           "   dBits[1] = descriptor.y;\n"
                                           "   dBits[2] = descriptor.z;\n"
                                           "   dBits[3] = descriptor.w;\n"
                                           "   uint bk;\n"
                                           "   for (int b = 0; b < DESCRIPTORSIZE; b++) {\n"
                                           "       if ((b & 31)==0) bk = dBits[b>>5];\n"                                       
                                           "       vec4 dp = descriptors12[b];\n"
                                           "       vec2 d1 = kp2 + vec2(dot(cosid,dp.xy), dot(nsicd,dp.xy));\n"
                                           "       vec2 d2 = kp2 + vec2(dot(cosid,dp.zw), dot(nsicd,dp.zw));\n"
                                           "       int b2 = textureLod(s, d2, m).x < textureLod(s, d1, m).x ? 1 : 0;\n"
                                           "       if (b2 != int(bk & uint(1)) ) {\n"
                                           "           vec2 gr = vec2( textureLod(s, d2 - gdx, m).x - textureLod(s, d2 + gdx, m).x , textureLod(s, d2 - gdy, m).x - textureLod(s, d2 + gdy, m).x )\n"
                                           "                    -vec2( textureLod(s, d1 - gdx, m).x - textureLod(s, d1 + gdx, m).x , textureLod(s, d1 - gdy, m).x - textureLod(s, d1 + gdy, m).x );\n"
                                           "           if (b2 != 0) gr = -gr;\n"
                                           "           xya += gr * descriptorsw[b];\n"
                                           "       }\n"
                                           "       bk >>= 1;\n"
                                           "   }\n"
                                           "   float l = length(xya);\n"
                                           "   if (l != 0.0) xya *= step / mipScale / l;\n"
                                           "   return kp + xya;\n"
                                           "}\n"
                                           "void main()\n"
                                           "{\n"
                                           "   int txx = int(floor(p.x+0.25));\n"
                                           "   int txy = int(floor(p.y+0.25));\n"
                                           "   int descriptorNr = txx + txy * texWidthKeyPoints;\n"
                                           "   for(int v = 0; v < 2; v++) {\n"
                                           "       vec2 kp = t2d_nearest(keyPoints,txx,txy,texWidthKeyPoints,texHeightKeyPoints);\n"
                                           "       for(int i = mipLevels-1; i >= 0; i--) {\n"
                                           "           int descriptorNr2 = descriptorNr + i * keyPointCount;\n"
                                           "           int descriptorNrX = descriptorNr2 % texWidthDescriptors;\n"
                                           "           int descriptorNrY = descriptorNr2 / texWidthDescriptors;\n"
                                           "           vec2 descriptorNrXY = vec2(float(descriptorNrX) + 0.25,float(descriptorNrY) + 0.25) / vec2(float(texWidthDescriptors),float(texHeightDescriptors));\n"
                                           "           uvec4 descriptorNeeded = textureLod( descriptors, descriptorNrXY, 0.0);\n"
                                           "           for(int k = 0; k < STEPCOUNT; k++) {\n"
                                           "               mipScale = pow(MIPSCALE, float(i));\n"
                                           "               descriptorScale = 1.0 / mipScale;\n"
                                           "               step = STEPSIZE * descriptorScale;\n"
                                           "               descriptorScale *= 1.0 + randomLike(k * 11 + i * 9 + v * 11 + 31239) * SCALEINVARIANCE * 2.0 - SCALEINVARIANCE;\n"
                                           "               angle = (randomLike(k * 13 + i * 7 + v * 9 + 1379) * ROTATIONINVARIANCE * 2.0 - ROTATIONINVARIANCE) / 360.0 * 2.0 * 3.1415927;\n"
                                           "               kp = refineKeyPoint( mipMap, kp, float(i), descriptorNeeded);\n"
                                           "           }\n"
                                           "       }\n"
                                           "       switch(v) {\n"
                                           "           case 0: frag_color.xy = kp; break;\n"
                                           "           case 1: frag_color.zw = kp; break;\n"
                                           "       }\n"
                                           "   }\n"
                                           "}\n"
                                           "\n";
const std::string displayMipMapProgram = "#version 300 es\nprecision highp float;precision highp int;\n"
                                         "in vec2 p;\n"
                                         "out vec4 frag_color;\n"
                                         "uniform sampler2D mipMap;\n"
                                         "uniform int mipLevel;\n"
                                         "uniform int mipMapWidth,mipMapHeight;\n"
                                         "void main()\n"
                                         "{\n"
                                         "   vec2 sc = vec2(1.0/float(mipMapWidth),1.0/float(mipMapHeight));\n"
                                         "   vec2 xy = p * sc;\n"
                                         "   xy.y = 1.0 - xy.y;\n"
                                         "   float mip = textureLod( mipMap, xy, float(mipLevel) ).x;\n"
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

    std::string descriptors12 = "vec4 descriptors12[" + std::to_string(DESCRIPTORSIZE) + "] = vec4 [](";
    std::string descriptorsw = "float descriptorsw[" + std::to_string(DESCRIPTORSIZE) + "] = float [](";
    for (int i = 0; i < DESCRIPTORSIZE; i++) {
        if (i != 0) {
            descriptors12 += ",";
            descriptorsw += ",";
        }
        descriptors12 += "vec4(" + std::to_string(descriptorsX1[i]) + "," + std::to_string(descriptorsY1[i]) + "," + std::to_string(descriptorsX2[i]) + "," + std::to_string(descriptorsY2[i]) + ")";
        float dx = descriptorsX1[i] - descriptorsX2[i];
        float dy = descriptorsY1[i] - descriptorsY2[i];
        float d = sqrtf(dx*dx+dy*dy);
        descriptorsw += std::to_string(d);
    }
    descriptors12 += ");\n";
    descriptorsw += ");\n";
    std::string descriptorCount = "int DESCRIPTORSIZE = " + std::to_string(DESCRIPTORSIZE) + ";\n";

    std::string refineKeys = "#version 300 es\nprecision highp float;precision highp int;\n" + descriptors12 + descriptorsw + descriptorCount + refineKeyPointsProgram;
    std::string sampleDescriptors = "#version 300 es\nprecision highp float;precision highp int;\n" + descriptors12 + descriptorCount + sampleDescriptorProgram;

    openGLFragmentShader_sampleDescriptors = pixelShader(sampleDescriptors.c_str(), (unsigned int)sampleDescriptors.length());
    openGLFragmentShader_refineKeyPoints = pixelShader(refineKeys.c_str(), (unsigned int)refineKeys.length());
    openGLFragmentShader_displayMipMap = pixelShader(displayMipMapProgram.c_str(), (unsigned int)displayMipMapProgram.length());
    openGLVertexShader_basic = vertexShader(basicProgram.c_str(), (unsigned int)basicProgram.length());
    openGLProgram_sampleDescriptors = program(openGLVertexShader_basic, openGLFragmentShader_sampleDescriptors);
    openGLProgram_refineKeyPoints = program(openGLVertexShader_basic, openGLFragmentShader_refineKeyPoints);
    openGLProgram_displayMipMap = program(openGLVertexShader_basic, openGLFragmentShader_displayMipMap);
    glGenTextures(MIPMAPIDCOUNT, openGLMipMaps);
    glGenTextures(4, openGLDescriptorShape);
    for (int i = 0; i < KEYPOINTIDCOUNT; i++) {
        glGenTextures(1, &(openGLKeyPoints[i]));
    }
    for (int i = 0; i < DESCRIPTORIDCOUNT; i++) {
        glGenTextures(1, &(openGLDescriptors[i]));
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

void upload2DKeyPointsTexture(const GLuint& tex, const KeyPoint* data, unsigned int width, unsigned int height, bool nearest) {
    glBindTexture(GL_TEXTURE_2D, tex); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, nearest ? GL_NEAREST : GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, (int)width, (int)height, 0, GL_RG, GL_FLOAT, data); checkGLError();
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

void upload1DKeyPointsTexture(const GLuint& tex, const KeyPoint* data, unsigned int count, int& width, int& height, bool nearest) {
    width = (int)count;
    height = 1;
    if (width > maxTextureSize) {
        width = maxTextureSize;
        height = (int)(count + maxTextureSize - 1) / maxTextureSize;
        KeyPoint* data2 = new KeyPoint[width * height];
        memcpy(data2, data, count * sizeof(float));
        upload2DKeyPointsTexture(tex, data2, width, height, nearest);
        delete[] data2;
        return;
    }
    upload2DKeyPointsTexture(tex, data, width, height, nearest);
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

void uploadMipMaps_openGL(int mipmapsId, const MipMap& baseLevel) {
    openGLMipMapsTextureWidth[mipmapsId] = baseLevel.width;
    openGLMipMapsTextureHeight[mipmapsId] = baseLevel.height;
    glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId]); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); checkGLError();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); checkGLError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, (int)baseLevel.width, (int)baseLevel.height, 0, GL_RED, GL_UNSIGNED_BYTE, baseLevel.data); checkGLError();
    glGenerateMipmap(GL_TEXTURE_2D); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

static std::vector<float> kpx;
static std::vector<float> kpy;

void uploadKeyPoints_openGL(int keyPointsId, const std::vector<KeyPoint>& keyPoints) {
    upload1DKeyPointsTexture(openGLKeyPoints[keyPointsId], &(keyPoints[0]), (unsigned int)keyPoints.size(), openGLKeyPointsTextureWidth[keyPointsId], openGLKeyPointsTextureHeight[keyPointsId], true);
    openGLKeyPointCount[keyPointsId] = (int)keyPoints.size();
}

void uploadDescriptors_openGL(int descriptorsId, int mipLevels, const std::vector<std::vector<Descriptor>>& sourceMips) {
    const int keyPointCount = (int)sourceMips[0].size();
    const int allDescriptorCount = keyPointCount * mipLevels;

    std::vector<unsigned int> bits;
    bits.resize(4 * allDescriptorCount);
    int v = 0;
    for (int i2 = 0; i2 < mipLevels; i2++) {
        for (int i = 0; i < keyPointCount; i++) {
            for (int j = 0; j < Descriptor::uint32count; j++) {
                bits[v+j] = sourceMips[i2][i].bits[j];
            }
            v += 4;
        }
    }
    upload1Duint324Texture(openGLDescriptors[descriptorsId], &(bits[0]), allDescriptorCount, openGLDescriptorsTextureWidth[descriptorsId], openGLDescriptorsTextureHeight[descriptorsId], true);
}

void fullScreenRect() {
    float points[] = {
            1.0f,  1.0f,  0.0f,
            1.0f, -1.0f,  0.0f,
            -1.0f, -1.0f,  0.0f,
            1.0f,  1.0f,  0.0f,
            -1.0f,  1.0f,  0.0f,
            -1.0f, -1.0f,  0.0f
    };
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * 6 * sizeof(float), points, GL_STATIC_DRAW);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void displayMipMap(int mipmapsId, int mipLevel, int screenWidth, int screenHeight, float scaleX, float scaleY) {
    GLint displayMipMap_location_mipMap = glGetUniformLocation(openGLProgram_displayMipMap, "mipMap"); checkGLError();
    GLint displayMipMap_location_mipLevel = glGetUniformLocation(openGLProgram_displayMipMap, "mipLevel"); checkGLError();
    GLint displayMipMap_location_mipMapWidth = glGetUniformLocation(openGLProgram_displayMipMap, "mipMapWidth"); checkGLError();
    GLint displayMipMap_location_mipMapHeight = glGetUniformLocation(openGLProgram_displayMipMap, "mipMapHeight"); checkGLError();
    GLint displayMipMap_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_displayMipMap, "texWidthKeyPoints"); checkGLError();
    GLint displayMipMap_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_displayMipMap, "texHeightKeyPoints"); checkGLError();
    glUseProgram(openGLProgram_displayMipMap); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId]); checkGLError();
    glUniform1i(displayMipMap_location_mipMap, 0); checkGLError();
    glUniform1i(displayMipMap_location_mipLevel, mipLevel); checkGLError();
    glUniform1i(displayMipMap_location_mipMapWidth, openGLMipMapsTextureWidth[mipmapsId] * scaleX); checkGLError();
    glUniform1i(displayMipMap_location_mipMapHeight, openGLMipMapsTextureHeight[mipmapsId] * scaleY); checkGLError();
    glUniform1i(displayMipMap_location_texWidthKeyPoints, screenWidth); checkGLError();
    glUniform1i(displayMipMap_location_texHeightKeyPoints, screenHeight); checkGLError();
    glViewport(0, 0, screenWidth, screenHeight);
    fullScreenRect();
    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void sampleDescriptors_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int mipLevels, std::vector<std::vector<Descriptor>>& destMips, float DESCRIPTORSCALE2, float MIPSCALE) {

    GLint sampleDescriptors_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthDescriptorShape"); checkGLError();
    GLint sampleDescriptors_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightDescriptorShape"); checkGLError();
    GLint sampleDescriptors_location_texWidthMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthMipMap"); checkGLError();
    GLint sampleDescriptors_location_texHeightMipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightMipMap"); checkGLError();
    GLint sampleDescriptors_location_mipLevels = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipLevels"); checkGLError();
    GLint sampleDescriptors_location_MIPSCALE = glGetUniformLocation(openGLProgram_sampleDescriptors, "MIPSCALE"); checkGLError();
    GLint sampleDescriptors_location_DESCRIPTORSCALE2 = glGetUniformLocation(openGLProgram_sampleDescriptors, "DESCRIPTORSCALE2"); checkGLError();
    GLint sampleDescriptors_location_keyPointCount = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointCount"); checkGLError();
    GLint sampleDescriptors_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthKeyPoints"); checkGLError();
    GLint sampleDescriptors_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightKeyPoints"); checkGLError();
    GLint sampleDescriptors_location_texWidthKeyPoints2 = glGetUniformLocation(openGLProgram_sampleDescriptors, "texWidthKeyPoints2"); checkGLError();
    GLint sampleDescriptors_location_texHeightKeyPoints2 = glGetUniformLocation(openGLProgram_sampleDescriptors, "texHeightKeyPoints2"); checkGLError();
    GLint sampleDescriptors_location_mipMap = glGetUniformLocation(openGLProgram_sampleDescriptors, "mipMap"); checkGLError();
    GLint sampleDescriptors_location_keyPointsx = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsx"); checkGLError();
    GLint sampleDescriptors_location_keyPointsy = glGetUniformLocation(openGLProgram_sampleDescriptors, "keyPointsy"); checkGLError();

    const int descriptorCount = openGLKeyPointCount[keyPointsId] * mipLevels;
    int descriptorX = descriptorCount;
    int descriptorY = 1;
    if (descriptorX > maxTextureSize) {
        descriptorY = (descriptorX + maxTextureSize - 1) / maxTextureSize;
        descriptorX = maxTextureSize;
    }
    if (descriptorX != openGLDescriptorRenderTargetWidth[descriptorsId] || descriptorY != openGLDescriptorRenderTargetHeight[descriptorsId]) {
        openGLDescriptorRenderTargetWidth[descriptorsId] = descriptorX;
        openGLDescriptorRenderTargetHeight[descriptorsId] = descriptorY;
        createUint324RenderTarget(openGLDescriptorRenderTarget[descriptorsId], (int)openGLDescriptorRenderTargetWidth[descriptorsId], (int)openGLDescriptorRenderTargetHeight[descriptorsId], true);
    }

    glUseProgram(openGLProgram_sampleDescriptors); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthDescriptorShape, openGLDescriptorShapeTextureWidth); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightDescriptorShape, openGLDescriptorShapeTextureHeight); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthMipMap, openGLMipMapsTextureWidth[mipmapsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightMipMap, openGLMipMapsTextureHeight[mipmapsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthKeyPoints, descriptorX); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightKeyPoints, descriptorY); checkGLError();
    glUniform1i(sampleDescriptors_location_texWidthKeyPoints2, openGLKeyPointsTextureWidth[keyPointsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_texHeightKeyPoints2, openGLKeyPointsTextureHeight[keyPointsId]); checkGLError();
    glUniform1f(sampleDescriptors_location_DESCRIPTORSCALE2, DESCRIPTORSCALE2); checkGLError();
    glUniform1f(sampleDescriptors_location_MIPSCALE, MIPSCALE); checkGLError();
    glUniform1i(sampleDescriptors_location_mipLevels, mipLevels); checkGLError();
    glUniform1i(sampleDescriptors_location_keyPointCount, openGLKeyPointCount[keyPointsId]); checkGLError();

    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLKeyPoints[keyPointsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_keyPointsx, 0); checkGLError();
    glActiveTexture(GL_TEXTURE1); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId]); checkGLError();
    glUniform1i(sampleDescriptors_location_mipMap, 1); checkGLError();

    glBindFramebuffer(GL_FRAMEBUFFER, openGLDescriptorRenderTarget[keyPointsId].first); checkGLError();
    glViewport(0, 0, descriptorX, descriptorY);
    fullScreenRect();

    unsigned int* data = new unsigned int[4 * descriptorX * descriptorY];
    glReadPixels(0, 0, descriptorX, descriptorY, GL_RGBA_INTEGER, GL_UNSIGNED_INT, data); checkGLError();

    destMips.resize(mipLevels);
    for (int mipMap = 0; mipMap < mipLevels; mipMap++) {
        std::vector<Descriptor>& descriptors = destMips[mipMap];
        descriptors.resize(openGLKeyPointCount[keyPointsId]);
        for (int i = 0; i < descriptors.size(); i++) {
            int i2 = mipMap * openGLKeyPointCount[keyPointsId] + i;
            int x = i2 % descriptorX;
            int y = i2 / descriptorX;
            int texAdr = x + y * descriptorX;
            for (int j = 0; j < Descriptor::uint32count; ++j) {
                descriptors[i].bits[j] = data[texAdr * 4 + j];
            }
        }
    }

    delete[] data;

    glBindFramebuffer(GL_FRAMEBUFFER, 0); checkGLError();
    glUseProgram(0); checkGLError();
    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, 0); checkGLError();
}

void refineKeyPoints_openGL(int keyPointsId, int descriptorsId, int mipmapsId, int keyPointCount, int mipLevels, int STEPCOUNT, float MIPSCALE, float STEPSIZE, float SCALEINVARIANCE, float ROTATIONINVARIANCE, std::vector<KeyPoint>& destKeyPoints, std::vector<float>& destErrors) {

    if (openGLKeyPointsTextureWidth[keyPointsId] != openGLKeyPointsRenderTargetWidth[keyPointsId] || openGLKeyPointsTextureHeight[keyPointsId] != (int)openGLKeyPointsRenderTargetHeight[keyPointsId]) {
        openGLKeyPointsRenderTargetWidth[keyPointsId] = openGLKeyPointsTextureWidth[keyPointsId];
        openGLKeyPointsRenderTargetHeight[keyPointsId] = openGLKeyPointsTextureHeight[keyPointsId];
        createFloat4RenderTarget(openGLKeyPointsRenderTarget[keyPointsId], (int)openGLKeyPointsRenderTargetWidth[keyPointsId], (int)openGLKeyPointsRenderTargetHeight[keyPointsId], true);
    }
    // OpenGL is not working, yet.
    GLint refineKeyPoints_location_descriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "descriptors"); checkGLError();
    GLint refineKeyPoints_location_mipMap = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMap"); checkGLError();
    GLint refineKeyPoints_location_mipMapWidth = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapWidth"); checkGLError();
    GLint refineKeyPoints_location_mipMapHeight = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipMapHeight"); checkGLError();
    GLint refineKeyPoints_location_texWidthDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptorShape"); checkGLError();
    GLint refineKeyPoints_location_texHeightDescriptorShape = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptorShape"); checkGLError();
    GLint refineKeyPoints_location_texWidthDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthDescriptors"); checkGLError();
    GLint refineKeyPoints_location_texHeightDescriptors = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightDescriptors"); checkGLError();
    GLint refineKeyPoints_location_mipLevels = glGetUniformLocation(openGLProgram_refineKeyPoints, "mipLevels"); checkGLError();
    GLint refineKeyPoints_location_STEPCOUNT = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPCOUNT"); checkGLError();
    GLint refineKeyPoints_location_MIPSCALE = glGetUniformLocation(openGLProgram_refineKeyPoints, "MIPSCALE"); checkGLError();
    GLint refineKeyPoints_location_STEPSIZE = glGetUniformLocation(openGLProgram_refineKeyPoints, "STEPSIZE"); checkGLError();
    GLint refineKeyPoints_location_SCALEINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "SCALEINVARIANCE"); checkGLError();
    GLint refineKeyPoints_location_ROTATIONINVARIANCE = glGetUniformLocation(openGLProgram_refineKeyPoints, "ROTATIONINVARIANCE"); checkGLError();
    GLint refineKeyPoints_location_texWidthKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texWidthKeyPoints"); checkGLError();
    GLint refineKeyPoints_location_texHeightKeyPoints = glGetUniformLocation(openGLProgram_refineKeyPoints, "texHeightKeyPoints"); checkGLError();
    GLint refineKeyPoints_location_keyPointsx = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsx"); checkGLError();
    GLint refineKeyPoints_location_keyPointsy = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointsy"); checkGLError();
    GLint refineKeyPoints_location_keyPointCount = glGetUniformLocation(openGLProgram_refineKeyPoints, "keyPointCount"); checkGLError();

    glUseProgram(openGLProgram_refineKeyPoints); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptorShape, openGLDescriptorShapeTextureWidth); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptorShape, openGLDescriptorShapeTextureHeight); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthDescriptors, openGLDescriptorsTextureWidth[descriptorsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightDescriptors, openGLDescriptorsTextureHeight[descriptorsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_mipLevels, mipLevels); checkGLError();
    glUniform1i(refineKeyPoints_location_STEPCOUNT, STEPCOUNT); checkGLError();
    glUniform1f(refineKeyPoints_location_MIPSCALE, MIPSCALE); checkGLError();
    glUniform1f(refineKeyPoints_location_STEPSIZE, STEPSIZE); checkGLError();
    glUniform1f(refineKeyPoints_location_SCALEINVARIANCE, SCALEINVARIANCE); checkGLError();
    glUniform1f(refineKeyPoints_location_ROTATIONINVARIANCE, ROTATIONINVARIANCE); checkGLError();
    glUniform1i(refineKeyPoints_location_texWidthKeyPoints, openGLKeyPointsTextureWidth[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_texHeightKeyPoints, openGLKeyPointsTextureHeight[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_mipMapWidth, openGLMipMapsTextureWidth[mipmapsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_mipMapHeight, openGLMipMapsTextureHeight[mipmapsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_keyPointCount, keyPointCount); checkGLError();

    glActiveTexture(GL_TEXTURE0); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLKeyPoints[keyPointsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_keyPointsx, 0); checkGLError();
    glActiveTexture(GL_TEXTURE1); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLMipMaps[mipmapsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_mipMap, 1); checkGLError();
    glActiveTexture(GL_TEXTURE2); checkGLError();
    glBindTexture(GL_TEXTURE_2D, openGLDescriptors[descriptorsId]); checkGLError();
    glUniform1i(refineKeyPoints_location_descriptors, 2); checkGLError();

    glBindFramebuffer(GL_FRAMEBUFFER, openGLKeyPointsRenderTarget[keyPointsId].first); checkGLError();
    const int w = (int)openGLKeyPointsRenderTargetWidth[keyPointsId];
    const int h = (int)openGLKeyPointsRenderTargetHeight[keyPointsId];
    glViewport(0, 0, w, h);
    fullScreenRect();
    float* data = new float[4 * w * h];
    glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, data); checkGLError();
    destKeyPoints.resize(keyPointCount);
    destErrors.resize(keyPointCount);
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