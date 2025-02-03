// Lucyte Created on: 03.02.2025 by Stefan Mader
#include "opengl_refinement.hpp"

int main(int argc, char** argv)
{
    initOpenGL();
    cv::namedWindow("Lucyte");
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glfwGetFramebufferSize(window, &window_width, &window_height);
        glViewport(0, 0, window_width, window_height);
        glClearColor(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        if (cv::waitKey(100) == 27)
            break;
    }

    return 0;
}
