cmake_minimum_required(VERSION 3.22.1)

project("lucyteandroid")

set(LucyteNativeSources
        constants.hpp
        descriptors.hpp
        descriptors.cpp
        opengl_refinement.cpp
        opengl_refinement.hpp
        refinement.hpp
        refinement.cpp
        ndkjni.cpp
)

add_library(lucytelib SHARED ${LucyteNativeSources})
target_link_libraries(lucytelib android log EGL GLESv3)