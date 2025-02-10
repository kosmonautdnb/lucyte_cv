cmake_minimum_required(VERSION 3.31.4)                            

project(Lucyte)                                         

function(dump_cmake_variables)
	get_cmake_property(_variableNames VARIABLES)
	list (SORT _variableNames)
	foreach (_variableName ${_variableNames})
	    message(STATUS "${_variableName}=${${_variableName}}")
	endforeach()
endfunction()

list(APPEND CMAKE_PREFIX_PATH "c:/!mad/lib/glew")
find_package( GLEW )                              
find_package( OpenCV REQUIRED )                              
find_package( OpenCL REQUIRED )                              
find_package( OpenMP )
find_package( OpenGL )
find_package( glfw3 )


if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${OpenCL_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS})

set(lucyte_common constants.hpp config.hpp refinement.cpp refinement.hpp descriptors.cpp descriptors.hpp)

set(lucyte_opencl opencl_refinement.cpp opencl_refinement.hpp)

set(lucyte_opengl opengl_refinement.cpp opengl_refinement.hpp)


add_executable(lucyte_basic main.cpp ${lucyte_common})
target_link_libraries(lucyte_basic ${OpenCV_LIBS})

add_executable(lucyte_check_klt check_klt.cpp ${lucyte_common})
target_link_libraries(lucyte_check_klt ${OpenCV_LIBS})

add_executable(lucyte_check_klt_standalone check_klt_standalone.cpp ${lucyte_common})
target_link_libraries(lucyte_check_klt_standalone ${OpenCV_LIBS})

add_executable(lucyte_check_odometry check_odometry.cpp ${lucyte_common})
target_link_libraries(lucyte_check_odometry ${OpenCV_LIBS})

add_executable(lucyte_check_exotic check_exotic.cpp ${lucyte_common})
target_link_libraries(lucyte_check_exotic ${OpenCV_LIBS})

add_executable(lucyte_opencl opencl_main.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_odometry_opencl check_odometry_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_odometry_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_rectangle_opencl check_rectangle_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_rectangle_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_voronoi_opencl check_voronoi_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_voronoi_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_voronoi_stereo_opencl check_voronoi_stereo_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_voronoi_stereo_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_feature_importance_opencl check_feature_importance_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_feature_importance_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_opencl_nonblocking opencl_main_nonblocking.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_opencl_nonblocking ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_KITTIE check_odometry_kittie_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_KITTIE ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_KITTIE_stereo check_odometry_kittie_stereo_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_KITTIE_stereo ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_check_webcam_opencl check_webcam_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_check_webcam_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

if (OpenGL_FOUND AND glfw3_FOUND)
	add_executable(lucyte_opengl opengl_main.cpp ${lucyte_common} ${lucyte_opengl})
	target_link_libraries(lucyte_opengl ${OpenCV_LIBS} glfw ${OPENGL_LIBRARIES} GLEW::GLEW)
endif()
