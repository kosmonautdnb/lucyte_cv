cmake_minimum_required(VERSION 3.31.4)                            

project(Lucyte)                                         

find_package( OpenCV REQUIRED )                              
find_package( OpenCL REQUIRED )                              
find_package( OpenMP )

include_directories(${OpenCL_INCLUDE_DIRS})

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(lucyte_common config.hpp refinement.cpp refinement.hpp descriptors.cpp descriptors.hpp)
set(lucyte_opencl opencl_refinement.cpp opencl_refinement.hpp)

add_executable(lucyte_basic main.cpp ${lucyte_common})
target_link_libraries(lucyte_basic ${OpenCV_LIBS})

add_executable(lucyte_test_klt test_klt.cpp ${lucyte_common})
target_link_libraries(lucyte_test_klt ${OpenCV_LIBS})

add_executable(lucyte_test_klt_standalone test_klt_standalone.cpp ${lucyte_common})
target_link_libraries(lucyte_test_klt_standalone ${OpenCV_LIBS})

add_executable(lucyte_test_odometry test_odometry.cpp ${lucyte_common})
target_link_libraries(lucyte_test_odometry ${OpenCV_LIBS})

add_executable(lucyte_test_exotic test_exotic.cpp ${lucyte_common})
target_link_libraries(lucyte_test_exotic ${OpenCV_LIBS})

add_executable(lucyte_opencl opencl_main.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_test_odometry_opencl test_odometry_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_test_odometry_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_test_rectangle_opencl test_rectangle_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_test_rectangle_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_test_voronoi_opencl test_voronoi_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_test_voronoi_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_test_voronoi_stereo_opencl test_voronoi_stereo_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_test_voronoi_stereo_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})

add_executable(lucyte_test_feature_importance_opencl test_feature_importance_opencl.cpp ${lucyte_common} ${lucyte_opencl})
target_link_libraries(lucyte_test_feature_importance_opencl ${OpenCV_LIBS} ${OpenCL_LIBRARIES})