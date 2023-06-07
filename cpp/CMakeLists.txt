cmake_minimum_required(VERSION 3.22)
project(mobile_stereo_vision VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

set(CMAKE_GENERATOR_PLATFORM x64)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)
# include(CTest)
# enable_testing()

include("D:/ws/vcpkg/scripts/buildsystems/vcpkg.cmake")

set(SOURCES src/utils/ScopeBasedTimer.cpp src/utils/datatools.cpp src/stereovision/stereovision.cpp)
add_executable(mobile_stereo_vision src/main.cpp ${SOURCES})

# OpenCL
# find_package(OpenCL REQUIRED)
# target_link_libraries(mobile_stereo_vision OpenCL::OpenCL)

# LodePNG C++ API
# find_package(lodepng REQUIRED)
# target_link_libraries(mobile_stereo_vision lodepng)

# libcurl
# find_package(CURL REQUIRED)
# target_link_libraries(mobile_stereo_vision CURL::libcurl)

# libzip
# find_package(libzip REQUIRED)
# target_link_libraries(mobile_stereo_vision libzip::zip)

# PCL
find_package(PCL CONFIG REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(mobile_stereo_vision ${PCL_LIBRARIES})

# find_package(Eigen3 CONFIG REQUIRED)
# target_link_libraries(mobile_stereo_vision Eigen3::Eigen)

# OpenCV
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(mobile_stereo_vision opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)


add_definitions("-DCMAKE_TOOLCHAIN_FILE=D:/ws/vcpkg/scripts/buildsystems/vcpkg.cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=D:/dev/open3d-devel -DCMAKE_GENERATOR_PLATFORM=x64")

# add_custom_command(TARGET mobile_stereo_vision POST_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E copy_directory
#                        ${CMAKE_SOURCE_DIR}/src/kernels/ $<TARGET_FILE_DIR:mobile_stereo_vision>/kernels/)

add_custom_command(TARGET mobile_stereo_vision POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                           ${CMAKE_SOURCE_DIR}/data/ $<TARGET_FILE_DIR:mobile_stereo_vision>/data/)





# # Open3D
# set(Open3D_DIR "D:/dev/open3d-devel/CMake")
# set(CMAKE_INSTALL_PREFIX "D:/dev/open3d-devel")


# # The options need to be the same as Open3D's default
# # If Open3D is configured and built with custom options, you'll also need to
# # specify the same custom options.
# option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
# if(STATIC_WINDOWS_RUNTIME)
#     set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
# else()
#     set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
# endif()
# find_package(Open3D REQUIRED)
# target_link_libraries(mobile_stereo_vision Open3D::Open3D)


# # On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
# if(WIN32)
#     get_target_property(open3d_type Open3D::Open3D TYPE)
#     if(open3d_type STREQUAL "SHARED_LIBRARY")
#         message(STATUS "Copying Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
#         add_custom_command(TARGET mobile_stereo_vision POST_BUILD
#                            COMMAND ${CMAKE_COMMAND} -E copy
#                                    ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
#                                    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
#     endif()
# endif()