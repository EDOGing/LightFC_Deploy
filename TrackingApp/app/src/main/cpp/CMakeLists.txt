cmake_minimum_required(VERSION 3.22.1)

project(ncnntracker LANGUAGES CXX)

if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "TrackingApp only supports arm64-v8a, got: ${ANDROID_ABI}")
endif()

# The official ncnn package was exported by a pre-CMake-4 policy version.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(ncnn_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/${ANDROID_ABI}/lib/cmake/ncnn"
)
find_package(ncnn REQUIRED CONFIG)

set(OpenCV_DIR "F:/opencv4.14.0/opencv-4.14.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV 4.14 REQUIRED CONFIG COMPONENTS core imgproc)

add_library(
    ncnntracker
    SHARED
    native-lib.cpp
    runtime/ncnn_runtime.cpp
)

target_include_directories(ncnntracker PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_features(ncnntracker PRIVATE cxx_std_17)
target_compile_options(ncnntracker PRIVATE -Wall -Wextra -Wpedantic)

find_library(log-lib log REQUIRED)
target_link_libraries(ncnntracker PRIVATE ncnn ${OpenCV_LIBS} ${log-lib})
