cmake_minimum_required(VERSION 3.21)
project(LightFCNcnnWeb VERSION 1.0 LANGUAGES CXX)

if(MSVC)
    # The supplied NCNN and the locally built static OpenCV package are Release
    # binaries. Restrict the generated VS solution to the compatible config.
    set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "VS configurations" FORCE)
endif()

if(NOT WIN32)
    message(FATAL_ERROR "当前 HTTP 实现使用 Winsock，仅支持 Windows/VS2022")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "请使用 x64 配置；随项目提供的默认 ncnn 链接目录为 ncnn/x64")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/x64/lib/cmake/ncnn" CACHE PATH "ncnn CMake package")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv-static/install/x64/vc17/staticlib" CACHE PATH "OpenCV CMake package")
set(OPENCV_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv-static/install/x64/vc17/bin" CACHE PATH "OpenCV FFmpeg runtime directory")
set(OpenCV_STATIC ON)
find_package(ncnn CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# The supplied static package only publishes a Release import. Reuse it for the
# other VS configurations; our own sources may still be compiled as Debug.
set_property(TARGET ncnn PROPERTY MAP_IMPORTED_CONFIG_DEBUG Release)
set_property(TARGET ncnn PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
set_property(TARGET ncnn PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL Release)

add_library(lightfc_core STATIC
    src/preprocess.cpp
    src/tracker.cpp
    src/rtsp_service.cpp
    src/http_server.cpp)
target_include_directories(lightfc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(lightfc_core PUBLIC ncnn ${OpenCV_LIBS} ws2_32)
target_compile_definitions(lightfc_core PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

add_executable(lightfc_web src/main.cpp)
target_link_libraries(lightfc_web PRIVATE lightfc_core shell32)
set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT lightfc_web)

add_executable(lightfc_compare_models tests/compare_tracking_models.cpp)
target_link_libraries(lightfc_compare_models PRIVATE ncnn)
enable_testing()
add_test(NAME lightfc_tracking_model_equivalence
         COMMAND lightfc_compare_models "${CMAKE_CURRENT_SOURCE_DIR}/model")
if(MSVC)
    target_compile_options(lightfc_compare_models PRIVATE /utf-8 /W4 /permissive- /EHsc)
endif()

if(MSVC)
    target_compile_options(lightfc_core PRIVATE /utf-8 /W4 /permissive- /EHsc)
    target_compile_options(lightfc_web PRIVATE /utf-8 /W4 /permissive- /EHsc)
endif()

add_custom_command(TARGET lightfc_web POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:lightfc_web>/model"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/model/lightfc_template.ncnn.param"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/lightfc_template.ncnn.bin"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/lightfc_tracking.ncnn.param"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/lightfc_tracking.ncnn.bin"
            "$<TARGET_FILE_DIR:lightfc_web>/model"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/web" "$<TARGET_FILE_DIR:lightfc_web>/web"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OPENCV_RUNTIME_DIR}/opencv_videoio_ffmpeg4140_64.dll"
            "$<TARGET_FILE_DIR:lightfc_web>"
    COMMENT "Copying LightFC models and web UI next to the executable")

install(TARGETS lightfc_web RUNTIME DESTINATION .)
install(FILES
    model/lightfc_template.ncnn.param
    model/lightfc_template.ncnn.bin
    model/lightfc_tracking.ncnn.param
    model/lightfc_tracking.ncnn.bin
    DESTINATION model)
install(DIRECTORY web DESTINATION .)
