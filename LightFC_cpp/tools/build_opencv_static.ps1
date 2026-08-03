[CmdletBinding()]
param(
    [string]$OpenCvSource = 'F:\opencv4.14.0\opencv\sources',
    [string]$CMakeExe = 'D:\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$staticRoot = Join-Path $projectRoot 'third_party\opencv-static'
$buildDir = Join-Path $staticRoot 'build'
$installDir = Join-Path $staticRoot 'install'
$downloadDir = Join-Path $staticRoot 'downloads'

if (-not (Test-Path -LiteralPath $OpenCvSource)) {
    throw "OpenCV source directory does not exist: $OpenCvSource"
}
if (-not (Test-Path -LiteralPath $CMakeExe)) {
    throw "CMake executable does not exist: $CMakeExe"
}

$configureArguments = @(
    '-S', $OpenCvSource,
    '-B', $buildDir,
    '-G', 'Visual Studio 17 2022',
    '-A', 'x64',
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    '-DBUILD_SHARED_LIBS=OFF',
    '-DBUILD_WITH_STATIC_CRT=OFF',
    '-DBUILD_LIST=core,imgproc,imgcodecs,videoio',
    '-DBUILD_opencv_world=OFF',
    '-DBUILD_TESTS=OFF',
    '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_EXAMPLES=OFF',
    '-DBUILD_opencv_apps=OFF',
    '-DBUILD_JAVA=OFF',
    '-DBUILD_opencv_python3=OFF',
    '-DWITH_PROTOBUF=OFF',
    '-DWITH_ADE=OFF',
    '-DWITH_FFMPEG=ON',
    '-DWITH_MSMF=ON',
    '-DWITH_IPP=OFF',
    '-DWITH_OPENCL=OFF',
    '-DWITH_OPENMP=ON',
    '-DWITH_TIFF=OFF',
    '-DWITH_WEBP=OFF',
    '-DWITH_OPENJPEG=OFF',
    '-DWITH_JASPER=OFF',
    '-DWITH_OPENEXR=OFF',
    "-DOPENCV_DOWNLOAD_PATH=$downloadDir"
)

& $CMakeExe @configureArguments
if ($LASTEXITCODE -ne 0) { throw "OpenCV configure failed: $LASTEXITCODE" }

& $CMakeExe --build $buildDir --config Release --target INSTALL -- /m
if ($LASTEXITCODE -ne 0) { throw "OpenCV build failed: $LASTEXITCODE" }

Write-Host "Static OpenCV installed to: $installDir"
Write-Host 'Note: the Windows FFmpeg wrapper remains a runtime DLL for RTSP support.'
