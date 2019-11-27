cmake_minimum_required(VERSION 2.8)
project(enettest)

# Switch build target
set(BUILD_TARGET PC CACHE STRING "Build target?")
set(LINK_ONLY OFF CACHE BOOL "Use pre-built library?")

# Common build settings
include(cmakes/common.cmake)

# Build settings for each target
if(${BUILD_TARGET} STREQUAL PC)
	if(WIN32)
		message(FATAL_ERROR "[BUILD] PC Windows is not supported")
	else()
		message("[BUILD] PC Linux")
		include(cmakes/PC_LINUX.cmake)
	endif()
elseif(${BUILD_TARGET} STREQUAL RASPI_NATIVE)
	message("[BUILD] Raspberry Pi Native")
	include(cmakes/RASPI_NATIVE.cmake)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()

# Create Main project
add_executable(enettest
	main.cpp
)

# For OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_include_directories(enettest PUBLIC ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(enettest ${OpenCV_LIBS})
endif()

# For Tensorflow Lite
if((${BUILD_TARGET} STREQUAL PC) AND UNIX)
	# use shared library
	target_link_libraries(enettest ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt/linux-cpu-x86_64/libtensorflowlite.so)
	file(COPY ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt/linux-cpu-x86_64/libtensorflowlite.so DESTINATION ${PROJECT_BINARY_DIR}/)
elseif(${BUILD_TARGET} STREQUAL RASPI_NATIVE)
	target_link_libraries(enettest ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt/raspi-cpu-armv7l/libtensorflowlite.so)
	file(COPY ${CMAKE_SOURCE_DIR}/external_libs/tensorflow_prebuilt/raspi-cpu-armv7l/libtensorflowlite.so DESTINATION ${PROJECT_BINARY_DIR}/)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/eigen )
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/absl)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/gemmlowp)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorlow/lite/tools/make/downloads/neon_2_sse)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/farmhash/src)
target_include_directories(enettest PUBLIC ${CMAKE_SOURCE_DIR}/external_libs/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)

# Copy resouce
file(COPY ${CMAKE_SOURCE_DIR}/resource/ DESTINATION ${PROJECT_BINARY_DIR}/resource/)
add_definitions(-DRESOURCE_DIR="${PROJECT_BINARY_DIR}/resource/")
