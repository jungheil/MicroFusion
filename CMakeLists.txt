cmake_minimum_required(VERSION 3.16)
project(zy_fusion)

if(WIN32)
    message(STATUS "Detected Windows")
    add_definitions(-DMACRO_WIN32)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Detected Linux")
    add_definitions(-DMACRO_LINUX)
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()

set_property(GLOBAL PROPERTY PROJECT_ROOT_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/dist)
add_subdirectory(micro_fusion)

if(WIN32)
  set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/micro_fusion/3rd_party/eigen/share/eigen3/cmake)
endif()
find_package(Eigen3 REQUIRED CONFIG)

if(WIN32)
  set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/micro_fusion/3rd_party/opencv/build)
endif()
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/micro_fusion/include)

add_executable(${PROJECT_NAME} demo.cpp)

target_link_libraries(${PROJECT_NAME}  Eigen3::Eigen ${OpenCV_LIBS} microfusion)

if(WIN32)
get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>

    )
endif()
