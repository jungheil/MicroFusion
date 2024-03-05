cmake_minimum_required(VERSION 3.16)
project(microfusion VERSION 1.0.1)

if(WIN32)
    message(STATUS "Detected Windows")
    add_definitions(-DMACRO_WINDOWS)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Detected Linux")
    add_definitions(-DMACRO_LINUX)
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()

set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(WIN32)
    get_filename_component(BOOST_ABSOLUTE_PATH ${PROJECT_SOURCE_DIR}/3rd_party/boost ABSOLUTE)
    set(Boost_ROOT ${BOOST_ABSOLUTE_PATH})
    set(Boost_LIBRARY_DIRS ${BOOST_ABSOLUTE_PATH}/lib64-msvc-14.1)
    set(Boost_INCLUDE_DIRS ${BOOST_ABSOLUTE_PATH})
    set(Boost_NO_SYSTEM_PATHS ON)
endif()
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if(WIN32)
  set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/3rd_party/eigen/share/eigen3/cmake)
endif()
find_package(Eigen3 REQUIRED CONFIG)


if(WIN32)
    set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rd_party/opencv/build)
endif()
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/gsl/include)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/gsl/lib
)
elseif(UNIX AND NOT APPLE)
find_package(GSL REQUIRED)
set(GSL_LINK_LIB "GSL::gsl")
endif()
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/onnxruntime/include)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/onnxruntime/lib)

if(WIN32)
include_directories(${PROJECT_SOURCE_DIR}/3rd_party/or-tools/include)
link_directories(${PROJECT_SOURCE_DIR}/3rd_party/or-tools/lib)
set(ortools_LINK_LIB "ortools")
elseif(UNIX AND NOT APPLE)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${PROJECT_SOURCE_DIR}/3rd_party/or-tools")
find_package(ortools REQUIRED CONFIG)
set(ortools_LINK_LIB "ortools::ortools")
endif()

include_directories(${PROJECT_SOURCE_DIR}/3rd_party/nanoflann/include)

aux_source_directory(. SRC_LIST)
include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR})

aux_source_directory(${PROJECT_SOURCE_DIR}/assignment ASSIGNMENT_EXTR_SRC_LIST)
aux_source_directory(${PROJECT_SOURCE_DIR}/feature_extr FEATURE_EXTR_SRC_LIST)
aux_source_directory(${PROJECT_SOURCE_DIR}/fusion FUSION_SRC_LIST)
aux_source_directory(${PROJECT_SOURCE_DIR}/persistent PERSISTENT_SRC_LIST)


add_library(${PROJECT_NAME} ${SRC_LIST} ${FEATURE_EXTR_SRC_LIST} ${ASSIGNMENT_EXTR_SRC_LIST} ${PERSISTENT_SRC_LIST} ${FUSION_SRC_LIST} )
target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime ${GSL_LINK_LIB}  ${ortools_LINK_LIB})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})