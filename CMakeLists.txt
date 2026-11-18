cmake_minimum_required(VERSION 3.26)
project(ReproducibilitySuperEMeans LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Commit of https://github.com/cwida/SuperKMeans used for the numbers in the blog post
set(SUPERKMEANS_TAG bb5101419c8cdb8a6d5a62f3289fa0aaac8e9ac5 CACHE STRING "SuperKMeans commit or tag")

include(FetchContent)
FetchContent_Declare(
    superkmeans
    GIT_REPOSITORY https://github.com/cwida/superkmeans
    GIT_TAG ${SUPERKMEANS_TAG}
)
FetchContent_MakeAvailable(superkmeans)

find_package(Threads REQUIRED)

add_executable(kmeans src/kmeans.cpp)
target_link_libraries(kmeans PRIVATE superkmeans Threads::Threads)

add_executable(emeans src/emeans.cpp)
target_link_libraries(emeans PRIVATE superkmeans Threads::Threads)
