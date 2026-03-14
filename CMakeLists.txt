cmake_minimum_required(VERSION 3.14)
project(efopt VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Eigen3 REQUIRED)

# Find nlohmann_json, if not found, fetch it
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# EFOPT library
add_library(efopt STATIC
    src/efopt.cpp
)

target_include_directories(efopt PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(efopt PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

# Test executable
option(EFOPT_BUILD_TESTS "Build test programs" ON)
if(EFOPT_BUILD_TESTS)
    add_executable(test_efopt test_efopt.cpp)
    target_link_libraries(test_efopt PRIVATE efopt)
endif()

# Installation (optional, requires system nlohmann_json)
option(EFOPT_INSTALL "Install EFOPT library" OFF)
if(EFOPT_INSTALL)
    install(TARGETS efopt
        EXPORT efoptTargets
        INCLUDES DESTINATION include
    )

    install(DIRECTORY include/efopt DESTINATION include)

    install(EXPORT efoptTargets
        FILE efoptTargets.cmake
        NAMESPACE efopt::
        DESTINATION lib/cmake/efopt
    )

    # Package configuration
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/efoptConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/efoptConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/efoptConfig.cmake"
        INSTALL_DESTINATION lib/cmake/efopt
    )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/efoptConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/efoptConfigVersion.cmake"
        DESTINATION lib/cmake/efopt
    )
endif()
