cmake_minimum_required(VERSION 3.24)
project(OOP_Labs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- warnings ----------
add_compile_options(-Wall -Wextra -Wpedantic)

# ---------- sanitizers (только Debug) ----------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---------- core library ----------
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/Dataset.cpp"
)

add_library(core STATIC ${CORE_SOURCES})
target_include_directories(core PUBLIC "${CMAKE_SOURCE_DIR}/include")

# ---------- console app ----------
add_executable(Lab1 src/Lab1.cpp)
target_link_libraries(Lab1 PRIVATE core)

# ---------- tests ----------
enable_testing()

set(TESTS
    binary_test
    column_type_test
    encoding_test
    missing_value_test
    quantile
    consistency_test
)

foreach(test_name IN LISTS TESTS)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE core)
    add_test(NAME ${test_name}
             COMMAND ${test_name}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()