set(cmake_cxx_flags "-std=c++14 -Ofast -march=native")
set(ArrayFire_DIR /opt/arrayfire/share/ArrayFire/cmake)
find_package(ArrayFire)
macro(add_code name)
	if(ArrayFire_CPU_FOUND)
		# Vectorization examples
		add_executable(${name}_cpu ${name}.cpp)
		target_link_libraries(${name}_cpu ArrayFire::afcpu)
	endif()

	if(ArrayFire_CUDA_FOUND)
		add_executable(${name}_cuda ${name}.cpp)
		target_link_libraries(${name}_cuda ArrayFire::afcuda)
	endif()
endmacro()

add_code(test_af)
