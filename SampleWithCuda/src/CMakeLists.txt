cmake_minimum_required(VERSION 3.1)

set(TARGET_NAME SampleWithCuda)

if(DEFINED PROJECT_NAME)
	set(insideSDK true)
endif(DEFINED PROJECT_NAME)

if(insideSDK)
	# inside sdk project
	set(EvoBinoSDK_LIBRARIES evo_depthcamera)
else()
	# external build
	project( ${TARGET_NAME} )
	
	if (CMAKE_VERSION VERSION_LESS "3.9.0")
		# Set only because there are old target_link_libraries() statements in the
		# FindCUDA.cmake file that comes with CMake that error out if the new behavior
		# is used.  In newer versions of CMake we can instead set CUDA_LINK_LIBRARIES_KEYWORD which fixes this issue.
		cmake_policy(SET CMP0023 OLD)
	else()
		set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
	endif()
	
	set(CMAKE_CXX_STANDARD 11) #C++ 11...
	set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
	set(CMAKE_CXX_EXTENSIONS OFF) #..without compiler extensions like gnu++11

	find_package(EvoBinoSDK REQUIRED)
endif()

if (EVO_BINO_SDK_WITH_CUDA)

	find_package(CUDA REQUIRED)
	find_package(OpenCV REQUIRED)

	include_directories(${EvoBinoSDK_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

	if(insideSDK)
		ADD_CUDAEXE_MODULE(${TARGET_NAME} ${FOLDERNAME})
		set_target_properties (${TARGET_NAME} PROPERTIES FOLDER ${FOLDERNAME})
	else()
		file(GLOB SOURCE_FILES "*.cuh" "*.cu" "*.cpp")
		# nvcc flags; you may have to change the sm_xx, compute_xx for your GPU
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -lineinfo -gencode arch=compute_30,code=\"sm_30,compute_30\")
		if(WIN32)
			set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler \"/wd 4819\"")
		endif(WIN32)	
		
		cuda_add_executable(${TARGET_NAME} ${SOURCE_FILES})
	endif()

	target_link_libraries(${TARGET_NAME} PRIVATE ${EvoBinoSDK_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES}) 

endif (EVO_BINO_SDK_WITH_CUDA)
