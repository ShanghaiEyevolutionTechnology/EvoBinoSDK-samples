cmake_minimum_required(VERSION 3.1)

set(TARGET_NAME SampleGetIMUData)

if(DEFINED PROJECT_NAME)
	set(insideSDK true)
endif(DEFINED PROJECT_NAME)

if(insideSDK)
	# inside sdk project
	SET(EvoBinoSDK_LIBRARIES evo_stereocamera)
else()
	# external build
	PROJECT( ${TARGET_NAME} )
	
	set(CMAKE_CXX_STANDARD 11) #C++ 11...
	set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
	set(CMAKE_CXX_EXTENSIONS OFF) #..without compiler extensions like gnu++11


	FIND_PACKAGE(EvoBinoSDK REQUIRED)
	
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}     "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EvoBinoSDK_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

if(insideSDK)
	ADD_EXE_MODULE(${TARGET_NAME} ${FOLDERNAME})
	set_target_properties (${TARGET_NAME} PROPERTIES FOLDER ${FOLDERNAME})
else()
	file(GLOB DIR_HEADERS "*.h")
	file(GLOB DIR_CPP_SRCS "*.cpp")

	add_executable(${TARGET_NAME} ${DIR_HEADERS} ${DIR_CPP_SRCS})
endif()

target_link_libraries(${TARGET_NAME} ${EvoBinoSDK_LIBRARIES}) 