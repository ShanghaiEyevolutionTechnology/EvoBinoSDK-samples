cmake_minimum_required(VERSION 3.1)

set(TARGET_NAME SampleWithOpenGL)

if(DEFINED PROJECT_NAME)
	set(insideSDK true)
endif(DEFINED PROJECT_NAME)

if(insideSDK)
	# inside sdk project
	set(EvoBinoSDK_LIBRARIES evo_depthcamera)
else()
	# external build
	project( ${TARGET_NAME} )
	
	set(CMAKE_CXX_STANDARD 11) #C++ 11...
	set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
	set(CMAKE_CXX_EXTENSIONS OFF) #..without compiler extensions like gnu++11

	find_package(EvoBinoSDK REQUIRED)
endif()
	
if (EVO_BINO_SDK_WITH_CUDA)

	find_package(CUDA REQUIRED)
	find_package(GLUT REQUIRED)
	find_package(OpenGL REQUIRED)

	include_directories(${EvoBinoSDK_INCLUDE_DIRS})
	include_directories(${CUDA_INCLUDE_DIRS})
	include_directories(${GLUT_INCLUDE_DIR})
	include_directories(${OPENGL_INCLUDE_DIR})

	if(insideSDK)
		ADD_EXE_MODULE(${TARGET_NAME} ${FOLDERNAME})
		set_target_properties (${TARGET_NAME} PROPERTIES FOLDER ${FOLDERNAME})
	else()
		file(GLOB DIR_HEADERS "*.h")
		file(GLOB DIR_CPP_SRCS "*.cpp")

		add_executable(${TARGET_NAME} ${DIR_HEADERS} ${DIR_CPP_SRCS})
	endif()

	target_link_libraries(${TARGET_NAME} PRIVATE ${EvoBinoSDK_LIBRARIES} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES}) 

endif (EVO_BINO_SDK_WITH_CUDA)
