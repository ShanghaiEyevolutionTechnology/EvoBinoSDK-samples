cmake_minimum_required(VERSION 3.1)

set(TARGET_NAME SampleWithPCL)

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
	# PCL Libraries
	find_package( PCL COMPONENTS common io visualization )
	if(PCL_FOUND)
		
		# add header and libraries
		add_definitions(${PCL_DEFINITIONS})
		include_directories(${PCL_INCLUDE_DIRS})
		link_directories(${PCL_LIBRARY_DIRS} )	
		include_directories(${EvoBinoSDK_INCLUDE_DIRS})
		list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
		
		if(insideSDK)
			ADD_EXE_MODULE(${TARGET_NAME} ${FOLDERNAME})
			set_target_properties (${TARGET_NAME} PROPERTIES FOLDER ${FOLDERNAME})
		else()
			file(GLOB DIR_HEADERS "*.h")
			file(GLOB DIR_CPP_SRCS "*.cpp")

			add_executable(${TARGET_NAME} ${DIR_HEADERS} ${DIR_CPP_SRCS})
		endif()

		target_link_libraries(${TARGET_NAME} PRIVATE ${EvoBinoSDK_LIBRARIES} ${PCL_LIBRARIES}) 
		
	endif(PCL_FOUND)

endif (EVO_BINO_SDK_WITH_CUDA)
