cmake_minimum_required(VERSION 3.16)
project(pgdl)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/pgsql/include/server)
include_directories(/home/hh/oneDNN/examples)
include_directories(usr/local/lib)
add_executable(pgdl
#        pgdl.c
#        Wrapper.cpp
#        Wrapper.h
        main.cpp
        Manager.cpp
        Manager.h
        Autodiff.cpp
        Autodiff.h
        CreateNode.h
        CreateNode.cpp
        AutodiffMethod.cpp
        AutodiffMethod.h Execute.cpp Execute.h CreateNodeClient.cpp CreateNodeClient.h ManagerClient.cpp ManagerClient.h Graph.cpp Graph.h ModelClient.cpp ModelClient.h)
target_link_libraries(pgdl dnnl pthread m)
