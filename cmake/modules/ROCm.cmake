# ROCm Module
if(USE_ROCM)
  find_rocm(${USE_ROCM} REQUIRED)
else(USE_ROCM)
  return()
endif()

###### Borrowed from MSHADOW project

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17"   SUPPORT_CXX17)

################################################################################################
# Config rocm compilation and append ROCm libraries to linker_libs
# Usage:
#  dgl_config_rocm(linker_libs)
macro(dgl_config_rocm linker_libs)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCm.")
  endif()

  enable_language(HIP)

  # always set the includedir when rocm is available
  # avoid global retrigger of cmake
	include_directories(/opt/rocm/include)

  add_definitions(-DDGL_USE_ROCM)

  list(APPEND ${linker_libs} 
    hip::host
    roc::hipblas
    roc::hipsparse
    hip::hiprand)
endmacro()
