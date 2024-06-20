#######################################################
# Usage:
#   find_rocm(${USE_ROCM})
#
# - When USE_ROCM=ON, use auto search
#
# Please use the CMAKE variable ROCM_HOME to set ROCm directory
#
# Provide variables:
#
# - ROCM_FOUND
#
macro(find_rocm use_rocm)
    find_package(HIP REQUIRED)
    find_package(hipBLAS REQUIRED)
    find_package(hipRAND REQUIRED)
    find_package(hipSPARSE REQUIRED)
    if(HIP_FOUND)
        set(ROCM_FOUND TRUE)
    endif()
endmacro(find_rocm)
