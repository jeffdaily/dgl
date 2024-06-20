#!/bin/bash

# go one level up from script/hipify.sh
top=$(dirname "${BASH_SOURCE[0]}")/..
echo "top=$top"
cd $top
echo "pwd=`pwd`"

CUDA_DIRS="
third_party/HugeCTR/gpu_cache/src
third_party/HugeCTR/gpu_cache/include
third_party/HugeCTR/gpu_cache/test
src/array/cuda
src/array/cuda/uvm
src/kernel/cuda
src/partition/cuda
src/runtime/cuda
src/geometry/cuda
src/graph/transform/cuda
src/graph/sampling/randomwalks
"

EXTENSIONS="cu cuh h cc hpp cpp"

rename_cuda_dir () {
    tmp=$(echo $1   | sed 's@cuda@hip@')
    tmp=$(echo $tmp | sed 's@randomwalks@randomwalks_hip@')
    tmp=$(echo $tmp | sed 's@gpu_cache@gpu_cache_hip@')
    echo $tmp
}

# create all destination directories for hipified files into sibling 'hip' directory
for cuda_dir in $CUDA_DIRS
do
    hip_dir=$(rename_cuda_dir $cuda_dir)
    mkdir -p $hip_dir
    echo "Created $hip_dir"
done

# run hipify-perl against all *.cu *.cuh *.h *.cc files, no renaming
# run all files in parallel to speed up
for cuda_dir in $CUDA_DIRS
do
    echo "searching for source files in $cuda_dir"
    for ext in $EXTENSIONS
    do
        for src in $(find $cuda_dir -name "*.$ext")
        do
            dst=$(rename_cuda_dir $src)
            echo "hipify-perl -o=$dst.tmp $src &"
            hipify-perl -o=$dst.tmp $src &
        done
    done
done
wait

# rename all hipified *.cu files to *.hip
for src in $(find . -name "*.cu.tmp")
do
    dst=${src%.cu.tmp}.hip.tmp
    mv $src $dst
done

# replace header include statements from cuda paths to new hip paths
# hipify-perl gets header paths wrong, swap hipblas.h for hipblas/hipblas.h et al
# replace thrust::cuda::par with thrust::hip::par
for cuda_dir in $CUDA_DIRS
do
    hip_dir=$(rename_cuda_dir $cuda_dir)
    for ext in $EXTENSIONS hip
    do
        for src in $(find $hip_dir -name "*.$ext.tmp")
        do
            sed -i 's@#include "../../array/cuda/@#include "../../array/hip/@' $src
            sed -i 's@#include "../../kernel/cuda/@#include "../../kernel/hip/@' $src
            sed -i 's@#include "../../partition/cuda/@#include "../../partition/hip/@' $src
            sed -i 's@#include "../../runtime/cuda/@#include "../../runtime/hip/@' $src
            sed -i 's@#include "../../geometry/cuda/@#include "../../geometry/hip/@' $src
            sed -i 's@#include "../../graph/transform/cuda/@#include "../../graph/transform/hip/@' $src
            sed -i 's@#include "../../graph/transform/cuda/@#include "../../graph/transform/hip/@' $src
            sed -i 's@#include <hipblas.h>@#include <hipblas/hipblas.h>@' $src
            sed -i 's@#include <hipsparse.h>@#include <hipsparse/hipsparse.h>@' $src
            sed -i 's@thrust::cuda::par@thrust::hip::par@' $src
        done
    done
done

# hipify was run in parallel above
# don't copy the tmp file if it is unchanged
for ext in $EXTENSIONS hip
do
    for src in $(find . -name "*.$ext.tmp")
    do
        dst=${src%.tmp}
        if test -f $dst
        then
            if diff -q $src $dst >& /dev/null
            then
                echo "$dst [unchanged]"
                rm $src
            else
                echo "$dst hipified"
                mv $src $dst
            fi
        else
            echo "$dst"
            mv $src $dst
        fi
    done
done
