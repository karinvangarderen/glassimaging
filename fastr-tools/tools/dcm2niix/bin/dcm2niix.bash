#!/bin/bash

#Check input dir
inputdir=$1
shift
if [ ! -d $inputdir ] ; then
    echo "Input dir not valid" >&2
fi

# Try to create temp dir
tmpdir=$(mktemp -d)
if [ ! -d $tmpdir ] ; then 
    echo "Creating temp dir failed" >&2
fi

ln -s $inputdir ${tmpdir}/input

#Start conversion
echo "Starting conversion"
dcm2niix -z y -v 1 $@ ${tmpdir}/input

# remove link
rm ${tmpdir}/input

# remove tmpdir
rmdir ${tmpdir}

