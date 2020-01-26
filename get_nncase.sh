#!/bin/bash

mkdir -p ncc
mkdir -p workspace
mkdir -p images
mkdir -p log

pushd ncc
wget https://github.com/kendryte/nncase/releases/download/v0.1.0-rc5/ncc-linux-x86_64.tar.xz
tar -Jxf ncc-linux-x86_64.tar.xz
rm ncc-linux-x86_64.tar.xz
echo "download nncase ok!"
popd
