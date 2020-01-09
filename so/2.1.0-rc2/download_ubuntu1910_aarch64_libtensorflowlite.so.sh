#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PeayNKafP9OSyFMD_FMc1hjTXD0Bmu6B" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PeayNKafP9OSyFMD_FMc1hjTXD0Bmu6B" -o libtensorflowlite.so

echo Download finished.
