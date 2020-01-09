#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NbgtV_1bPgaAebJsG2GyUx-a5VvAKF2P" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NbgtV_1bPgaAebJsG2GyUx-a5VvAKF2P" -o libtensorflowlite.so

echo Download finished.
