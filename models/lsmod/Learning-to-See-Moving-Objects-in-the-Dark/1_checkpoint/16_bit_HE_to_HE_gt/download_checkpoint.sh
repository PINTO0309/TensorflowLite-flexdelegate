#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=11i8syPcTZQ5zBmbJTGGbWfbO0Lm0Tl2J" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=11i8syPcTZQ5zBmbJTGGbWfbO0Lm0Tl2J" -o checkpoint.tar.gz
tar -zxvf checkpoint.tar.gz
rm checkpoint.tar.gz

echo Download finished.

