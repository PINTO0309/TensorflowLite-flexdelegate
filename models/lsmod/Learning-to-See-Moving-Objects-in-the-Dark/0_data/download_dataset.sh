#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1tl4Byi5tA7RJFiv8YG_LRD-8ZzIjl5wu" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1tl4Byi5tA7RJFiv8YG_LRD-8ZzIjl5wu" -o Cam1.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NL7eO90oax15gbbUnZ2L4-p4UHP06LvH" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NL7eO90oax15gbbUnZ2L4-p4UHP06LvH" -o Cam2.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1d_TtebcuEibYhIotED7TxyG9xHOP6sZq" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1d_TtebcuEibYhIotED7TxyG9xHOP6sZq" -o Outdoor.tar.gz

tar -zxvf Cam1.tar.gz
tar -zxvf Cam2.tar.gz
tar -zxvf Outdoor.tar.gz

rm Cam1.tar.gz
rm Cam2.tar.gz
rm Outdoor.tar.gz

echo Download finished.

