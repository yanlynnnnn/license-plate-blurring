#!/bin/bash

if [ $# -eq 0 ]; then
    pyadb pull "/storage/emulated/0/DCIM/OpenCamera/" 
else
    pyadb pull "/storage/emulated/0/DCIM/OpenCamera/VID_$1_$2.mp4" 
    pyadb pull "/storage/emulated/0/DCIM/OpenCamera/$2" 
fi 

