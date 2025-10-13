# Vehicle Postprocessing

This directory contains scripts for postprocessing vehicle images, including background removal and image enhancement.

It is a bit messy right now since I was experimenting with different methods. I had the idea to use sam2 to remove the background and then use SRGAN to increase the image size. This actually works pretty well but I haven't had the time to clean up the code and run it on the whole dataset.