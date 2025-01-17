// Created on: 17.01.2025 by Stefan Mader
#pragma once

const char* outputVideoFileName = "c:/!mad/video.avi";
const int outputVideoFrameRate = 25;

#define FILESET 1

#if FILESET == 1
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street3/wandern%04d.png";
const int firstFrame = 0;
const int lastFrame = 482;
#endif
#if FILESET == 2
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street2/auto%05d.png";
const int firstFrame = 12400;
const int lastFrame = 12600;
#endif
#if FILESET == 3
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street1/auto%04d.jpg";
const int firstFrame = 1;
const int lastFrame = 2375;
#endif
