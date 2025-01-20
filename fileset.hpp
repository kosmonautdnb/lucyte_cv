// Created on: 17.01.2025 by Stefan Mader
#pragma once

const char* outputVideoFileName = "c:/!mad/video.avi";
const char* outputBenchmarkVideoFileName = "c:/!mad/bench_video.avi";
const double outputVideoFrameRate = 25;

#define FILESET 7

#if FILESET == 1
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street3/wandern%04d.png";
const int firstFrame = 0;
const int lastFrame = 482;
const int frameStep = 1;
#endif
#if FILESET == 2
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street2/auto%05d.png";
const int firstFrame = 12400;
const int lastFrame = 12600;
const int frameStep = 1;
#endif
#if FILESET == 3
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street1/auto%04d.jpg";
const int firstFrame = 1;
const int lastFrame = 2375;
const int frameStep = 1;
#endif
#if FILESET == 4
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street4/auto%04d.png";
const int firstFrame = 1;
const int lastFrame = 1627;
const int frameStep = 1;
#endif
#if FILESET == 5
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street5/auto%04d.jpg";
const int firstFrame = 5200;
const int lastFrame = 7700;
const int frameStep = 1;
#endif
#if FILESET == 6
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street6/auto%04d.jpg";
const int firstFrame = 1600;
const int lastFrame = 7284;
const int frameStep = 1;
#endif
#if FILESET == 7
const char* fileNames = "C:/!mad/Daten/Odometry/DataSets/Selection/DancingTheatreGirl/dancing%04d.png ";
const int firstFrame = 2;
const int lastFrame = 509;
const int frameStep = 2;
#endif
