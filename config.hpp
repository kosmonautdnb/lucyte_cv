// Lucyte Created on: 17.01.2025 by Stefan Mader
#pragma once

#include "constants.hpp"

#define FILESET 3
#define STEREO_FILESET 3
const char* outputVideoFileName = "c:/!mad/video.avi";
const char* outputBenchmarkVideoFileName = "c:/!mad/bench_video.avi";
const char* benchmarkPictureFileName = "c:/!mad/bench.png";
const double outputVideoFrameRate = 25;
const double stereoOutputVideoFrameRate = 8;
const bool outputVideo = false;

#define CONFIG_GOOD 0
#define CONFIG_FAST 1
#define CONFIG_OK 2
#define CONFIG_FAST_MANYMIPS 3
#define CONFIG CONFIG_GOOD

#if CONFIG == 0
float MIPSCALE = 0.5f;
float SCALEINVARIANCE = (0.5f / 2.f); // 0.5 / 2 is good
float ROTATIONINVARIANCE = (20.f / 2.f); // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
int STEPCOUNT = 100;
float STEPSIZE = 0.002f;
float DESCRIPTORSCALE = 5.f;
bool BOOLSTEPPING = false;
#endif

#if CONFIG == 1
float MIPSCALE = 0.5f;
float SCALEINVARIANCE = (0.5f / 2.f); // 0.5 / 2 is good
float ROTATIONINVARIANCE = (20.f / 2.f); // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
int STEPCOUNT = 25;
float STEPSIZE = 0.002f;
float DESCRIPTORSCALE = 5.f;
bool BOOLSTEPPING = false;
#endif

#if CONFIG == 2
float MIPSCALE = 0.5f;
float SCALEINVARIANCE = (0.5f / 2.f); // 0.5 / 2 is good
float ROTATIONINVARIANCE = (20.f / 2.f); // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
int STEPCOUNT = 50;
float STEPSIZE = 0.004f;
float DESCRIPTORSCALE = 5.f;
bool BOOLSTEPPING = false;
#endif

#if CONFIG == 3
float MIPSCALE = 0.86f;
float SCALEINVARIANCE = (0.5f / 2.f); // 0.5 / 2 is good
float ROTATIONINVARIANCE = (20.f / 2.f); // 45.f / 2 is good you may do 8 separate sampled versions to get full cirlce to 360 degrees
int STEPCOUNT = 12;
float STEPSIZE = 0.0035f;
float DESCRIPTORSCALE = 5.f;
bool BOOLSTEPPING = false;
#endif


#if STEREO_FILESET == 1
const char* stereoFileNamesLeft = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/left/2018-07-11-14-48-52/left%04d.jpg";
const char* stereoFileNamesRight = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/right/2018-07-11-14-48-52/right%04d.jpg";
const int stereoFirstFrame = 1;
const int stereoLastFrame = 1720;
const int stereoFrameStep = 1;
#endif
#if STEREO_FILESET == 2
const char* stereoFileNamesLeft = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/left/2018-08-01-11-13-14/left%04d.jpg";
const char* stereoFileNamesRight = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/right/2018-08-01-11-13-14/right%04d.jpg";
const int stereoFirstFrame = 1;
const int stereoLastFrame = 1532;
const int stereoFrameStep = 1;
#endif
#if STEREO_FILESET == 3
const char* stereoFileNamesLeft = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/left/2018-10-11-16-03-19/left%04d.jpg";
const char* stereoFileNamesRight = "c:/!mad/Daten/Odometry/www.drivingstereo-dataset.github.io/right/2018-10-11-16-03-19/right%04d.jpg";
const int stereoFirstFrame = 1;
const int stereoLastFrame = 1637;
const int stereoFrameStep = 1;
#endif
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
const int frameStep = 10;
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
const char* fileNames = "C:/!mad/Daten/Odometry/DataSets/Selection/DancingTheatreGirl/dancing%04d.png";
const int firstFrame = 2;
const int lastFrame = 509;
const int frameStep = 2;
#endif
#if FILESET == 8
const char* fileNames = "C:/!mad/Daten/Odometry/DataSets/Selection/kpopgirl_dancing1/dancing%04d.png";
const int firstFrame = 1;
const int lastFrame = 231;
const int frameStep = 1;
#endif
#if FILESET == 9
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street7/auto%04d.jpg";
const int firstFrame = 0;
const int lastFrame = 2162;
const int frameStep = 5;
#endif
#if FILESET == 10
const char* fileNames = "c:/!mad/Daten/Odometry/DataSets/Selection/schwebegirl/schwebegirl%04d.png";
const int firstFrame = 0;
const int lastFrame = 275;
const int frameStep = 1;
#endif
#if FILESET == 11
const char* fileNames = "c:/!mad/Daten/Odometry/DataSets/Selection/!!!!anime_girls_dancing/dancing2%04d.png";
const int firstFrame = 1;
const int lastFrame = 211;
const int frameStep = 1;
#endif
#if FILESET == 12
const char* fileNames = "c:/!mad/Daten/Odometry/DataSets/Selection/dance/dance%04d.jpg";
const int firstFrame = 1;
const int lastFrame = 210;
const int frameStep = 1;
#endif
#if FILESET == 13
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street8/auto%04d.jpg";
const int firstFrame = 0;
const int lastFrame = 2162;
const int frameStep = 5;
#endif
#if FILESET == 14
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street10/auto%04d.jpg";
const int firstFrame = 1000;
const int lastFrame = 7000;
const int frameStep = 5;
#endif
#if FILESET == 15
const char* fileNames = "c:/!mad/Daten/Odometry/STREETDRIVES/Street11/walk%04d.jpg";
const int firstFrame = 0;
const int lastFrame = 720;
const int frameStep = 1;
#endif