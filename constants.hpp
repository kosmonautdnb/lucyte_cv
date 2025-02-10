// Lucyte Created on: 29.01.2025 by Stefan Mader
#pragma once

#define __PLATFORM_WINDOWS__ 1
#define __PLATFORM_ANDROID__ 2
#define __PLATFORM_IOS__ 3
#if __ANDROID__ == 1
#define __CURRENT_PLATFORM__ __PLATFORM_ANDROID__
#else
#define __CURRENT_PLATFORM__ __PLATFORM_WINDOWS__
#endif

#define lcLog printf
#define KEYPOINTCOUNT 10000
