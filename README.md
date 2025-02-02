# Lucyte with OpenCV
by Stefan Mader

A novel feature/point tracking approach that uses the descriptors to locate the features in an image. Ensuring luminance invariance.
Lucyte can run fully parallel, e.g. with OpenCL, but this is not yet supported.

Actually this repository should just hint at this basic algorithm, that can be found in refinement.cpp.

## Demo videos

Day drive: https://www.youtube.com/watch?v=6K82WeEhKBI

Night drive: https://www.youtube.com/watch?v=A_0tm-1DTm4

Lucyte Algorithm vs. Optical Flow(KLT) by Kanade Lucas Tomasi: https://www.youtube.com/watch?v=pYw6Db9fh_Y

Complicated light with Lucyte vs. KLT: https://www.youtube.com/watch?v=P3CnGRe1MFs

Luminance agnostic demo: https://www.youtube.com/watch?v=FZHaOj7gu7E

Object tracking base: https://www.youtube.com/watch?v=Vfx42kb2aPM

Stereo Vision Disparity/Depth: https://www.youtube.com/watch?v=sh3JB8mnVaY

Lucyte with KITTIE benchmark (speed is read from file): https://www.youtube.com/watch?v=C-9sCdJsaTE 

![ScreenShot](https://raw.github.com/kosmonautdnb/lucyte_cv/main/desc.png)
