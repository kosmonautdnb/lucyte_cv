# Lucyte with OpenCV
by Stefan Mader

A novel feature/point tracking approach that uses the descriptors to locate the features in an image. Ensuring luminance invariance.

Demo: https://www.youtube.com/watch?v=A_0tm-1DTm4

Simple scene: https://www.youtube.com/watch?v=3C6wPafwNTQ

Lucyte Algorithm vs. Optical Flow(KLT) by Kanade Lucas Tomasi: https://www.youtube.com/watch?v=pYw6Db9fh_Y

Complicated light with Lucyte vs. KLT: https://www.youtube.com/watch?v=P3CnGRe1MFs

Luminance agnostic demo: https://www.youtube.com/watch?v=FZHaOj7gu7E

Object tracking base: https://www.youtube.com/watch?v=Vfx42kb2aPM

Stereo Vision Disparity/Depth: https://www.youtube.com/watch?v=sh3JB8mnVaY

![ScreenShot](https://raw.github.com/kosmonautdnb/lucyte_cv/main/desc.png)

Lucyte is not yet very fast even with OpenCL. However, most feature tracking tasks can take larger time steps and may be executed in parallel on dedicated hardware.
