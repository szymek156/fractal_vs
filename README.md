# fractal_vs
![Rosetta stoned](https://github.com/szymek156/fractal_vs/blob/master/double_bench/rosetta_stoned.png)


Application for rendering Maldenbrot set using CUDA and OpenGL interoperability (output buffer is directly passed from CUDA to OpenGL)

Inspired by fractal zooms:
https://www.youtube.com/watch?v=pCpLWbHVNhk

## Features
* Zooming and panning
* Changing number of iterations during runtime (less iterations at the beginning, to zoom faster, increasing amount to get more details)
* CUDA and CPU versions
* CPU version paralelilzed over the cores
* CPU version has float custom implementation https://github.com/szymek156/fractal_vs/blob/master/fractal_vs_3/quadruple.cpp to increase fractal zoom resolution (doubled the "double" :) ). See https://github.com/szymek156/fractal_vs/blob/master/double_bench/2_comparsion_double_vs_doubledouble.png
* Rendering using OpenGL


The code is OLD, written in a style of a company as was working on at that time. My style changed thankfully.
