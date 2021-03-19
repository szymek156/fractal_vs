# fractal_vs
![Rosetta stoned](https://github.com/szymek156/fractal_vs/blob/master/double_bench/rosetta_stoned.png)


Application for rendering Maldenbrot set using CUDA and OpenGL interoperability (output buffer is directly passed from CUDA to OpenGL)

## Features
* Zooming and panning
* CUDA and CPU versions
* CPU version paralelilzed over the cores
* CPU version has float custom implementation https://github.com/szymek156/fractal_vs/blob/master/fractal_vs_3/quadruple2.cpp to increase fractal zoom resolution (doubled the "double" :) ). See https://github.com/szymek156/fractal_vs/blob/master/double_bench/2_comparsion_double_vs_doubledouble.png
