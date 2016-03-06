#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

#include <cuda_runtime.h>

class CudaUtil
{
  public:
    CudaUtil();
    virtual ~CudaUtil();

    static int initCuda();

    static void cleanCuda();

    static void mandelbrotGPU(int partial, int iterations);

    static void copyOnTexture(int *hostBuffer, int size);

    static cudaEvent_t timeStart();

    //
    // Returns difference in ms and cleans event resources.
    //

    static float timeDiffMs(cudaEvent_t start);

    static int getBufferSize();

  private:

//    static struct cudaGraphicsResource *resource_;
};

#endif /* CUDAUTIL_H_ */
