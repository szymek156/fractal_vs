#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>

#include "CommonUtil.h"
#include "GLUtil.h"
#include "CudaUtil.h"

extern "C" void process(int *cuda, int size, int width, int height, double ratio,
                            double mouseX, double mouseY, int iterations);

extern "C" void processPartial(int *buffCuda, int buffSize, int width, int height,
                                   double ratio, double mouseX, double mouseY,
                                       int offsetWidth, int offsetHeight,
                                           dim3 *grid, dim3 *block, int iterations);

struct cudaGraphicsResource *resource_;

int *bufferCuda_;
int bufferSize_;

CudaUtil::CudaUtil()
{
}

CudaUtil::~CudaUtil()
{
}

void CudaUtil::cleanCuda()
{
  cudaGraphicsUnregisterResource(resource_);

  if (bufferCuda_ != NULL)
  {
    cudaFree(bufferCuda_);

    bufferCuda_ = NULL;
  }
}

int CudaUtil::initCuda()
{
  cudaError_t result = cudaSuccess;

  result = cudaGLSetGLDevice(0);

  GLuint texId = GLUtil::getTextureId();

  result = cudaGraphicsGLRegisterImage(
               &resource_, texId, GL_TEXTURE_2D,
                   cudaGraphicsMapFlagsNone);

  bufferSize_ = GLUtil::getTexWidth() * GLUtil::getTexHeight() * 4 * sizeof(GLubyte);

  result = cudaMalloc(&bufferCuda_, bufferSize_);

  return result;
}

void CudaUtil::mandelbrotGPU(int partial, int iterations)
{
  cudaEvent_t start;
  float elapsed;

  start = timeStart();

  cudaArray_t array;

  cudaError_t result = cudaSuccess;

  result = cudaGraphicsMapResources(1, &resource_, 0);

  result = cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0);

  int width = GLUtil::getTexWidth();

  int height = GLUtil::getTexHeight();

  if (partial == 1)
  {
    int division = 4;

    int blockDim = 32;

    dim3 grid((width / (float) blockDim + 0.5f), (height / (float) blockDim + 0.5f), 1);

    grid.x = grid.x / division;
    grid.y = grid.y / division;

    dim3 block(blockDim, blockDim, 1);

    for (int i = 0; i < division; i++)
    {
      int offsetHeight = i * (grid.x * block.x);

      for (int j = 0; j < division; j++)
      {
        int offsetWidth = j * (grid.y * block.y);

        #ifdef TEST
        cout << "mandelbrotGPU: Computing part [" << i << ", " << j << "] "
             << "from [" << division - 1 << ", " << division - 1 << "].\n";
        #endif

        processPartial(bufferCuda_, bufferSize_, width, height,
                           GLUtil::getBounduary(), GLUtil::getMouseX(), GLUtil::getMouseY(),
                                 offsetWidth, offsetHeight, &grid, &block, iterations);
      }
    }
  }
  else
  {
    process(bufferCuda_, bufferSize_, height, width,
		GLUtil::getBounduary(), GLUtil::getMouseX(), GLUtil::getMouseY(),
                    iterations);
  }

  result = cudaMemcpyToArray(array, 0, 0, bufferCuda_, bufferSize_, cudaMemcpyDeviceToDevice);

  cudaGraphicsUnmapResources(1, &resource_, 0);

  elapsed = timeDiffMs(start);

  cout << "Kernel execution time: " << elapsed << "ms.\n";
}

void CudaUtil::copyOnTexture(int *hostBuffer, int size)
{
  cudaError_t result = cudaSuccess;

  cudaArray_t array;

  result = cudaGraphicsMapResources(1, &resource_, 0);

  result = cudaGraphicsSubResourceGetMappedArray(&array, resource_, 0, 0);

  result = cudaMemcpy(bufferCuda_, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);

  result = cudaMemcpyToArray(array, 0, 0, bufferCuda_, bufferSize_, cudaMemcpyDeviceToDevice);

  cudaGraphicsUnmapResources(1, &resource_, 0);

}

int CudaUtil::getBufferSize()
{
  return bufferSize_;
}

cudaEvent_t CudaUtil::timeStart()
{
  cudaEvent_t start;

  cudaEventCreate(&start);
  cudaEventRecord(start,0);

  return start;
}

float CudaUtil::timeDiffMs(cudaEvent_t start)
{
  cudaEvent_t stop;

  float elapsed;

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed, start,stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed;
}
