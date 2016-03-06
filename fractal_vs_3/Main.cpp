#include <stdlib.h>

#include "CommonUtil.h"
#include "CudaUtil.h"
#include "GLUtil.h"
#include "Fractal.h"

void clean()
{
  GLUtil::cleanGL();
  CudaUtil::cleanCuda();

  Fractal::cleanup();
}

int main(int argc, char *argv[])
{
  GLUtil::initGL(argc, argv /*1, (char **) (&"-gldebug")*/);

  CudaUtil::initCuda();

  Fractal::init();

  initLog();

  atexit(clean);

  GLUtil::startRender();

  return 0;
}
