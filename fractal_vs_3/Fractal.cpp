
#include "Fractal.h"

#include "CommonUtil.h"
#include "CudaUtil.h"
#include "GLUtil.h"
#include "FractalCPU.h"

#include <cmath>
#include <thread>
#include <vector>
#include <memory>

int Fractal::iterations_     = 50;
int *Fractal::hostBuffer_    = NULL;
int Fractal::hostBufferLength_ = 0;

std::unique_ptr<IFractal> Fractal::m_fractal = nullptr;
std::unique_ptr<IFractal> Fractal::m_fractal_quad = nullptr;

void Fractal::init()
{
  hostBufferLength_ = GLUtil::getTexWidth() * GLUtil::getTexHeight();

  hostBuffer_ = new int[hostBufferLength_];

  m_fractal = std::make_unique<FractalCPU>(hostBuffer_, hostBufferLength_, GLUtil::getTexWidth(), GLUtil::getTexHeight(), false);
  m_fractal_quad = std::make_unique<FractalCPU>(hostBuffer_, hostBufferLength_, GLUtil::getTexWidth(), GLUtil::getTexHeight(), true);
}

void Fractal::cleanup()
{
  if (hostBuffer_ != NULL)
  {
    delete [] hostBuffer_;

    hostBuffer_ = NULL;
  }
}

void Fractal::mandelbrotGPUNative(int partial)
{
  CudaUtil::mandelbrotGPU(partial, iterations_);
}

void Fractal::mandelbrotGPUQuad(int partial)
{

}

void Fractal::mandelbrotCPUNativeParallel(int partial)
{
	computeMandelbrotCPUNativeParallel(hostBuffer_, hostBufferLength_,
								GLUtil::getTexWidth(), GLUtil::getTexHeight(),
									GLUtil::getBounduary(), GLUtil::getMouseX(),
										GLUtil::getMouseY(), 0, 0, iterations_);


	CudaUtil::copyOnTexture(hostBuffer_, hostBufferLength_);
}

void Fractal::mandelbrotCPUNative(int partial)
{
	if (GLUtil::getQuad())
	{
		m_fractal_quad->render(GLUtil::getBounduary(), GLUtil::getMouseX(), GLUtil::getMouseY(), iterations_);
	}
	else
	{ 
		m_fractal->render(GLUtil::getBounduary(), GLUtil::getMouseX(), GLUtil::getMouseY(), iterations_);
	}
  

  CudaUtil::copyOnTexture(hostBuffer_, hostBufferLength_);
}

void Fractal::mandelbrotCPUQuad(int partial)
{
  computeMandelbrotCPUQuad(hostBuffer_, hostBufferLength_,
                                 GLUtil::getTexWidth(), GLUtil::getTexHeight(),
                                     GLUtil::getZoom(), GLUtil::getMouseX(),
                                         GLUtil::getMouseY(), 0, 0, iterations_);


  CudaUtil::copyOnTexture(hostBuffer_, hostBufferLength_);
}

int Fractal::colorRainbow(int counter)
{
  int i;
  float f, q, h;

  //
  // Remove that if.
  //

  if (counter < iterations_)
  {
    h = counter % 360;

    h /= 60;   // sector 0 to 5
    i = h;
    f = h - i; // factorial part of h
    q = 1 - f;

    f *= 255;
    q *= 255;

    int rArr[6] = {255, q, 0, 0, f, 255};

    int gArr[6] = {f, 255, 255, q, 0, 0};

    int bArr[6] = {0, 0, f, 255, 255, q};

    int color = (bArr[i] << 16) | (gArr[i] << 8) | rArr[i];

    return color;
  }

  return 0;
}

int Fractal::colorGray(int counter)
{
  if (counter < iterations_)
  {
    int luminance = (((int)(counter * (255 / (float) iterations_)) + 1));

    int color = (luminance << 16) | (luminance << 8) | (luminance << 0);

    return color;
  }

  return 0xffffff;
}

void Fractal::subTask(int id, int *buff, int size, int width, int height,
						double bounduary, double mouseX, double mouseY,
							int offsetWidth, int offsetHeight,
								int iterations)
{
	const int nrTrhreads = 8;

	int maxIterations = 0;
	double minX = 4.0;
	double maxX = -4.0;

	for (int i = id; i < height; i += nrTrhreads)
	{
		for (int j = 0; j < width; j++)
		{
			//
			// A(?)BGR
			//

			int color;

			if (j == (width / 2) || i == (height / 2))
			{
				color = 0xffffffff;
			}
			else
			{
				double xRatio = bounduary / width;
				double yRatio = bounduary / height;

				double x0 = (xRatio * j);
				double y0 = (yRatio * i);

				x0 += mouseX;
				y0 += mouseY;

				x0 -= bounduary / 2;
				y0 -= bounduary / 2;

				minX = std::fmin(x0, minX);
				maxX = std::fmax(x0, maxX);


				double x = 0;
				double y = 0;
				double xtemp = 0;
				int iteration = 0;

				double x2 = x * x;
				double y2 = y * y;

				while ((x2 + y2) < 3.0f && iteration < iterations)
				{
					xtemp = x2 - y2 + x0;

					y = (x + x) * y + y0;

					x = xtemp;

					x2 = x * x;

					y2 = y * y;

					++iteration;
				}

				if (iteration > maxIterations)
				{
					maxIterations = iteration;
				}

				color = colorRainbow(iteration);
			}
			//    color = colorGray(iteration);

			hostBuffer_[i * width + j] = color;
		}
	}

	//cout << "max iterations: " << maxIterations << "\n";

	if (id == 0)
	{
		cout << "minX " << minX << " maxX " << maxX << " boundary " << maxX - minX << "\n";
	}
}

void Fractal::computeMandelbrotCPUNativeParallel(int *buff, int size, int width, int height,
													double bounduary, double mouseX, double mouseY,
														int offsetWidth, int offsetHeight,
															int iterations)
{
	unsigned int n = std::thread::hardware_concurrency();

	std::vector<std::unique_ptr<std::thread>> threads(n);
	int i = 0;

	for (auto &thread : threads)
	{
		thread = std::make_unique<std::thread>(Fractal::subTask, i++, buff, size, width, height, 
					bounduary, mouseX, mouseY, offsetWidth, offsetHeight, iterations);
	}

	for (auto &thread : threads)
	{
		thread->join();
	}

}

void Fractal::computeMandelbrotCPUQuad(int *buff, int size, int width, int height,
                                            double ratiod, double mouseX, double mouseY,
                                                int offsetWidth, int offsetHeight,
                                                    int iterations)
{
	/*
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      //
      // A(?)BGR
      //

      int color;

      quad ratio = quadInit(ratiod);

      quad boundary = quadMul(quadInit(width >> 1), ratio);

      quad x0 = quadSub(quadMul(ratio, quadInit(j)), boundary);

      quad y0 = quadAdd(quadMul(quadNeg(ratio), quadInit(i)), boundary);

      x0 = quadAdd(x0, quadInit(mouseX));
      y0 = quadAdd(y0, quadInit(mouseY));

      quad x       = {0, 0};
      quad y       = {0, 0};
      quad xtemp   = {0, 0};
      int iteration = 0;

      quad x2 = quadMul(x, x);

      quad y2 = quadMul(y, y);

      while (quadLess(quadAdd(x2, y2), quadInit(4.0)) && iteration < iterations)
      {
        xtemp = quadAdd(quadSub(x2, y2), x0);

        y = quadAdd(quadMul((quadAdd(x, x)), y), y0);

        x = xtemp;

        x2 = quadMul(x, x);

        y2 = quadMul(y, y);

        ++iteration;
      }

      color = colorRainbow(iteration);

      //color = colorGray(iteration);

      buff[i * width + j] = color;
    }
  }
  */
}

void Fractal::setIterations(int iterations)
{
  if (iterations > 0)
  {
    iterations_ = iterations;
  }
  else
  {
    iterations_ = 1;
  }

  cout << "iterations set to: " << iterations_ << "\n";
}

int Fractal::getIterations()
{
  return iterations_;
}
