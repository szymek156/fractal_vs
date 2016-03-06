#ifndef Fractal_H
#define Fractal_H

#include "IFractal.h"

#include <memory>

class Fractal
{
  public:
  static void init();

  static void cleanup();

  static void mandelbrotGPUNative(int partial);

  static void mandelbrotGPUQuad(int partial);

  static void mandelbrotCPUNativeParallel(int partial);

  static void mandelbrotCPUNative(int partial);

  static void mandelbrotCPUQuad(int partial);

  static void setIterations(int iterations);

  static int getIterations();

  private:

  static void subTask(int id, int *buff, int size, int width, int height,
						double bounduary, double mouseX, double mouseY,
							int offsetWidth, int offsetHeight,
								int iterations);

  static void computeMandelbrotCPUNativeParallel(int *buff, int size, int width, int height,
													double ratio, double mouseX, double mouseY,
														int offsetWidth, int offsetHeight,
															int iterations);

  static void computeMandelbrotCPUQuad(int *buff, int size, int width, int height,
                                            double ratio, double mouseX, double mouseY,
                                                int offsetWidth, int offsetHeight,
                                                    int iterations);

  static int colorRainbow(int counter);
  static int colorGray(int counter);

  static int iterations_;

  static int *hostBuffer_;

  static int hostBufferLength_;

  static std::unique_ptr<IFractal> m_fractal;
  static std::unique_ptr<IFractal> m_fractal_quad;
};

#endif // Fractal_H
