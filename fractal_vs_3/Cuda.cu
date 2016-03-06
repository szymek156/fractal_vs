/*
TODO:
- probably texture to screen mapping is incorrect - navigating doesnt work if res changed
- create CPU version multithreaded
    threads: DONE
	async
- doublepass rendering, first get nr of iterations from every pixel, then according to neighbours set appropiate kolor (some blur for example)

double-double (quadruple2) on GPU

use implementation of double^4 from QD package, there is cool publication regarding implementation of algorithms:
http://crd-legacy.lbl.gov/~dhbailey/mpdist/

- implement perturbation algorithm: http://www.superfractalthing.co.nf/sft_maths.pdf

fix somehow navigation...

2) Find optimal work group size.
3) Divide rendering on several kernel calls - works on group size 16.
5) Optimizations.
7) Add other extended arithmetic (based on IEEE standard) as a struct with int fields
9) Dump render to file / movie.
10) Get rid of copy between gl and cuda resources...
11) Add Julia fractal, and generalization of maldenbrot.
12) Cleanup code.
*/


int __device__ Bonduary;

//
// Converts counter to H component in HSV color model.
// Makes rainbow palette.
//

int __device__ colorRainbow(int counter)
{
  int i;
  float f, q, h;

  //
  // Remove that if.
  //

  if (counter < Bonduary)
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

//    h /= 60;      // sector 0 to 5
//    i = h;
//    f = h - i;      // factorial part of h
//    q = ( 1 - f );
//
//    float rArr[6] = {1, q, 0, 0, f, 1};
//
//    float gArr[6] = {f, 1, 1, q, 0, 0};
//
//    float bArr[6] = {0, 0, f, 1, 1, q};
//
//    int color = (((int)(bArr[i] * 255)) << 16) | (((int)(gArr[i] * 255)) << 8) |
//                    ((int)(rArr[i] * 255));
//    switch( i )
//    {
//      case 0:  r = 1; g = f; b = 0;
//        break;
//      case 1:  r = q; g = 1; b = 0;
//        break;
//      case 2:  r = 0; g = 1; b = f;
//        break;
//      case 3:  r = 0; g = q; b = 1;
//        break;
//      case 4:  r = f; g = 0; b = 1;
//        break;
//      case 5:
//      default: r = 1; g = 0; b = q;
//        break;
//    }
//
//    int color = (((int)(b * 255)) << 16) | (((int)(g * 255)) << 8) | ((int)(r * 255));

    return color;
  }

  return 0;
}

int __device__ colorGray(int counter)
{
  if (counter < Bonduary)
  {
    int luminance = (((int)(counter * (255 / (float) Bonduary)) + 1));

    int color = (luminance << 16) | (luminance << 8) | (luminance << 0);

    return color;
  }

  return 0xffffff;
}

void __global__ mandelbrotNative(int *buff, int size, int width, int height,
                                 float ratio, float mouseX, float mouseY,
                                     int offsetWidth, int offsetHeight,
                                         int iterations)
{
  Bonduary = iterations;

  int thX = (blockIdx.x * blockDim.x + threadIdx.x) + offsetWidth;
  int thY = (blockIdx.y * blockDim.y + threadIdx.y) + offsetHeight;

  if (thX < width && thY < height)
  {
	  if (thX == width >> 1 || thY == height >> 1)
	  {
		  buff[thY * width + thX] = 0;
		  return;
	  }
    //
    // A(?)BGR
    //

    int color;

    double boundary = (width >> 1) * ratio;

    double x0 = ( ratio * thX) - boundary;
	double y0 = (-ratio * thY) + boundary;

    x0 += mouseX;
    y0 += mouseY;

	double x = 0;
	double y = 0;
	double xtemp = 0;
    int iteration = 0;

	double x2 = x * x;
	double y2 = y * y;

    while ((x2 + y2) < 4.0 && iteration < Bonduary)
    {
      xtemp = x2 - y2 + x0;

      y = (x + x) * y + y0;

      x = xtemp;

      x2 = x * x;

      y2 = y * y;

      ++iteration;
    }

    color = colorRainbow(iteration);
	//color = colorGray(iteration);

    

    buff[thY * width + thX] = color;
  }
}

//
// Quad (double double) precision. Compiler will
// convert components to float automatically on
// older devices.
//
// http://hal.archives-ouvertes.fr/docs/00/06/33/56/PDF/float-float.pdf
// http://andrewthall.org/papers/df64_qf128.pdf
//

typedef struct
{
    double hi;
    double lo;
} quad;

typedef struct
{
    float hi;
    float lo;
} doubleff;
//
////
//// For float p = 24, double p = 53 ((2 << 27) + 1).
////
//
//#if defined(__CUDA_ARCH__)
//  // Device code trajectory.
//
//  #if __CUDA_ARCH__ > 120
//    #warning Compiling with double precision.
//
//    #define SPLIT ((2 << 27) + 1)
//
//  #else
//    #warning Compiling with single precision.
//
//    #define SPLIT ((2 << 12) + 1)
//  #endif
//
//#else
//  // Host code trajectory.
//
//  #define SPLIT ((2 << 12) + 1)
//#endif
//
//quad __device__ __host__ quadInit(double a)
//{
//  quad result;
//
//  double c = SPLIT * a;
//
//  double aBig = c - a;
//
//  result.hi = c - aBig;
//
//  result.lo = a - result.hi;
//
//  return result;
//}
//
////
//// Must be used on host side in case of older graphics card.
////
//
//doubleff __device__ __host__ doubleFFInit(double a)
//{
//  doubleff result;
//
//  double c = SPLIT * a;
//
//  double aBig = c - a;
//
//  result.hi = c - aBig;
//
//  result.lo = a - result.hi;
//
//  return result;
//}
//
//quad __device__ quadNeg(quad a)
//{
//  quad result;
//
//  result.hi = a.hi * (-1.0);
//  result.lo = a.lo * (-1.0);
//
//  return result;
//}
//
//quad __device__ add12(double a, double b)
//{
//  quad result;
//
//  double s = a + b;
//
//  double v = s - a;
//
//  double r = (a - (s - v)) + (b - v);
//
//  result.hi = s;
//
//  result.lo = r;
//
//  return result;
//}
//
//quad __device__ mul12(double a, double b)
//{
//  quad result;
//
//  result.hi = a * b;
//
//  quad aquad = quadInit(a);
//
//  quad bquad = quadInit(b);
//
//  float err1 = result.hi - (aquad.hi * bquad.hi);
//
//  float err2 = err1 - (aquad.lo * bquad.hi);
//
//  float err3 = err2 - (aquad.hi * bquad.lo);
//
//  result.lo = (aquad.lo * bquad.lo) - err3;
//
//  return result;
//}
//
//quad __device__ quadAdd(quad a, quad b)
//{
///*
//FIXME: Implement version without a branch.
//*/
//  quad result;
//
//  double r = a.hi + b.hi;
//
//  double s;
//
//  if (abs(a.hi) >= abs(b.hi))
//  {
//    s = (((a.hi - r) + b.hi) + b.lo) + a.lo;
//  }
//  else
//  {
//    s = (((b.hi - r) + a.hi) + a.lo) + b.lo;
//  }
//
//  result = add12(r, s);
//
//  return result;
//}
//
////
//// Quad-Double Arithmetic: Algorithms, Implementation, and Application:
//// Substraction (a - b) is implemented as addition a + (-b). To negate
//// a quad-double number we can just simply negate each component.
////
//
//quad __device__ quadSub(quad a, quad b)
//{
//  return quadAdd(a, quadNeg(b));
//}
//
//quad __device__ quadMul(quad a, quad b)
//{
//  quad result;
//
//  quad abh = mul12(a.hi, b.hi);
//
//  float t3 = ((a.hi * b.lo) * (a.lo * b.hi)) + abh.lo;
//
//  //
//  // Error in publication, not sure how t3 should be used?
//  //
//
//  result = add12(abh.hi, t3);
//
//  return result;
//}
//
//bool __device__ quadLess(quad a, quad b)
//{
//  return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
//}

void __global__ mandelbrotQuad(int *buff, int size, int width, int height,
                                   doubleff ffratio, float mouseX, float mouseY,
                                       int offsetWidth, int offsetHeight,
                                           int iterations)
{
//  Bonduary = iterations;
//
//  int thX = (blockIdx.x * blockDim.x + threadIdx.x) + offsetWidth;
//  int thY = (blockIdx.y * blockDim.y + threadIdx.y) + offsetHeight;
//
//  if (thX < width && thY < height)
//  {
//    //
//    // A(?)BGR
//    //
//
//    int color;
//
//    quad ratio = {ffratio.hi, ffratio.lo};
//
//    quad boundary = quadMul(quadInit(width >> 1), ratio);
//
//    quad x0 = quadSub(quadMul(ratio, quadInit(thX)), boundary);
//
//    quad y0 = quadAdd(quadMul(quadNeg(ratio), quadInit(thY)), boundary);
//
//    x0 = quadAdd(x0, quadInit(mouseX));
//    y0 = quadAdd(y0, quadInit(mouseY));
//
//    quad x       = {0, 0};
//    quad y       = {0, 0};
//    quad xtemp   = {0, 0};
//    int iteration = 0;
//
//    quad x2 = quadMul(x, x);
//
//    quad y2 = quadMul(y, y);
//
//    while (quadLess(quadAdd(x2, y2), quadInit(4.0)) && iteration < Bonduary)
//    {
//      xtemp = quadAdd(quadSub(x2, y2), x0);
//
//      y = quadAdd(quadMul((quadAdd(x, x)), y), y0);
//
//      x = xtemp;
//
//      x2 = quadMul(x, x);
//
//      y2 = quadMul(y, y);
//
//      ++iteration;
//    }
//
////    color = colorRainbow(iteration);
//
//    color = colorGray(iteration);
//
//    buff[thY * width + thX] = color;
//  }
}

extern "C" void process(int *buffCuda, int buffSize, int width, int height,
                            double ratio, float mouseX, float mouseY,
                                int iterations)
{
  //
  // 512 is the max number of threads per group.
  //

  int blockDim = 22;

  dim3 grid((width / (float) blockDim + 0.5f), (height / (float) blockDim + 0.5f), 1);

  dim3 block(blockDim, blockDim, 1);

//  doubleff quadRatio = doubleFFInit(ratio);

//  mandelbrotQuad <<<grid, block, 0>>> (buffCuda, buffSize, width, height,
//                                           quadRatio, mouseX, mouseY, 0, 0);

  mandelbrotNative<<<grid, block, 0>>> (buffCuda, buffSize, width, height,
                                            ratio, mouseX, mouseY, 0, 0, iterations);
}

extern "C" void processPartial(int *buffCuda, int buffSize, int width, int height,
                                   double ratio, float mouseX, float mouseY,
                                       int offsetWidth, int offsetHeight,
                                           dim3 *grid, dim3 *block,
                                               int iterations)
{

//  doubleff quadRatio = doubleFFInit(ratio);

  mandelbrotNative<<<*grid, *block, 0>>> (buffCuda, buffSize, width, height,
                                            ratio, mouseX, mouseY,
                                                offsetWidth, offsetHeight,
                                                    iterations);
}
