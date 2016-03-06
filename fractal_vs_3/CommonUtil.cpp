#include "CommonUtil.h"

#include <limits>
#include <chrono>

#undef DEBUG

std::chrono::time_point<std::chrono::system_clock> start;
std::chrono::time_point<std::chrono::system_clock> stop;

void initLog()
{
  cout.precision(std::numeric_limits<double>::digits10);
}

void startWatch()
{
	start = std::chrono::system_clock::now();
}

void stopWatch()
{
	stop = std::chrono::system_clock::now();
}

uint64_t getTimeDiffMs()
{
	
	auto elapsed_millis = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	
	return elapsed_millis.count();

  #ifdef DEBUG
  cout << "getTimeDiffMs: b: " << b << " e: " << e << endl;
  #endif

  //if (e > b)
 // {
 //   return e - b;
 // }

  return 0;
}
