

#ifndef COMMONUTIL_H_
#define COMMONUTIL_H_

#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <cstddef>
//#include <sys/time.h>

using namespace std;

void startWatch();

void stopWatch();

uint64_t getTimeDiffMs();

void initLog();


#endif /* COMMONUTIL_H_ */
