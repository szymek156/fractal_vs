#include "FractalCPU.h"
#include "quadruple.h"
#include "quadruple2.h"
#include <iostream>
#include <string>

FractalCPU::FractalCPU(int *outBuffer, int outBufferLength, int width, int height, bool quad) : 
	IFractal(outBuffer, outBufferLength, width, height), m_keepRunning(true), m_colorizer(0), m_finishedJob(0)
{
	m_threadsCount = std::min(std::thread::hardware_concurrency(), MAX_THREADS);

	for (int i = 0; i < m_threadsCount; i++)
	{
		m_startJob[i] = false;

		if (quad)
		{
			//m_threads[i] = std::make_unique<std::thread>(FractalCPU::subTask<quadruple>, this, i);
			m_threads[i] = std::make_unique<std::thread>(FractalCPU::subTask<quadruple2>, this, i);
		}
		else
		{
			m_threads[i] = std::make_unique<std::thread>(FractalCPU::subTask<double>, this, i);
		}
	}
	
}

FractalCPU::~FractalCPU()
{
	m_keepRunning = false;
	m_workToDo.notify_all();

	for (int i = 0; i < m_threadsCount; i++)
	{
		m_threads[i]->join();
	}
}

void FractalCPU::render(double peepholeSize, double centerX, double centerY, int iterations)
{
	{
		std::unique_lock<std::mutex> hold(m_workMutex);

		m_jobData.peepholeSize = peepholeSize;
		m_jobData.centerX = centerX;
		m_jobData.centerY = centerY;
		m_jobData.iterations = iterations;

		m_colorizer = Colorizer(iterations);

		for (int i = 0; i < m_threadsCount; i++)
		{
			m_startJob[i] = true;
		}

		m_finishedJob = m_threadsCount;
	}

	m_workToDo.notify_all();

	{
		std::unique_lock<std::mutex> hold(m_workMutex);
		m_workToDo.wait(hold, [this] {return m_finishedJob == 0; });
	}
}

template <class T>
static void FractalCPU::subTask(FractalCPU *context, int id)
{
	while (true)
	{
		{
			std::unique_lock<std::mutex> hold(context->m_workMutex);
			context->m_workToDo.wait(hold, [context, id] { return context->m_startJob[id] == true; });

			// This causes deadlock, (because of race condition):
			//context->m_workToDo.wait(hold, [context] { return context->m_finishedJob == context->m_threadsCount; });
			// intention: hold all threads here until producer will not set finishedJob to threadsCount, then producer will notify all threads
			// result: notify all works sequentially, it wakes up threads one after another, so if thread 1 starts job and finishes it BEFORE 
			// thread 2 will be waken up, then th1 will decrement m_finishedJob, th2 is waked up, condition is not true in lambda - goes to sleep
			// causes deadlock.
			// So there is a race condition between threads on m_finishedJob variable

			if (context->m_keepRunning == false)
			{	
				break;
			}
		}


	//	{
	//		std::unique_lock<std::mutex> hold(context->m_workMutex);
	//		std::cout << "thread: " << id << " started \n";
	//	}
		

		T xRatio = context->m_jobData.peepholeSize / context->m_width;
		T yRatio = context->m_jobData.peepholeSize / context->m_height;

		T centeringX = context->m_jobData.centerX - (context->m_jobData.peepholeSize / (2.0));
		T centeringY = context->m_jobData.centerY - (context->m_jobData.peepholeSize / (2.0));

		for (int i = id; i < context->m_height; i += context->m_threadsCount)
		{
			T y0 = (yRatio * T((float)i));
			y0 = y0 + centeringY;

			for (int j = 0; j < context->m_width; j++)
			{
			/*	if (j == context->m_width / 2 || i == context->m_height / 2)
				{
					context->m_outBuffer[i * context->m_width + j] = 0xffffffff;
					continue;
				}*/

				T x0 = (xRatio * T((float)j));
				x0 = x0 + centeringX;

				T x = 0.0f;
				T y = 0.0f;
				int iteration = 0;

				T x2 = 0.0f;
				T y2 = 0.0f;
				T sum = 0.0f;

				while (sum < T(4.0) && iteration < context->m_jobData.iterations)
				{
					y = (x + x) * y + y0;

					x = x2 - y2 + x0;

					x2 = x * x;

					y2 = y * y;

					sum = (x2 + y2);

					++iteration;
				}

				context->m_outBuffer[i * context->m_width + j] = context->m_colorizer.colorRainbow(iteration);
			}
		}

		{	
			std::unique_lock<std::mutex> hold(context->m_workMutex);
			context->m_startJob[id] = false;

			--(context->m_finishedJob);

			context->m_workToDo.notify_all();
		}

	}

	std::cout << "thread : " << id << " says goodbye\n";
}
