#pragma once
#include "IFractal.h"
#include "Colorizer.h"
#include <memory>
#include <array>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

class FractalCPU : public IFractal
{
public:
	FractalCPU(int *outBuffer, int outBufferLength, int width, int height, bool quad);
	virtual ~FractalCPU();

	virtual void render(double peepholeSize, double centerX, double centerY, int iterations);

	template <class T>
	static void subTask(FractalCPU *context, int id);

protected:
	static const unsigned int MAX_THREADS = 32;

	std::array<std::unique_ptr<std::thread>, MAX_THREADS> m_threads;

	std::array<bool, MAX_THREADS> m_startJob;

	int m_threadsCount;

	bool m_keepRunning;

	struct JobData
	{
		JobData() : peepholeSize(0.0), centerX(0.0), centerY(0.0), iterations(0)
		{}

		double peepholeSize;
		double centerX;
		double centerY;
		int iterations;
	};

	JobData m_jobData;
	Colorizer m_colorizer;

	std::mutex m_workMutex;
	std::condition_variable m_workToDo;
	std::atomic<int> m_finishedJob;
};

