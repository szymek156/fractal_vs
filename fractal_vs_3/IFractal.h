
#pragma once

class IFractal
{
public:
	IFractal(int *outBuffer, int outBufferLength, int width, int height) : 
		m_outBuffer(outBuffer), m_outBufferLength(outBufferLength), m_width(width), m_height(height)
	{

	}

	virtual ~IFractal(){}

	virtual void render(double peepholeSize, double centerX, double centerY, int iterations) = 0;

protected:
	int *m_outBuffer;
	int m_outBufferLength;
	int m_width;
	int m_height;
};