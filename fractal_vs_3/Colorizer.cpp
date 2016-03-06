#include "Colorizer.h"


Colorizer::Colorizer(int maxCount) : m_maxCount(maxCount)
{
}


Colorizer::~Colorizer()
{
}

unsigned int Colorizer::colorRainbow(int count)
{
	if (count < m_maxCount)
	{
		float h = count % 360;

		h /= 60;   // sector 0 to 5
		int i = h;
		float f = h - i; // factorial part of h
		float q = 1 - f;

		f *= 255;
		q *= 255;

		unsigned int rArr[6] = { 255u, q, 0u, 0u, f, 255u };

		unsigned int gArr[6] = { f, 255u, 255u, q, 0u, 0u };

		unsigned int bArr[6] = { 0u, 0u, f, 255u, 255u, q };

		unsigned int color = (bArr[i] << 16) | (gArr[i] << 8) | rArr[i];

		return color;
	}

	return 0u;
}

unsigned int Colorizer::colorGray(int count)
{
	if (count < m_maxCount)
	{
		unsigned int luminance = (((unsigned int)(count * (255 / (float)m_maxCount)) + 1));

		unsigned int color = (luminance << 16) | (luminance << 8) | (luminance << 0);

		return color;
	}

	return 0xffffff;
}
