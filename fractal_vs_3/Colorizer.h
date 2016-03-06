#pragma once
class Colorizer
{
public:
	Colorizer(int maxCount);
	virtual ~Colorizer();

	unsigned int colorRainbow(int count);
	unsigned int colorGray(int count);

protected:
	int m_maxCount;
};

