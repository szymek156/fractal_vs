#pragma once
class quadruple2
{
public:
	quadruple2(double a);
	quadruple2(double low, double high);
	~quadruple2();

	quadruple2 operator+(const quadruple2 &q) const;
	quadruple2 operator-(const quadruple2 &q) const;
	quadruple2 operator*(const quadruple2 &q) const;
	bool operator <(const quadruple2 &q) const;

	// For float p = 24, double p = 53 ((2 << 27) + 1).
	static const unsigned int SPLIT = ((2 << 27) + 1);

//protected:
	double hi;
	double lo;
};

