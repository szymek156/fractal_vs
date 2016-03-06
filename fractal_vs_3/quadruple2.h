#pragma once
class quadruple2
{
public:
	inline quadruple2(double a);
	inline quadruple2(double low, double high);
	~quadruple2();

	inline quadruple2 operator+(const quadruple2 &q) const;
	inline quadruple2 operator-(const quadruple2 &q) const;
	inline quadruple2 operator*(const quadruple2 &q) const;
	inline bool operator <(const quadruple2 &q) const;

	// For float p = 24, double p = 53 ((2 << 27) + 1).
	static const unsigned int SPLIT = ((2 << 27) + 1);

//protected:
	double hi;
	double lo;
};

