#pragma once
class quadruple
{
public:
	quadruple(double a);
	~quadruple();

	// negation
	quadruple operator-() const;

	quadruple operator+(const quadruple &q) const;
	quadruple operator-(const quadruple &q) const;
	quadruple operator*(const quadruple &q) const;
	bool operator <(const quadruple &q) const;

protected:
	quadruple(double high, double low);
	inline quadruple add12(const double a, const double b) const;
	inline quadruple mul12(const double a, const double b) const;

	// For float p = 24, double p = 53 ((2 << 27) + 1).
	static const unsigned int SPLIT = ((2 << 27) + 1);
	double hi;
	double lo;
};

