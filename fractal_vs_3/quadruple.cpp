#include "quadruple.h"
#include <cmath>


//
// Quad (double double) precision.
//
// http://hal.archives-ouvertes.fr/docs/00/06/33/56/PDF/float-float.pdf
// http://andrewthall.org/papers/df64_qf128.pdf
//


quadruple::quadruple(double a)
{
	double c = SPLIT * a;

	double aBig = c - a;

	hi = c - aBig;

	lo = a - hi;
}

quadruple::quadruple(double high, double low) : hi(high), lo(low)
{}

quadruple::~quadruple()
{
}

quadruple quadruple::operator-() const
{
	quadruple q(*this);

	q.hi *= -1.0;
	q.lo *= -1.0;

	return q;
}

quadruple quadruple::operator+(const quadruple &q) const
{
	/*
	FIXME: Implement version without a branch.
	*/
	
	double r = hi + q.hi;

	double s;

	if (std::fabs(hi) >= std::fabs(q.hi))
	{
		s = (((hi - r) + q.hi) + q.lo) + lo;
	}
	else
	{
		s = (((q.hi - r) + hi) + lo) + q.lo;
	}
	
	return add12(r, s);
}

quadruple quadruple::operator-(const quadruple &q) const
{
	return *this + -quadruple(q);
}

quadruple quadruple::operator*(const quadruple &q) const
{
	quadruple abh = mul12(hi, q.hi);

	double t3 = ((hi * q.lo) * (lo * q.hi)) + abh.lo;

	//
	// Error in publication, not sure how t3 should be used?
	//

	return add12(abh.hi, t3);
}

bool quadruple::operator<(const quadruple &q) const
{
	return (hi < q.hi) || (hi == q.hi && lo < q.lo);
}

quadruple quadruple::add12(const double a, const double b) const
{	
	double s = a + b;

	double v = s - a;

	double r = (a - (s - v)) + (b - v);

	return quadruple(s, r);
}

quadruple quadruple::mul12(const double a, const double b) const
{
	quadruple aquad(a);

	quadruple bquad(b);

	double ab = a * b;

	double err1 = ab - (aquad.hi * bquad.hi);

	double err2 = err1 - (aquad.lo * bquad.hi);

	double err3 = err2 - (aquad.hi * bquad.lo);

	return quadruple(ab, (aquad.lo * bquad.lo) - err3);
}