#include "quadruple2.h"

// Double single functions based on DSFUN90 package:
// http://crd.lbl.gov/~dhbailey/mpdist/index.html

quadruple2::quadruple2(double a)
{
	hi = a;
	lo = 0.0;
}


quadruple2::quadruple2(double low, double high) : lo(low), hi(high)
{
}

quadruple2::~quadruple2()
{
}


quadruple2 quadruple2::operator+(const quadruple2 &q) const
{
	// Compute dsa + dsb using Knuth's trick.
	double t1 = hi + q.hi;
	double e = t1 - hi;
	double t2 = ((q.hi - e) + (hi - (t1 - e))) + lo + q.lo;

	// The result is t1 + t2, after normalization.
	e = t1 + t2;
	double c0 = e;
	double c1 = t2 - (e - t1);

	return quadruple2(c1, c0);
}

quadruple2 quadruple2::operator-(const quadruple2 &q) const
{
	// Compute dsa - dsb using Knuth's trick.
	double t1 = hi - q.hi;
	double e = t1 - hi;
	double t2 = ((-q.hi - e) + (hi - (t1 - e))) + lo - q.lo;

	// The result is t1 + t2, after normalization.
	e = t1 + t2;
	double c0 = e;
	double c1 = t2 - (e - t1);

	return quadruple2(c1, c0);
}

quadruple2 quadruple2::operator*(const quadruple2 &q) const
{
	// This splits dsa(1) and dsb(1) into high-order and low-order words.
	double cona = hi * SPLIT;
	double conb = q.hi * 8193.0f;
	double sa1 = cona - (cona - hi);
	double sb1 = conb - (conb - q.hi);
	double sa2 = hi - sa1;
	double sb2 = q.hi - sb1;

	// Multilply a0 * b0 using Dekker's method.
	double c11 = hi * q.hi;
	double c21 = (((sa1 * sb1 - c11) + sa1 * sb2) + sa2 * sb1) + sa2 * sb2;

	// Compute a0 * b1 + a1 * b0 (only high-order word is needed).
	double c2 = hi * q.lo + lo * q.hi;

	// Compute (c11, c21) + c2 using Knuth's trick, also adding low-order product.
	double t1 = c11 + c2;
	double e = t1 - c11;
	double t2 = ((c2 - e) + (c11 - (t1 - e))) + c21 + lo * q.lo;

	// The result is t1 + t2, after normalization.
	e = t1 + t2;
	double c0 = e;
	double c1 = t2 - (e - t1);

	return quadruple2(c1, c0);
}

bool quadruple2::operator<(const quadruple2 &q) const
{
	return (hi < q.hi) || (hi == q.hi && lo < q.lo);
}


