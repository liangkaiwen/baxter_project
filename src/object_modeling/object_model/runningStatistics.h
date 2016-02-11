/*
 * runningStatistics.h
 *
 *  Created on: Apr 20, 2011
 *      Author: peter
 *
 *  Copied and modified from: http://www.johndcook.com/standard_deviation.html
 */

#pragma once

#include <float.h>
#include <sstream>

class RunningStatistics {
public:
	RunningStatistics() : m_n(0), m_oldM(0), m_newM(0), m_oldS(0), m_newS(0), m_min(0), m_max(0), m_lastValue(0) {}

	void clear() {
		m_n = 0;
	}

	void push(double x) {
		m_n++;

		// added:
		m_lastValue = x;

		// See Knuth TAOCP vol 2, 3rd edition, page 232
		if (m_n == 1) {
			m_oldM = m_newM = x;
			m_oldS = 0.0;
			m_min = x;
			m_max = x;
		}
		else {
			m_newM = m_oldM + (x - m_oldM) / m_n;
			m_newS = m_oldS + (x - m_oldM) * (x - m_newM);

			// set up for next iteration
			m_oldM = m_newM;
			m_oldS = m_newS;

			m_min = (x < m_min) ? x : m_min;
			m_max = (x > m_max) ? x : m_max;
		}
	}

	int numDataValues() const {
		return m_n;
	}

	double mean() const {
		return (m_n > 0) ? m_newM : 0.0;
	}

	double variance() const {
		return ((m_n > 1) ? m_newS / (m_n - 1) : 0.0);
	}

	double standardDeviation() const {
		return sqrt(variance());
	}

	double min() const {
		return (m_n > 0) ? m_min : -DBL_MAX;
	}

	double max() const {
		return (m_n > 0) ? m_max : DBL_MAX;
	}

	double lastValue() const {
		return (m_n > 0) ? m_lastValue : 0.0; // yeah...not sure what this should be if empty
	}

	double total() const {
		return (mean() * numDataValues());
	}

	std::string summary() const {
		std::stringstream ss;
		ss		<< " Mean: " << mean()
				<< " Std: " << standardDeviation()
				<< " Last Value: " << lastValue()
				<< " Count: " << numDataValues()
				<< " Min: " << min()
				<< " Max: " << max()
				<< " Total: " << total()
				;
		return ss.str();
	}

private:
	int m_n;
	double m_oldM, m_newM, m_oldS, m_newS;
	// added:
	double m_min, m_max;
	double m_lastValue;
};

