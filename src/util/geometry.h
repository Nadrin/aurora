/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Geometry
{
public:
	enum GeometryAllocMode {
		AllocEmpty = 0,
		AllocDefault,
		AllocStaging,
	};

	float* vertices;
	unsigned int count;
	GeometryAllocMode mode;
protected:
	__host__ bool resizeDefault(const unsigned int n);
	__host__ bool resizeStaging(const unsigned int n);
public:
	__host__ void initialize();
	__host__ bool resize(const unsigned int n, GeometryAllocMode allocMode);
	__host__ void free();

	__host__ bool copyToDevice(Geometry& other) const;
	__host__ bool padToEven(const unsigned int n);

	static const size_t TriangleSize = 9 * sizeof(float);
};

} // Aurora