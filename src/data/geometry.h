/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/math.h>
#include <util/transform.h>

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
	float* normals;
	float* texcoords;
	unsigned short* shaders;

	float* tangents;
	float* bitangents;

	unsigned int count;
	GeometryAllocMode mode;
protected:
	bool resizeDefault(const unsigned int n);
	bool resizeStaging(const unsigned int n);
	Geometry convertToDevice() const;
public:
	void initialize();
	bool resize(const unsigned int n, GeometryAllocMode allocMode);
	void free();

	bool copyToDevice(Geometry& other) const;
	bool copyToDeviceTransform(Geometry& other, const Transform* transforms, const unsigned int objects) const;
	bool padToEven(const unsigned int n);

	bool rebuild();
	void generateTB();

	static const size_t TriangleSize   = 9 * sizeof(float);
	static const size_t TriangleParams = 9;
	static const size_t TriangleUVSize = 6 * sizeof(float);
	static const size_t TriangleUVs    = 6;
};

} // Aurora