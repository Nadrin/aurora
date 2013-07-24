/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <util/geometry.h>

using namespace Aurora;

void Geometry::initialize()
{
	count    = 0;
	mode     = Geometry::AllocEmpty;
	vertices = NULL;
}

bool Geometry::resizeDefault(const unsigned int n)
{
	try 
	{
		if(gpu::cudaMalloc(&vertices, n * Geometry::TriangleSize) != gpu::cudaSuccess)
			throw std::exception();
	} 
	catch(const std::exception&)
	{
		free();
		return false;
	}
	return true;
}

bool Geometry::resizeStaging(const unsigned int n)
{
	try
	{
		if(gpu::cudaHostAlloc(&vertices, n * Geometry::TriangleSize,
			cudaHostAllocMapped | cudaHostAllocWriteCombined) != gpu::cudaSuccess)
			throw std::exception();
	}
	catch(const std::exception&)
	{
		free();
		return false;
	}
	return true;
}

bool Geometry::resize(const unsigned int n, GeometryAllocMode allocMode)
{
	unsigned int _n = n + (n%2);
	if(_n == count)
		return true;
	else
		free();

	switch(allocMode) {
	case Geometry::AllocDefault:
		mode = Geometry::AllocDefault;
		if(!resizeDefault(_n))
			return false;
		break;
	case Geometry::AllocStaging:
		mode = Geometry::AllocStaging;
		if(!resizeStaging(_n))
			return false;
		break;
	}

	count = _n;
	return true;
}

void Geometry::free()
{
	switch(mode) {
	case Geometry::AllocDefault:
		gpu::cudaFree(vertices);
		break;
	case Geometry::AllocStaging:
		gpu::cudaFreeHost(vertices);
		break;
	}
	initialize();
}

bool Geometry::copyToDevice(Geometry& other) const
{
	if(mode != Geometry::AllocStaging)
		return false;

	float* deviceVertices;
	gpu::cudaHostGetDevicePointer(&deviceVertices, vertices, 0);

	gpu::cudaMemcpy(other.vertices, deviceVertices,
		count * Geometry::TriangleSize, gpu::cudaMemcpyDeviceToDevice);
	return true;
}

bool Geometry::padToEven(const unsigned int n)
{
	if(count < 2 || mode != Geometry::AllocStaging)
		return false;
	if(n % 2 == 0)
		return true;

	memcpy(&vertices[count-1], &vertices[count-2], Geometry::TriangleSize);
	return true;
}