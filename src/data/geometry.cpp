/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <data/geometry.h>
#include <kernels/kernels.h>

using namespace Aurora;

void Geometry::initialize()
{
	count    = 0;
	mode     = Geometry::AllocEmpty;
	vertices = NULL;
	normals  = NULL;
	texcoords= NULL;
	shaders  = NULL;
}

bool Geometry::resizeDefault(const unsigned int n)
{
	try 
	{
		if(gpu::cudaMalloc(&vertices, n * Geometry::TriangleSize) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&normals, n * Geometry::TriangleSize) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&texcoords, n * Geometry::TriangleUVSize) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaMalloc(&shaders, n * sizeof(unsigned short)) != gpu::cudaSuccess)
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
		if(gpu::cudaHostAlloc(&normals, n * Geometry::TriangleSize,
			cudaHostAllocMapped | cudaHostAllocWriteCombined) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaHostAlloc(&texcoords, n * Geometry::TriangleUVSize,
			cudaHostAllocMapped | cudaHostAllocWriteCombined) != gpu::cudaSuccess)
			throw std::exception();
		if(gpu::cudaHostAlloc(&shaders, n * sizeof(unsigned short),
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

Geometry Geometry::convertToDevice() const
{
	Geometry dev;
	dev.count = count;
	dev.mode  = mode;
	
	gpu::cudaHostGetDevicePointer(&dev.vertices, vertices, 0);
	gpu::cudaHostGetDevicePointer(&dev.normals, normals, 0);
	gpu::cudaHostGetDevicePointer(&dev.texcoords, texcoords, 0);
	gpu::cudaHostGetDevicePointer(&dev.shaders, shaders, 0);
	return dev;
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
		gpu::cudaFree(normals);
		gpu::cudaFree(texcoords);
		gpu::cudaFree(shaders);
		break;
	case Geometry::AllocStaging:
		gpu::cudaFreeHost(vertices);
		gpu::cudaFreeHost(normals);
		gpu::cudaFreeHost(texcoords);
		gpu::cudaFreeHost(shaders);
		break;
	}
	initialize();
}

bool Geometry::copyToDevice(Geometry& dest) const
{
	if(mode != Geometry::AllocStaging)
		return false;

	Geometry source = convertToDevice();

	gpu::cudaMemcpy(dest.vertices, source.vertices,
		count * Geometry::TriangleSize, gpu::cudaMemcpyDeviceToDevice);
	gpu::cudaMemcpy(dest.normals, source.normals,
		count * Geometry::TriangleSize, gpu::cudaMemcpyDeviceToDevice);
	gpu::cudaMemcpy(dest.texcoords, source.texcoords,
		count * Geometry::TriangleUVSize, gpu::cudaMemcpyDeviceToDevice);
	gpu::cudaMemcpy(dest.shaders, source.shaders,
		count * sizeof(unsigned short), gpu::cudaMemcpyDeviceToDevice);
	return true;
}

bool Geometry::copyToDeviceTransform(Geometry& dest, const Transform* transforms, const unsigned int objects) const
{
	if(mode != Geometry::AllocStaging)
		return false;

	Geometry source = convertToDevice();
	cudaTransform(source, dest, transforms, objects);
	gpu::cudaDeviceSynchronize();
	return true;
}

bool Geometry::padToEven(const unsigned int n)
{
	if(count < 2 || mode != Geometry::AllocStaging)
		return false;
	if(n % 2 == 0)
		return true;

	memcpy(&vertices[count-1], &vertices[count-2], Geometry::TriangleSize);
	memcpy(&normals[count-1], &normals[count-2], Geometry::TriangleSize);
	memcpy(&texcoords[count-1], &texcoords[count-2], Geometry::TriangleUVSize);
	memcpy(&shaders[count-1], &shaders[count-2], sizeof(unsigned short));
	return true;
}

bool Geometry::rebuild()
{
	return cudaRebuildNMH(*this);
}