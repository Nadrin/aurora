/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/memory.h>

using namespace Aurora;

void* Aurora::malloc(const size_t size, MemoryPolicy policy)
{
	void* ptr;
	switch(policy) {
	case HostMemory:
		return std::malloc(size);
	case DeviceMemory:
		if(gpu::cudaMalloc(&ptr, size) != gpu::cudaSuccess)
			return NULL;
		return ptr;
	}
	return NULL;
}

void Aurora::memzero(void* ptr, const size_t size, MemoryPolicy policy)
{
	switch(policy) {
	case HostMemory:
		std::memset(ptr, 0, size);
		break;
	case DeviceMemory:
		gpu::cudaMemset(ptr, 0, size);
		break;
	}
}

void Aurora::free(void* ptr, MemoryPolicy policy)
{
	switch(policy) {
	case HostMemory:
		std::free(ptr);
		break;
	case DeviceMemory:
		gpu::cudaFree(ptr);
		break;
	}
}

void Aurora::memupload(void* dst, const void* src, const size_t size, MemoryPolicy policy)
{
	switch(policy) {
	case HostMemory:
		std::memcpy(dst, src, size);
		break;
	case DeviceMemory:
		gpu::cudaMemcpy(dst, src, size, gpu::cudaMemcpyHostToDevice);
		break;
	}
}