/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <core/memory.h>

namespace Aurora {

inline __host__ __device__ unsigned int getID(const unsigned int val)
{ return val-1; }
inline __host__ __device__ unsigned int setID(const unsigned int val)
{ return val+1; }

template <class T, MemoryPolicy policy>
class Array
{
public:
	__host__ 
	Array() : size(0), items(NULL)
	{ }

	__host__
	~Array()
	{
		free(items, policy);
	}

	__host__ __device__
	const T& operator[](const size_t index) const { return items[index]; }

	__host__ __device__
	T& operator[](const size_t index) { return items[index]; }

	__host__
	bool resize(const size_t count)
	{
		free(items, policy);

		if(count > 0) {
			items = (T*)malloc(count * sizeof(T), policy);
			if(!items) 
				return false;
			memzero(items, count * sizeof(T), policy);
		}
		else
			items = NULL;

		size = count;
		return true;
	}

	__host__
	void copyToDevice(Array<T, DeviceMemory>& deviceArray)
	{
		memupload(deviceArray.items, items, size*sizeof(T), DeviceMemory);
	}

	size_t size;
	T* items;
};

} // Aurora