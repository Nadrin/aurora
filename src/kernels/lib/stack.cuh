/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

template <class T, unsigned int depth>
class Stack
{
public:
	__device__ Stack() : size(0) { }

	__device__ inline void push(const T& item)
	{
		data[size++] = item;
	}

	__device__ inline T& pop()
	{
		return data[--size];
	}

	T data[depth];
	unsigned int size;
};