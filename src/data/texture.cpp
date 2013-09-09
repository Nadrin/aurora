/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <util/math.h>
#include <data/texture.h>

using namespace Aurora;

Texture::Texture() : width(0), height(0), pixels(NULL)
{ }

bool Texture::load(const unsigned int w, const unsigned int h, const unsigned char* data)
{
	channelDesc = gpu::cudaCreateChannelDesc(8, 8, 8, 8, gpu::cudaChannelFormatKindUnsigned);
	if(gpu::cudaMallocArray(&pixels, &channelDesc, w, h) != gpu::cudaSuccess)
		return false;

	gpu::cudaMemcpyToArray(pixels, 0, 0, data, w*h*sizeof(uchar4), gpu::cudaMemcpyHostToDevice);

	width  = w;
	height = h;
	return true;
}

bool Texture::load(const unsigned int w, const unsigned int h, const float* data)
{
	uchar4* buffer;
	uchar4* devbuffer;

	if(gpu::cudaHostAlloc(&buffer, w*h*sizeof(uchar4),
		cudaHostAllocMapped | cudaHostAllocWriteCombined) != gpu::cudaSuccess)
		return false;

	channelDesc = gpu::cudaCreateChannelDesc(8, 8, 8, 8, gpu::cudaChannelFormatKindUnsigned);
	if(gpu::cudaMallocArray(&pixels, &channelDesc, w, h) != gpu::cudaSuccess) {
		gpu::cudaFreeHost(buffer);
		return false;
	}

	const float* ptr = data;
	for(unsigned int i=0; i<w*h; i++) {
		buffer[i] = make_uchar4(
			unsigned char(ptr[0]),
			unsigned char(ptr[1]), 
			unsigned char(ptr[2]),
			unsigned char(ptr[3]*255.0f));
		//buffer[i] = make_float4(ptr[0]/255.0f, ptr[1]/255.0f, ptr[2]/255.0f, ptr[3]);
		ptr += 4;
	}

	gpu::cudaHostGetDevicePointer(&devbuffer, buffer, 0);
	gpu::cudaMemcpyToArray(pixels, 0, 0, devbuffer, w*h*sizeof(uchar4), gpu::cudaMemcpyDeviceToDevice);
	gpu::cudaFreeHost(buffer);

	width  = w;
	height = h;
	return true;
}

void Texture::free()
{
	gpu::cudaFreeArray(pixels);
	width  = 0;
	height = 0;
	pixels = NULL;
}