/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <data/texture.h>

using namespace Aurora;

Texture::Texture() : width(0), height(0), pixels(NULL)
{ }

bool Texture::load(const unsigned int w, const unsigned int h, const float* data)
{
	channelDesc = gpu::cudaCreateChannelDesc(8, 8, 8, 8, gpu::cudaChannelFormatKindFloat);
	if(gpu::cudaMallocArray(&pixels, &channelDesc, w, h) != gpu::cudaSuccess)
		return false;

	gpu::cudaMemcpyToArray(pixels, 0, 0, data, w*h*4*sizeof(float), gpu::cudaMemcpyHostToDevice);

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