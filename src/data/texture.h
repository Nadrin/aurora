/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <util/array.h>

namespace Aurora {

#ifndef __CUDACC__
typedef gpu::cudaChannelFormatDesc cudaChannelFormatDesc;
typedef gpu::cudaArray cudaArray;
#endif

class Texture
{
public:
	Texture();

	bool load(const unsigned int w, const unsigned int h, const float* data);
	void free();

	unsigned int width;
	unsigned int height;

	cudaChannelFormatDesc channelDesc;
	cudaArray* pixels;
};

typedef Array<Texture, HostMemory> TexturesArray;

} // Aurora