/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

// Hacks to cope with CUDA 4.x extremely wired texture unit semantics.

#define CUDA_MAX_TEXTURES 8

#define CUDA_BIND_TEXTURE(n, texarray)                \
	if(n < texarray.size) {                           \
	texunit##n.addressMode[0] = cudaAddressModeWrap;  \
	texunit##n.addressMode[1] = cudaAddressModeWrap;  \
	texunit##n.filterMode     = cudaFilterModeLinear; \
	texunit##n.normalized     = true;                 \
	cudaBindTextureToArray(texunit##n, texarray[n].pixels, texarray[n].channelDesc); \
	}

texture<float4, 2, cudaReadModeElementType> texunit0;
texture<float4, 2, cudaReadModeElementType> texunit1;
texture<float4, 2, cudaReadModeElementType> texunit2;
texture<float4, 2, cudaReadModeElementType> texunit3;
texture<float4, 2, cudaReadModeElementType> texunit4;
texture<float4, 2, cudaReadModeElementType> texunit5;
texture<float4, 2, cudaReadModeElementType> texunit6;
texture<float4, 2, cudaReadModeElementType> texunit7;

inline __device__ float3 texfetch(const int id, const float u, const float v)
{
	float4 px;
	switch(id) {
	case 0: px = tex2D(texunit0, u, v); break;
	case 1: px = tex2D(texunit1, u, v); break;
	case 2: px = tex2D(texunit2, u, v); break;
	case 3: px = tex2D(texunit3, u, v); break;
	case 4: px = tex2D(texunit4, u, v); break;
	case 5: px = tex2D(texunit5, u, v); break;
	case 6: px = tex2D(texunit6, u, v); break;
	case 7: px = tex2D(texunit7, u, v); break;
	default:
		return make_float3(0.0f, 0.0f, 0.0f);
	}
	return make_float3(px.x, px.y, px.z);
}

__host__ static void bindTextures(const TexturesArray& textures)
{
	CUDA_BIND_TEXTURE(0, textures);
	CUDA_BIND_TEXTURE(1, textures);
	CUDA_BIND_TEXTURE(2, textures);
	CUDA_BIND_TEXTURE(3, textures);
	CUDA_BIND_TEXTURE(4, textures);
	CUDA_BIND_TEXTURE(5, textures);
	CUDA_BIND_TEXTURE(6, textures);
	CUDA_BIND_TEXTURE(7, textures);
}