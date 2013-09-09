/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

// Hacks to cope with CUDA 4.x extremely wired texture unit semantics.

#define CUDA_MAX_TEXTURES 32

#define CUDA_BIND_TEXTURE(n, texarray)                \
	if(n < texarray.size) {                           \
	texunit##n.addressMode[0] = cudaAddressModeWrap;  \
	texunit##n.addressMode[1] = cudaAddressModeWrap;  \
	texunit##n.filterMode     = cudaFilterModeLinear; \
	texunit##n.normalized     = true;                 \
	cudaBindTextureToArray(texunit##n, texarray[n].pixels, texarray[n].channelDesc); \
	}

texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit0;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit1;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit2;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit3;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit4;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit5;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit6;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit7;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit8;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit9;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit10;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit11;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit12;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit13;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit14;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit15;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit16;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit17;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit18;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit19;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit20;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit21;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit22;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit23;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit24;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit25;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit26;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit27;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit28;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit29;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit30;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texunit31;

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
	case 8: px = tex2D(texunit8, u, v); break;
	case 9: px = tex2D(texunit9, u, v); break;
	case 10: px = tex2D(texunit10, u, v); break;
	case 11: px = tex2D(texunit11, u, v); break;
	case 12: px = tex2D(texunit12, u, v); break;
	case 13: px = tex2D(texunit13, u, v); break;
	case 14: px = tex2D(texunit14, u, v); break;
	case 15: px = tex2D(texunit15, u, v); break;
	case 16: px = tex2D(texunit16, u, v); break;
	case 17: px = tex2D(texunit17, u, v); break;
	case 18: px = tex2D(texunit18, u, v); break;
	case 19: px = tex2D(texunit19, u, v); break;
	case 20: px = tex2D(texunit20, u, v); break;
	case 21: px = tex2D(texunit21, u, v); break;
	case 22: px = tex2D(texunit22, u, v); break;
	case 23: px = tex2D(texunit23, u, v); break;
	case 24: px = tex2D(texunit24, u, v); break;
	case 25: px = tex2D(texunit25, u, v); break;
	case 26: px = tex2D(texunit26, u, v); break;
	case 27: px = tex2D(texunit27, u, v); break;
	case 28: px = tex2D(texunit28, u, v); break;
	case 29: px = tex2D(texunit29, u, v); break;
	case 30: px = tex2D(texunit30, u, v); break;
	case 31: px = tex2D(texunit31, u, v); break;
	default:
		return make_float3(1.0f, 1.0f, 1.0f);
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
	CUDA_BIND_TEXTURE(8, textures);
	CUDA_BIND_TEXTURE(9, textures);
	CUDA_BIND_TEXTURE(10, textures);
	CUDA_BIND_TEXTURE(11, textures);
	CUDA_BIND_TEXTURE(12, textures);
	CUDA_BIND_TEXTURE(13, textures);
	CUDA_BIND_TEXTURE(14, textures);
	CUDA_BIND_TEXTURE(15, textures);
	CUDA_BIND_TEXTURE(16, textures);
	CUDA_BIND_TEXTURE(17, textures);
	CUDA_BIND_TEXTURE(18, textures);
	CUDA_BIND_TEXTURE(19, textures);
	CUDA_BIND_TEXTURE(20, textures);
	CUDA_BIND_TEXTURE(21, textures);
	CUDA_BIND_TEXTURE(22, textures);
	CUDA_BIND_TEXTURE(23, textures);
	CUDA_BIND_TEXTURE(24, textures);
	CUDA_BIND_TEXTURE(25, textures);
	CUDA_BIND_TEXTURE(26, textures);
	CUDA_BIND_TEXTURE(27, textures);
	CUDA_BIND_TEXTURE(28, textures);
	CUDA_BIND_TEXTURE(29, textures);
	CUDA_BIND_TEXTURE(30, textures);
	CUDA_BIND_TEXTURE(31, textures);
}