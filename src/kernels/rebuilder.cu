/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

#include <thrust/sequence.h>

using namespace Aurora;

struct Split 
{
	__host__ __device__ 
	Split() : index(0), size(0) { }

	__host__ __device__ 
	Split(const unsigned int _index, const unsigned int _size) :
		index(_index), size(_size) { }

	unsigned int index;
	unsigned int size;
};

__host__ bool cudaRebuildNMH(Geometry& geometry)
{
	const size_t numLevels = log2i(geometry.count);
	const size_t numSplits = geometry.count/2 + 1;

	// Triangle indices
	unsigned int* indices;
	cudaMalloc(&indices, geometry.count * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> thrustIndices(indices);
	thrust::sequence(thrustIndices, thrustIndices + geometry.count);
	
	// Node indices
	unsigned int* nodes;
	cudaMalloc(&nodes, geometry.count * sizeof(unsigned int));
	cudaMemset(nodes, 0, geometry.count * sizeof(unsigned int));

	// Active splits
	Split* splits[2];
	cudaMalloc(&splits[0], numSplits * sizeof(Split));
	cudaMalloc(&splits[1], numSplits * sizeof(Split));

	const Split initialSplit(0, geometry.count);
	cudaMemcpy(splits[0], &initialSplit, sizeof(Split), cudaMemcpyHostToDevice);

	// Vertices bounds
	float* verticesMin[3];
	float* verticesMax[3];
	for(int i=0; i<3; i++) {
		cudaMalloc(&verticesMin[i], geometry.count * sizeof(float));
		cudaMalloc(&verticesMax[i], geometry.count * sizeof(float));
	}

	// GPU generated split count
	unsigned int* hostGeneratedSplits;
	unsigned int* deviceGeneratedSplits;
	cudaHostAlloc(&hostGeneratedSplits, sizeof(unsigned int), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&deviceGeneratedSplits, hostGeneratedSplits, 0);

	// Computation state
	int  axis = 0;
	dim3 gridSize, blockSize;

	// TODO: Computation

	// Free resources
	cudaFree(indices);
	cudaFree(nodes);
	cudaFree(splits[0]);
	cudaFree(splits[1]);
	cudaFreeHost(hostGeneratedSplits);

	for(int i=0; i<3; i++) {
		cudaFree(verticesMin[i]);
		cudaFree(verticesMax[i]);
	}
	return true;
}