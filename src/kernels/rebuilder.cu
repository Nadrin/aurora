/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <kernels/kernels.h>

#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

using namespace Aurora;

#include <kernels/common.h>

#define REBUILDER_TPS_THRESHOLD 2048

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

struct ActiveSplitPredicate
{
	__device__ bool operator()(const Split& split)
	{ return split.size > 0; }
};

inline __device__ void calcPartition(const unsigned int N, unsigned int& L, unsigned int& R)
{
	const unsigned int n = N / 2;
	const unsigned int H = log2i(n);
	const unsigned int s = exp2i(H-1) - 1;
	const unsigned int S = exp2i(H) - 1;
	const unsigned int O = max(0, int((n-1) - s - S));

	R = 2 * (s+O);
	L = 2 * (n-1) - R;
}

inline __device__ unsigned int findSplit(const unsigned int index, const unsigned int pendingSplits,
	const Split* splitArray, Split& outSplit)
{
	unsigned int imin = 0;
	unsigned int imax = pendingSplits;
	unsigned int outIndex;

	while(imax >= imin) {
		outIndex = (imin + imax) / 2;
		outSplit = splitArray[outIndex];

		if(index < outSplit.index)
			imax = outIndex - 1;
		else if(index > outSplit.index && index >= outSplit.index + outSplit.size)
			imin = outIndex + 1;
		else break;
	}
	return outIndex;
}

__global__ static void computeTriangleBounds(const Geometry geometry,
	float* verticesMinX, float* verticesMinY, float* verticesMinZ,
	float* verticesMaxX, float* verticesMaxY, float* verticesMaxZ)
{
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= geometry.count)
		return;

	Primitive vertices;
	vertices.readPoints(geometry.vertices + index * Geometry::TriangleParams);

	verticesMinX[index] = fminf(fminf(vertices.v1.x, vertices.v2.x), vertices.v3.x);
	verticesMaxX[index] = fmaxf(fmaxf(vertices.v1.x, vertices.v2.x), vertices.v3.x);

	verticesMinY[index] = fminf(fminf(vertices.v1.y, vertices.v2.y), vertices.v3.y);
	verticesMaxY[index] = fmaxf(fmaxf(vertices.v1.y, vertices.v2.y), vertices.v3.y);

	verticesMinZ[index] = fminf(fminf(vertices.v1.z, vertices.v2.z), vertices.v3.z);
	verticesMaxZ[index] = fmaxf(fmaxf(vertices.v1.z, vertices.v2.z), vertices.v3.z);
}

__global__ static void swapMaxTriangleTPT(const unsigned int count, const float* keys,
	const unsigned int pendingSplits, const Split* splits, unsigned int* indices,
	const unsigned int doneNodes, unsigned int* nodes)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= count - 2*doneNodes)
		return;
	index = count - index - 1;

	Split split;
	findSplit(index, pendingSplits, splits, split);

	if(index == split.index || index == split.index+1)
		return;

	unsigned int threadIndex = indices[index];
	float threadValue        = keys[threadIndex];

	while(true) {
		unsigned int currentIndex = indices[split.index+1];
		float currentValue        = keys[currentIndex];

		if(threadValue > currentValue) {
			if(atomicCAS(&indices[split.index+1], currentIndex, threadIndex) == currentIndex) {
				indices[index] = currentIndex;
				swap(nodes[index], nodes[split.index+1]);
				break;
			}
		}
		else
			break;
	}
}

__global__ static void swapMaxTriangleTPS(const float* keys,
	const unsigned int pendingSplits, const Split* splits,
	unsigned int* indices, unsigned int* nodes)
{
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= pendingSplits)
		return;

	const Split split     = splits[index];
	float maxValue        = keys[indices[split.index+1]];
	unsigned int maxIndex = split.index+1;

	for(unsigned int i=split.index+2; i<split.index+split.size; i++) {
		const float value = keys[indices[i]];
		if(value > maxValue) {
			maxIndex = i;
			maxValue = value;
		}
	}

	if(maxIndex != split.index+1) {
		swap(indices[split.index+1], indices[maxIndex]);
		swap(nodes[split.index+1], nodes[maxIndex]);
	}
}

__global__ static void updateNodesTPT(const unsigned int count,
	const unsigned int pendingSplits, const Split* splits,
	const unsigned int doneNodes, unsigned int* nodes)
{
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x + 2*doneNodes;
	if(index >= count)
		return;

	Split split;
	const unsigned int splitIndex = findSplit(index, pendingSplits, splits, split);
	const unsigned int nodeIndex  = doneNodes + splitIndex;

	if(index < split.index+2)
		nodes[index] = nodeIndex;
	else
		nodes[index] = 2*nodeIndex+1;
}

__global__ static void updateNodesTPS(
	const unsigned int pendingSplits, const Split* splits,
	const unsigned int doneNodes, unsigned int* nodes)
{
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= pendingSplits)
		return;

	const Split split = splits[index];
	const unsigned int nodeIndex = doneNodes + index;

	nodes[split.index]   = nodeIndex;
	nodes[split.index+1] = nodeIndex;
	for(unsigned int i=split.index+2; i<split.index + split.size; i++) {
		nodes[i] = 2*nodeIndex+1;
	}
}

__global__ static void emitSplitsKernel(const unsigned int pendingSplits,
	const Split* inSplits, Split* outSplits, unsigned int* generatedSplits)
{
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= pendingSplits)
		return;

	Split split = inSplits[index];
	if(split.size <= 2)
		return;

	unsigned int newSplits = 0;
	unsigned int numL, numR, pL, pR;
	calcPartition(split.size, numL, numR);
	pL = split.index + 2 * (pendingSplits - index);
	pR = pL + numL;

	outSplits[2*index].index = pL;
	outSplits[2*index].size  = numL;
	if(numL > 0) newSplits++;

	outSplits[2*index+1].index = pR;
	outSplits[2*index+1].size  = numR;
	if(numR > 0) newSplits++;

	atomicAdd(generatedSplits, newSplits);
}

__global__ static void applyIndices(const Geometry source, Geometry dest, const unsigned int* indices)
{
	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index >= source.count)
		return;

	Primitive buffer;
	const unsigned int position = indices[index];

	// Copy vertices
	buffer.readValues(source.vertices + position * Geometry::TriangleParams);
	buffer.writeValues(dest.vertices + index * Geometry::TriangleParams);

	// Copy normals
	buffer.readValues(source.normals + position * Geometry::TriangleParams);
	buffer.writeValues(dest.normals + index * Geometry::TriangleParams);
}

__host__ static void emitSplits(const unsigned int pendingSplits,
	Split* inSplits, Split* outSplits, unsigned int* hptrGeneratedSplits, unsigned int* dptrGeneratedSplits)
{
	dim3 blockSize(256);
	dim3 gridSize = make_grid(blockSize, dim3(pendingSplits));

	*hptrGeneratedSplits = 0;
	emitSplitsKernel<<<gridSize, blockSize>>>(pendingSplits, inSplits, outSplits, dptrGeneratedSplits);
	cudaDeviceSynchronize();

	thrust::device_ptr<Split> ptrInSplits(inSplits);
	thrust::device_ptr<Split> ptrOutSplits(outSplits);
	thrust::copy_if(ptrOutSplits, ptrOutSplits + (*hptrGeneratedSplits), ptrInSplits, ActiveSplitPredicate());
}

__host__ static void sortTriangles(const unsigned int count,
	const float* keys, unsigned int* indices, unsigned int* nodes,
	thrust::device_vector<unsigned int>& permutation,
	thrust::device_vector<unsigned int>& tempi, thrust::device_vector<float>& tempf)
{
	thrust::device_ptr<const float>  ptrKeys(keys);
	thrust::device_ptr<unsigned int> ptrIndices(indices);
	thrust::device_ptr<unsigned int> ptrNodes(nodes);

	// Generate identity permutation
	thrust::sequence(permutation.begin(), permutation.end(), 0, 1);

	// Spatial sort
	thrust::gather(ptrIndices, ptrIndices + count, ptrKeys, tempf.begin());
	thrust::stable_sort_by_key(tempf.begin(), tempf.end(), permutation.begin());

	// Node sort
	thrust::gather(permutation.begin(), permutation.end(), ptrNodes, tempi.begin());
	thrust::stable_sort_by_key(tempi.begin(), tempi.end(), permutation.begin());

	// Apply permutations to indices
	thrust::copy(ptrIndices, ptrIndices + count, tempi.begin());
	thrust::gather(permutation.begin(), permutation.end(), tempi.begin(), ptrIndices);

	// Apply permutations to nodes
	thrust::copy(ptrNodes, ptrNodes + count, tempi.begin());
	thrust::gather(permutation.begin(), permutation.end(), tempi.begin(), ptrNodes);
}

__host__ bool cudaRebuildNMH(Geometry& geometry)
{
	const size_t numLevels = log2i(geometry.count / 2) + 1;
	const size_t maxSplits = 1 << numLevels;

	// Result geometry
	Geometry result;
	result.initialize();

	// Prefer L1 cache
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

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
	Split *inSplits, *outSplits;
	cudaMalloc(&inSplits, 2 * maxSplits * sizeof(Split));
	cudaMalloc(&outSplits, 2 * maxSplits * sizeof(Split));

	const Split initialSplit(0, geometry.count);
	cudaMemcpy(inSplits, &initialSplit, sizeof(Split), cudaMemcpyHostToDevice);

	// Vertices bounds
	float* verticesMin[3];
	float* verticesMax[3];
	for(int i=0; i<3; i++) {
		cudaMalloc(&verticesMin[i], geometry.count * sizeof(float));
		cudaMalloc(&verticesMax[i], geometry.count * sizeof(float));
	}

	// GPU generated split count
	unsigned int* hptrGeneratedSplits;
	unsigned int* dptrGeneratedSplits;
	cudaHostAlloc(&hptrGeneratedSplits, sizeof(unsigned int), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&dptrGeneratedSplits, hptrGeneratedSplits, 0);

	// Temporary arrays for sorting
	thrust::device_vector<unsigned int> permutation(geometry.count);
	thrust::device_vector<unsigned int> tempi(geometry.count);
	thrust::device_vector<float> tempf(geometry.count);

	// Computation state
	int axis = 0;
	unsigned int pendingSplits = 1;
	unsigned int doneNodes     = 0;

	dim3 gridSize, blockSize;

	// Compute triangle bounds
	blockSize = dim3(192);
	gridSize  = make_grid(blockSize, dim3(geometry.count));
	computeTriangleBounds<<<gridSize, blockSize>>>(geometry,
		verticesMin[0], verticesMin[1], verticesMin[2],
		verticesMax[0], verticesMax[1], verticesMax[2]);

	// Loop every level of the hierarchy
	for(unsigned int i=0; i<numLevels; i++) {
		// 1. Lexicographical sort
		const float* keysMin = verticesMin[axis];
		const float* keysMax = verticesMax[axis];

		sortTriangles(geometry.count, keysMin, indices, nodes, permutation, tempi, tempf);

		// Finish if on last level
		if(i == numLevels-1)
			break;

		// 2. Find maximal triangle in every split
		blockSize = dim3(256);
		if(pendingSplits <= REBUILDER_TPS_THRESHOLD) {
			gridSize = make_grid(blockSize, dim3(geometry.count - 2*doneNodes));
			swapMaxTriangleTPT<<<gridSize, blockSize>>>(geometry.count, keysMax, pendingSplits, inSplits,
				indices, doneNodes, nodes);
		}
		else {
			gridSize = make_grid(blockSize, dim3(pendingSplits));
			swapMaxTriangleTPS<<<gridSize, blockSize>>>(keysMax, pendingSplits, inSplits, indices, nodes);
		}

		// 3. Update node values
		blockSize = dim3(256);
		if(pendingSplits <= REBUILDER_TPS_THRESHOLD) {
			gridSize = make_grid(blockSize, dim3(geometry.count - 2*doneNodes));
			updateNodesTPT<<<gridSize, blockSize>>>(geometry.count, pendingSplits, inSplits, doneNodes, nodes);
		}
		else {
			gridSize = make_grid(blockSize, dim3(pendingSplits));
			updateNodesTPS<<<gridSize, blockSize>>>(pendingSplits, inSplits, doneNodes, nodes);
		}

		if(i < max(0, int(numLevels-2))) {
			// 4. Emit new splits
			emitSplits(pendingSplits, inSplits, outSplits, hptrGeneratedSplits, dptrGeneratedSplits);

			doneNodes    += pendingSplits;
			pendingSplits = *hptrGeneratedSplits;
		}

		// 5. Increment axis
		axis = (axis + 1) % 3;
	}

	// Free resources
	cudaFree(nodes);
	cudaFree(inSplits);
	cudaFree(outSplits);
	cudaFreeHost(hptrGeneratedSplits);

	for(int i=0; i<3; i++) {
		cudaFree(verticesMin[i]);
		cudaFree(verticesMax[i]);
	}

	// Apply indices
	result.resize(geometry.count, Geometry::AllocDefault);

	blockSize = dim3(256);
	gridSize  = make_grid(blockSize, dim3(geometry.count));
	applyIndices<<<gridSize, blockSize>>>(geometry, result, indices);
	cudaFree(indices);

	geometry.free();
	geometry = result;
	
	// Restore default cache setting
	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
	return true;
}