/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <kernels/lib/stack.cuh>
#include <kernels/lib/ray.cuh>
#include <kernels/lib/primitive.cuh>

// NMH traversal state
struct TraversalState
{
	__device__ TraversalState() : index(0), axis(0) { }
	__device__ TraversalState(const unsigned int i, const int a, const float2& r)
		: index(i), axis(a), range(r) { }

	unsigned int index;
	int axis;
	float2 range;
};

// Helper functions
inline __device__ float min(const float* vertices, const int axis)
{
	return fminf(fminf(vertices[3*axis], vertices[3*axis+1]), vertices[3*axis+2]);
}

inline __device__ float max(const float* vertices, const int axis)
{
	return fmaxf(fmaxf(vertices[3*axis], vertices[3*axis+1]), vertices[3*axis+2]);
}

// Intersect any triangle
inline __device__ bool intersectAny(const Geometry& geometry, Ray& ray)
{
	Stack<TraversalState, AURORA_INTSTACK_DEPTH> stack;

	stack.push(TraversalState(0, 0, make_float2(Epsilon, Infinity)));
	while(stack.size > 0) {
		TraversalState state = stack.pop();
		if(state.index >= geometry.count)
			continue;

		const float* vertices1 = geometry.vertices + state.index * Geometry::TriangleParams;
		const float* vertices2 = geometry.vertices + (state.index+1) * Geometry::TriangleParams;

		const float2 slab = make_float2(
			fminf(min(vertices1, state.axis), min(vertices2, state.axis)),
			fmaxf(max(vertices1, state.axis), max(vertices2, state.axis)));

		if(!ray.intersect(slab, state.axis, state.range))
			continue;

		Primitive3 triangle1, triangle2;
		triangle1.readPoints(geometry.vertices + state.index * Geometry::TriangleParams);
		triangle2.readPoints(geometry.vertices + (state.index+1) * Geometry::TriangleParams);

		float u, v;
		float t;

		if(ray.intersect(triangle1, u, v, t) && t > Epsilon && t < ray.t)
			return true;
		if(ray.intersect(triangle2, u, v, t) && t > Epsilon && t < ray.t)
			return true;

		state.axis = (state.axis + 1) % 3;
		unsigned int L = 2*state.index + 2;
		unsigned int R = 2*state.index + 4;

		switch(state.axis) {
		case 0: if(ray.dir.x < 0.0f) swap(L, R); break;
		case 1: if(ray.dir.y < 0.0f) swap(L, R); break;
		case 2: if(ray.dir.z < 0.0f) swap(L, R); break;
		}

		stack.push(TraversalState(R, state.axis, state.range));
		stack.push(TraversalState(L, state.axis, state.range));
	}
	return false;
}

// Intersect closest triangle
inline __device__ bool intersect(const Geometry& geometry, Ray& ray, HitPoint& hp)
{
	bool hit = false;
	Stack<TraversalState, AURORA_INTSTACK_DEPTH> stack;

	stack.push(TraversalState(0, 0, make_float2(Epsilon, Infinity)));
	while(stack.size > 0) {
		TraversalState state = stack.pop();		
		if(state.index >= geometry.count)
			continue;

		const float* vertices1 = geometry.vertices + state.index * Geometry::TriangleParams;
		const float* vertices2 = geometry.vertices + (state.index+1) * Geometry::TriangleParams;

		const float2 slab = make_float2(
			fminf(min(vertices1, state.axis), min(vertices2, state.axis)),
			fmaxf(max(vertices1, state.axis), max(vertices2, state.axis)));

		if(!ray.intersect(slab, state.axis, state.range))
			continue;

		Primitive3 triangle1, triangle2;
		triangle1.readPoints(geometry.vertices + state.index * Geometry::TriangleParams);
		triangle2.readPoints(geometry.vertices + (state.index+1) * Geometry::TriangleParams);

		float u, v;
		float t;

		if(ray.intersect(triangle1, u, v, t) && t > Epsilon && t < ray.t) {
			hit   = true;
			ray.t = t;
			hp.u  = u;
			hp.v  = v;
			hp.triangleID = state.index;
		}
		if(ray.intersect(triangle2, u, v, t) && t > Epsilon && t < ray.t) {
			hit   = true;
			ray.t = t;
			hp.u  = u;
			hp.v  = v;
			hp.triangleID = state.index+1;
		}

		state.axis = (state.axis + 1) % 3;
		unsigned int L = 2*state.index + 2;
		unsigned int R = 2*state.index + 4;

		switch(state.axis) {
		case 0: if(ray.dir.x < 0.0f) swap(L, R); break;
		case 1: if(ray.dir.y < 0.0f) swap(L, R); break;
		case 2: if(ray.dir.z < 0.0f) swap(L, R); break;
		}

		stack.push(TraversalState(R, state.axis, state.range));
		stack.push(TraversalState(L, state.axis, state.range));
	}
	return hit;
}