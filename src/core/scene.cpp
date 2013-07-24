/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/scene.h>
#include <util/transformation.h>

#include <vector>

#include <maya/MMatrix.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MItDag.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>

/*
#include <maya/MGeometry.h>
#include <maya/MGeometryData.h>
#include <maya/MGeometryManager.h>
#include <maya/MGeometryRequirements.h>
*/

using namespace Aurora;

Scene::Scene()
{
	m_geometry.initialize();
}

Scene::~Scene()
{
	m_geometry.free();
}

MStatus Scene::updateMeshes(MDagPathArray& meshPaths, MStatus& status)
{
	unsigned int primitiveCount  = 0;
	std::vector<MObject> objects;

	for(unsigned int i=0; i<meshPaths.length(); i++) {
		MDagPath& dagPath = meshPaths[i];
		MItMeshPolygon polyIterator(dagPath.node());

		unsigned int currentPrimitiveCount = 0;
		bool isValidObject = true;
		for(; !polyIterator.isDone(); polyIterator.next()) {
			int numTriangles;
			if(!polyIterator.hasValidTriangulation()) {
				isValidObject = false;
				break;
			}

			polyIterator.numTriangles(numTriangles);
			currentPrimitiveCount += numTriangles;
		}

		if(isValidObject) {
			primitiveCount += currentPrimitiveCount;
			objects.push_back(dagPath.node());
		}
	}

	Geometry buffer;
	buffer.initialize();
	if(!m_geometry.resize(primitiveCount, Geometry::AllocDefault)) {
		return status = MS::kInsufficientMemory;
	}
	if(!buffer.resize(primitiveCount, Geometry::AllocStaging)) {
		m_geometry.free();
		return status = MS::kInsufficientMemory;
	}

	unsigned int vertexOffset = 0;
	unsigned int normalOffset = 0;
	for(auto obj=objects.begin(); obj!=objects.end(); obj++) {
		MItMeshPolygon polyIterator(*obj);

		for(; !polyIterator.isDone(); polyIterator.next()) {
			int numTriangles;
			polyIterator.numTriangles(numTriangles);

			for(int i=0; i<numTriangles; i++) {
				MPointArray positions;
				MIntArray   indices;
				polyIterator.getTriangle(i, positions, indices, MSpace::kWorld);

				// Store vertices (in format optimised for fast traversal)
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].x;
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].y;
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].z;

				// Store normals
				for(int k=0; k<3; k++) {
					MVector normal;
					polyIterator.getNormal(indices[k], normal, MSpace::kWorld);
					buffer.normals[normalOffset++] = (float)normal.x;
					buffer.normals[normalOffset++] = (float)normal.y;
					buffer.normals[normalOffset++] = (float)normal.z;
				}
			}
		}
	}

	buffer.padToEven(primitiveCount);
	buffer.copyToDevice(m_geometry);
	buffer.free();
	return status = MS::kSuccess;
}

MStatus Scene::update(const int updateMask)
{
	MStatus status;

	MItDag   dagIterator(MItDag::kDepthFirst);
	MDagPath dagPath;

	MDagPathArray dagMeshes;

	for(; !dagIterator.isDone(); dagIterator.next()) {
		if(!dagIterator.getPath(dagPath))
			continue;

		// Transform extended to shape
		if(dagPath.hasFn(MFn::kMesh) && dagPath.hasFn(MFn::kTransform))
			continue;

		// Shape node
		if((updateMask & Scene::NodeGeometry) && dagPath.hasFn(MFn::kMesh))
			dagMeshes.append(dagPath);
	}

	if(dagMeshes.length() > 0) {
		if(!updateMeshes(dagMeshes, status))
			return status;
	}

	return MS::kSuccess;
}