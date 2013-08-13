/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/scene.h>

#include <vector>

#include <maya/MObjectHandle.h>
#include <maya/MMatrix.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MItDag.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>

#include <maya/MImage.h>

#include <maya/MFnLambertShader.h>
#include <maya/MFnPhongShader.h>
#include <maya/MFnBlinnShader.h>

using namespace Aurora;

Scene::Scene()
{
	m_geometry.initialize();
}

Scene::~Scene()
{
	m_geometry.free();
}

unsigned int Scene::getConnectedIndex(const int type, const MObjectHandle& handle, const MString& attribute, const ObjectHash& hash)
{
	MPlugArray connections;

	const MPlug plug = MFnDependencyNode(handle.objectRef()).findPlug(attribute);
	plug.connectedTo(connections, true, false);

	for(unsigned int i=0; i<connections.length(); i++) {
		if(connections[i].node().apiType() == type) {
			const auto it = hash.find(MObjectHandle(connections[i].node()).hashCode());
			if(it != hash.end())
				return it->second;
		}
	}
	return 0;
}

unsigned int Scene::getIndexByHandle(const MObjectHandle& handle, const ObjectHash& hash)
{
	const auto it = hash.find(handle.hashCode());
	if(it == hash.end())
		return 0;
	return it->second;
}

void Scene::getLocalIndices(const MIntArray& polygonIndices, const MIntArray& triangleIndices, MIntArray& localIndices)
{
	for(unsigned int i=0; i<triangleIndices.length(); i++) {
		for(unsigned int j=0; j<polygonIndices.length(); j++) {
			if(triangleIndices[i] == polygonIndices[j]) {
				localIndices.append(j);
				break;
			}
		}
	}
}

MStatus Scene::updateMeshes(MDagPathArray& meshPaths, const ObjectHash& hShaders)
{
	unsigned int primitiveCount  = 0;
	const unsigned int meshCount = meshPaths.length();

	if(meshCount == 0) {
		m_geometry.free();
		return MS::kSuccess;
	}

	std::vector<MObject> objects;

	Transform* transforms;
	if(gpu::cudaHostAlloc(&transforms, meshCount * sizeof(Transform), cudaHostAllocMapped) != gpu::cudaSuccess)
		return MS::kInsufficientMemory;

	unsigned int objectsCount = 0;
	for(unsigned int i=0; i<meshCount; i++) {
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
			transforms[objectsCount].offset = primitiveCount;
			transforms[objectsCount].size   = currentPrimitiveCount;
			dagPath.inclusiveMatrix().get(transforms[objectsCount].worldMatrix);
			dagPath.inclusiveMatrixInverse().transpose().get(transforms[objectsCount].normalMatrix);

			objectsCount++;
			primitiveCount += currentPrimitiveCount;
			objects.push_back(dagPath.node());
		}
	}

	Geometry buffer;
	buffer.initialize();
	if(!m_geometry.resize(primitiveCount, Geometry::AllocDefault)) {
		gpu::cudaFreeHost(transforms);
		return MS::kInsufficientMemory;
	}
	if(!buffer.resize(primitiveCount, Geometry::AllocStaging)) {
		m_geometry.free();
		gpu::cudaFreeHost(transforms);
		return MS::kInsufficientMemory;
	}

	unsigned int vertexOffset   = 0;
	unsigned int normalOffset   = 0;
	unsigned int texcoordOffset = 0;
	for(auto obj=objects.begin(); obj!=objects.end(); obj++) {
		MFnMesh dagMesh(*obj);
		MItMeshPolygon polyIterator(*obj);

		for(; !polyIterator.isDone(); polyIterator.next()) {
			MIntArray polygonIndices;
			polyIterator.getVertices(polygonIndices);

			int numTriangles;
			polyIterator.numTriangles(numTriangles);

			for(int i=0; i<numTriangles; i++) {
				MPointArray positions;
				MIntArray   triangleIndices;
				MIntArray	indices;

				polyIterator.getTriangle(i, positions, triangleIndices);
				getLocalIndices(polygonIndices, triangleIndices, indices);

				// Store vertices (in format optimised for fast traversal)
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].x;
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].y;
				for(int k=0; k<3; k++)
					buffer.vertices[vertexOffset++] = (float)positions[k].z;

				// Store others
				for(int k=0; k<3; k++) {
					::MVector normal;
					::float2  texcoord;
					
					polyIterator.getNormal(indices[k], normal);
					buffer.normals[normalOffset++] = (float)normal.x;
					buffer.normals[normalOffset++] = (float)normal.y;
					buffer.normals[normalOffset++] = (float)normal.z;

					if(polyIterator.hasUVs()) {
						polyIterator.getUV(indices[k], texcoord);
						buffer.texcoords[texcoordOffset++] = (float)texcoord[0];
						buffer.texcoords[texcoordOffset++] = (float)texcoord[1];
					}
					else {
						buffer.texcoords[texcoordOffset++] = 0.0f;
						buffer.texcoords[texcoordOffset++] = 0.0f;
					}
				}
			}
		}

		MObjectArray dagShaders;
		MIntArray indices;

		dagMesh.getConnectedShaders(0, dagShaders, indices);
		for(unsigned int i=0; i<indices.length(); i++) {
			if(indices[i] == -1)
				buffer.shaders[i] = 0;
			else
				buffer.shaders[i] = (unsigned short)getIndexByHandle(MObjectHandle(dagShaders[indices[i]]), hShaders);
		}
	}

	buffer.padToEven(primitiveCount);
	buffer.copyToDeviceTransform(m_geometry, transforms, objectsCount);
	buffer.free();

	gpu::cudaFreeHost(transforms);

	if(!m_geometry.rebuild()) {
		m_geometry.free();
		return MS::kFailure;
	}

	return MS::kSuccess;
}

MStatus Scene::updateShaders(MDagPathArray& shaderPaths, const ObjectHash& hTextures, ObjectHash& hShaders)
{
	m_shaders.resize(shaderPaths.length());
	if(m_shaders.size == 0)
		return MS::kSuccess;

	Array<Shader, HostMemory> buffer;
	buffer.resize(m_shaders.size);

	for(unsigned int i=0; i<buffer.size; i++) {
		const MObject node = shaderPaths[i].node();
		const MFnLambertShader dagLambertShader(node);

		buffer[i].color        = make_float3(dagLambertShader.color());
		buffer[i].ambientColor = make_float3(dagLambertShader.ambientColor());
		buffer[i].diffuse      = dagLambertShader.diffuseCoeff();

		buffer[i].texture[Shader::ChannelColor] = getConnectedIndex(MFn::kFileTexture, MObjectHandle(node), "color", hTextures);
		
		switch(shaderPaths[i].apiType()) {
		case MFn::kLambert:
			{
				buffer[i].bsdf.type = BSDF::BSDF_Lambert;
			}
			break;
		case MFn::kPhong:
			{
				MFnPhongShader dagPhongShader(node);
				buffer[i].bsdf.type = BSDF::BSDF_Phong;
			}
			break;
		case MFn::kBlinn:
			{
				MFnBlinnShader dagBlinnShader(node);
				buffer[i].bsdf.type = BSDF::BSDF_Blinn;
			}
			break;
		default:
			buffer[i].bsdf.type = BSDF::BSDF_Lambert;
			break;
		}

		hShaders[MObjectHandle(node).hashCode()] = setID(i);
	}

	buffer.copyToDevice(m_shaders);
	return MS::kSuccess;
}

MStatus Scene::updateTextures(MDagPathArray& texturePaths, ObjectHash& hTextures)
{
	m_textures.resize(texturePaths.length());
	if(m_textures.size == 0)
		return MS::kSuccess;

	for(unsigned int i=0; i<texturePaths.length(); i++) {
		MObject dagNode = texturePaths[i].node();
		hTextures[MObjectHandle(dagNode).hashCode()] = setID(i);
	}
	return MS::kSuccess;
}

MStatus Scene::update(UpdateType type)
{
	MStatus status;

	MItDag   dagIterator(MItDag::kDepthFirst);
	MDagPath dagPath;

	MDagPathArray dagMeshes;
	MDagPathArray dagShaders;
	MDagPathArray dagTextures;

	for(; !dagIterator.isDone(); dagIterator.next()) {
		if(!dagIterator.getPath(dagPath))
			continue;

		// Instancing is not supported
		if(dagPath.isInstanced())
			continue;
		// Only process shapes
		if(dagPath.hasFn(MFn::kTransform))
			continue;

		// Shape node
		if(dagPath.hasFn(MFn::kMesh))
			dagMeshes.append(dagPath);
		// Shader node
		if(dagPath.hasFn(MFn::kLambert))
			dagShaders.append(dagPath);
		// Texture node
		if(dagPath.hasFn(MFn::kFileTexture))
			dagTextures.append(dagPath);
	}

	ObjectHash hShaders;
	ObjectHash hTextures;

	if(!(status = updateTextures(dagTextures, hTextures)))
		return status;
	if(!(status = updateShaders(dagShaders, hTextures, hShaders)))
		return status;
	if(!(status = updateMeshes(dagMeshes, hShaders)))
		return status;

	return MS::kSuccess;
}