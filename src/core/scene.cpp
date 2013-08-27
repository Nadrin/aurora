/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/scene.h>

#include <vector>

#include <maya/MObjectArray.h>
#include <maya/MObjectHandle.h>
#include <maya/MMatrix.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MDagPath.h>

#include <maya/MItDependencyNodes.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>

#include <maya/MImage.h>

#include <maya/MFnLambertShader.h>
#include <maya/MFnPhongShader.h>
#include <maya/MFnBlinnShader.h>

#include <maya/MFnLight.h>

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

MStatus Scene::updateMeshes(MObjectArray& nodes, const ObjectHash& hShaders)
{
	unsigned int primitiveCount  = 0;
	const unsigned int meshCount = nodes.length();

	if(meshCount == 0) {
		m_geometry.free();
		return MS::kSuccess;
	}

	MObjectArray objects;

	Transform* transforms;
	if(gpu::cudaHostAlloc(&transforms, meshCount * sizeof(Transform), cudaHostAllocMapped) != gpu::cudaSuccess)
		return MS::kInsufficientMemory;

	unsigned int objectsCount = 0;
	for(unsigned int i=0; i<meshCount; i++) {
		const MObject node = nodes[i];

		MDagPath dagPath;
		MFnDagNode(node).getPath(dagPath);
		if(dagPath.isInstanced())
			continue;

		MItMeshPolygon polyIterator(node);

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
			objects.append(node);
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
	unsigned int shaderOffset   = 0;

	for(unsigned int obj=0; obj<objects.length(); obj++) {
		MFnMesh dagMesh(objects[obj]);
		
		MObjectArray shaderGroups;
		MPlugArray shaderPlugs;
		MIntArray  shaderIDs;

		dagMesh.getConnectedShaders(0, shaderGroups, shaderIDs);
		for(unsigned int i=0; i<shaderGroups.length(); i++) {
			const MPlug surfaceShader = MFnDependencyNode(shaderGroups[i]).findPlug("surfaceShader");
			surfaceShader.connectedTo(shaderPlugs, true, false);
		}

		MItMeshPolygon polyIterator(objects[obj]);
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

				// Store shader indices
				const unsigned int polygonIndex = polyIterator.index();
				if(shaderIDs[polygonIndex] == -1)
					buffer.shaders[shaderOffset++] = 0;
				else {
					buffer.shaders[shaderOffset++] = 
						(unsigned short)getIndexByHandle(MObjectHandle(shaderPlugs[shaderIDs[polygonIndex]].node()), hShaders);
				}
			}
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
	m_geometry.generateTB();

	return MS::kSuccess;
}

MStatus Scene::updateShaders(MObjectArray& nodes, const ObjectHash& hTextures, ObjectHash& hShaders)
{
	m_shaders.resize(nodes.length());
	if(m_shaders.size == 0)
		return MS::kSuccess;

	Array<Shader, HostMemory> buffer;
	buffer.resize(m_shaders.size);

	for(unsigned int i=0; i<nodes.length(); i++) {
		const MObject node = nodes[i];
		const MFnLambertShader dagLambertShader(node);

		buffer[i].diffuseColor  = make_float3(dagLambertShader.color());
		buffer[i].emissionColor = make_float3(dagLambertShader.incandescence());
		buffer[i].diffuse       = dagLambertShader.diffuseCoeff();

		float _unused;
		dagLambertShader.incandescence().get(MColor::kHSV, _unused, _unused, buffer[i].emission);

		//buffer[i].texture[Shader::ChannelColor] = getConnectedIndex(MFn::kFileTexture, MObjectHandle(node), "color", hTextures);
		
		switch(node.apiType()) {
		case MFn::kLambert:
			{
				buffer[i].type = Shader::LambertShader;
			}
			break;
		case MFn::kPhong:
			{
				MFnPhongShader dagPhongShader(node);
				buffer[i].type = Shader::PhongShader;
			}
			break;
		case MFn::kBlinn:
			{
				MFnBlinnShader dagBlinnShader(node);
				buffer[i].type = Shader::BlinnShader;
			}
			break;
		default:
			buffer[i].type = Shader::LambertShader;
			break;
		}

		hShaders[MObjectHandle(node).hashCode()] = setID(i);
	}

	buffer.copyToDevice(m_shaders);
	return MS::kSuccess;
}

MStatus Scene::updateTextures(MObjectArray& nodes, ObjectHash& hTextures)
{
	m_textures.resize(nodes.length());
	if(m_textures.size == 0)
		return MS::kSuccess;

	for(unsigned int i=0; i<nodes.length(); i++) {
		const MObject node = nodes[i];
		hTextures[MObjectHandle(node).hashCode()] = setID(i);
	}
	return MS::kSuccess;
}

MStatus Scene::updateLights(MObjectArray& nodes)
{
	m_lights.resize(nodes.length());
	if(m_lights.size == 0)
		return MS::kSuccess;

	Array<Light, HostMemory> buffer;
	buffer.resize(m_lights.size);

	for(unsigned int i=0; i<nodes.length(); i++) {
		const MObject node = nodes[i];
		const MFnLight dagLight(node);

		buffer[i].color     = make_float3(dagLight.color());
		buffer[i].intensity = dagLight.intensity();
		buffer[i].samples   = dagLight.numShadowSamples();

		MTransformationMatrix transform;
		if(node.apiType() == MFn::kPointLight || node.apiType() == MFn::kAreaLight) {
			MDagPath dagPath;
			dagLight.getPath(dagPath);
			transform = dagPath.inclusiveMatrix();
		}

		switch(node.apiType()) {
		case MFn::kAmbientLight:
			{
				buffer[i].type    = Light::AmbientLight;
				buffer[i].area    = 0.0f;
			}
			break;
		case MFn::kPointLight:
			{
				buffer[i].type     = Light::PointLight;
				buffer[i].position = make_float3(transform.getTranslation(MSpace::kWorld));
				buffer[i].area     = 0.0f;
			}
			break;
		case MFn::kDirectionalLight:
			{
				buffer[i].type      = Light::DirectionalLight;
				buffer[i].direction = make_float3(dagLight.lightDirection(0, MSpace::kWorld));
				buffer[i].area      = 0.0f;
			}
			break;
		case MFn::kAreaLight:
			{

				buffer[i].type      = Light::AreaLight;
				buffer[i].direction = make_float3(dagLight.lightDirection(0, MSpace::kWorld));

				const float3 v1     = transform.asMatrix() * make_float3(-0.5f, 0.5f, 0.0f);
				const float3 v2     = transform.asMatrix() * make_float3(0.5f, 0.5f, 0.0f);
				const float3 v3     = transform.asMatrix() * make_float3(-0.5f, -0.5f, 0.0f);

				buffer[i].position  = v1;
				buffer[i].e1        = v2 - v1;
				buffer[i].e2        = v3 - v1;
				buffer[i].area      = length(buffer[i].e1) * length(buffer[i].e2);
			}
			break;
		default:
			{
				buffer[i].type      = Light::AmbientLight;
				buffer[i].samples   = 0;
			}
			break;
		}
	}

	buffer.copyToDevice(m_lights);
	return MS::kSuccess;
}

MStatus Scene::update(UpdateType type)
{
	MStatus status;

	MObjectArray meshes;
	MObjectArray shaders;
	MObjectArray textures;
	MObjectArray lights;

	MItDependencyNodes depIterator;
	for(; !depIterator.isDone(); depIterator.next()) {
		const MObject node = depIterator.thisNode();
		
		// Skip transform nodes
		if(node.hasFn(MFn::kTransform))
			continue;

		// Mesh shape node
		if(node.hasFn(MFn::kMesh) && type == Scene::UpdateFull)
			meshes.append(node);
		// Texture node
		if(node.hasFn(MFn::kFileTexture) && type == Scene::UpdateFull)
			textures.append(node);

		// Shader node
		if(node.hasFn(MFn::kLambert))
			shaders.append(node);
		// Light nodes
		if(node.hasFn(MFn::kLight))
			lights.append(node);
	}

	ObjectHash hShaders;
	ObjectHash hTextures;

	if(type == Scene::UpdateFull) {
		if(!(status = updateTextures(textures, hTextures)))
			return status;
	}
	if(m_shaders.size == shaders.length() || type == Scene::UpdateFull) {
		if(!(status = updateShaders(shaders, hTextures, hShaders)))
			return status;
	}
	if(!(status = updateLights(lights)))
		return status;
	if(type == Scene::UpdateFull) {
		if(!(status = updateMeshes(meshes, hShaders)))
			return status;
	}

	return MS::kSuccess;
}

void Scene::free()
{
	m_geometry.free();
	m_textures.resize(0);
	m_shaders.resize(0);
	m_lights.resize(0);
}