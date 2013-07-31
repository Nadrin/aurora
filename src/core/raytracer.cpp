/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#include <stdafx.h>
#include <core/raytracer.h>
#include <kernels/kernels.h>

#include <maya/MFnCamera.h>

using namespace Aurora;

void Raytracer::generateRays(const MDagPath& camera, const Dim& size, const Rect& region, Ray* rays)
{
	MFnCamera dagCamera(camera);
	MPoint camEyePoint   = dagCamera.eyePoint(MSpace::kWorld);
	MVector camDirection = dagCamera.viewDirection(MSpace::kWorld);
	double camAspect     = dagCamera.aspectRatio();

	double camFovX, camFovY;
	dagCamera.getPortFieldOfView(size.width, size.height, camFovX, camFovY);

	Camera rayCamera(
		make_float3((float)camEyePoint.x, (float)camEyePoint.y, (float)camEyePoint.z),
		make_float3((float)camDirection.x, (float)camDirection.y, (float)camDirection.z),
		make_float2((float)camFovX, (float)camFovY), (float)camAspect);

	cudaGenerateRays(region, rayCamera, rays);
}