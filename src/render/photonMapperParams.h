/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

struct PhotonMapperParams
{
	unsigned int numPhotons;
	unsigned int numEmitters;
	unsigned int numLights;
	unsigned int numHitPoints;
	unsigned int numEmitterSamples;

	unsigned short maxPhotonDepth;
	unsigned short maxRayDepth;

	float* lightCDF;
};

} // Aurora