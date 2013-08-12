/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class BSDF
{
public:
	enum BSDFType {
		BSDF_Lambert,
		BSDF_Phong,
		BSDF_Blinn,
	};
	BSDFType type;
};

} // Aurora