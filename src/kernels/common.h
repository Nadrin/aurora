/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

inline __host__ dim3 make_grid(const dim3& blockSize, const dim3& domainSize)
{
	dim3 gridSize;
	gridSize.x = (domainSize.x / blockSize.x) + (domainSize.x % blockSize.x ? 1 : 0);
	gridSize.y = (domainSize.y / blockSize.y) + (domainSize.y % blockSize.y ? 1 : 0);
	gridSize.z = (domainSize.z / blockSize.z) + (domainSize.z % blockSize.z ? 1 : 0);
	return gridSize;
}