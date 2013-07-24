/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

class Transformation
{
public:
	float matrix[4][4];
	unsigned int start;
	unsigned int end;
public:
	__host__ Transformation() : start(0), end(0)
	{
		memset(matrix, 0, sizeof(matrix));
	}

	__host__ Transformation(const double mv[4][4], const unsigned int start, const unsigned int end)
	{
		for(int i=0; i<4; i++) {
			for(int j=0; j<4; j++) {
				this->matrix[i][j] = (float)mv[i][j];
			}
		}

		this->start = start;
		this->end   = end;
	}
};

} // Aurora