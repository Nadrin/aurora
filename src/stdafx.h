/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef __CUDACC__
#include <cmath>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

namespace gpu {
#include <cuda_runtime.h>
}

#include "config.h"