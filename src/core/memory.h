/* Aurora Renderer
 * Copyright (c) 2013 Michal Siejak
 * Licensed under MIT open-source license, see COPYING.txt file for details.
 */

#pragma once

namespace Aurora {

enum MemoryPolicy
{
	HostMemory,
	DeviceMemory
};

void* malloc(const size_t size, MemoryPolicy policy);
void  memzero(void* ptr, const size_t size, MemoryPolicy policy);
void  memupload(void* dst, const void* src, const size_t size, MemoryPolicy policy);
void  free(void* ptr, MemoryPolicy policy);

} // Aurora