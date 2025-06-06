/*
 * Defines all global variables (host + device).
 * Since we are not using headers, both declaration and definition appear here.
 */

#include <cstdint>

int khost;

__device__ int k;
__device__ int arrayPosition = 0;
