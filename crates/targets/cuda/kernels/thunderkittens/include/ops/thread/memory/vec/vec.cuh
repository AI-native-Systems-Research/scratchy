/**
 * @file
 * @brief An aggregate header of warp memory operations on vectors, where a single warp loads or stores data on its own.
 */

#pragma once

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#include "tma.cuh"
#endif
