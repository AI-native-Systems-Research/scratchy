/**
 * @file
 * @brief An aggregate header file for all the system-wide types defined by ThunderKittens.
 */

#pragma once

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)
#ifndef KITTENS_NO_HOST
#include "ipc.cuh"
#include "vmm.cuh"
#endif
#include "pgl.cuh"
#endif
