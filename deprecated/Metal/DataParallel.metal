// Project: Semantic.Notes.
// Author: Ashot Vardanian.
// Created: 10/9/19.
// Copyright: See "License" file.
//

#include <metal_stdlib>
using namespace metal;

kernel void gArithmAddArr(device const float * xA,
                          device const float * xB,
                          device float * y,
                          uint lId [[thread_position_in_grid]]) {
    y[lId] = xA[lId] + xB[lId];
}
