// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

typedef float sReal;
typedef int sIdx;
typedef size_t bSize;

/**
 *  Most of the functions support float, float2, float4, float8, or float16
 *  as the type for the arguments.
 *  If extended with "cl_khr_fp64", generic type name gentype may indicate
 *  double and double{2|4|8|16} as arguments and return values.
 *  If extended with "cl_khr_fp16", generic type name gentype may indicate
 *  half and half{2|4|8|16} as arguments and return values.
 */
typedef float4 sReal16;

/*
 *  The table below describes the list of built-in math functions.
 *  These functions can take scalar or vector arguments.
 *  https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/mathFunctions.html
 *
 *  Trigonomtry:
 *  -   cos, cosh, cospi
 *  -   sin, sincos, sinh, sinpi
 *  -   tan, tanh, tanpi
 *  -   acos, acosh, acospi
 *  -   asin, asinh, asinpi
 *  -   atan, atan2, atanh, atanpi, atan2pi
 *  Powers:
 *  -   log, log2, log10, log1p, logb, ilogb
 *  -   exp, exp2, exp10, expm1, ldexp, (x*2^n)
 *  -   pow, pown, powr
 *  -   rootn (x^(1/y)), rsqrt, sqrt, cbrt, hypot(sqrt(x^2+y^2))
 *  Binary representation:
 *  -   round, ceil, floor, trunc, rint
 *  -   fract (fmin(x-floor(x), 0x1.fffffep-1f)), frexp, nextafter, nan
 *  Arithmetics:
 *  -   fmax, fmin, fabs, copysign
 *  -   fmod, remainder (x-round(x/y)*y), fma (a*b+c), fdim (max(x-y, 0)), remquo (x-n*y)
 *  Special math:
 *  -   tgamma, lgamma, lgamma_r
 *  -   erfc, erf
 *  -   mad, modf
 */

#pragma mark - Arithmetics (vector-vector, vector-scalar)

__kernel void gArithmAddConst(__global sReal const * xArr,
                              sReal const xNum,
                              __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xArr[lId] + xNum;
}

__kernel void gArithmMulConst(__global sReal const * xArr,
                              sReal const xNum,
                              __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xArr[lId] * xNum;
}

__kernel void gArithmAddArr(__global sReal const * xA,
                            __global sReal const * xB,
                            __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xA[lId] + xB[lId];
}

__kernel void gArithmMulArr(__global sReal const * xA,
                            __global sReal const * xB,
                            __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xA[lId] * xB[lId];
}

__kernel void gArithmMulConst4(__global float4 const * xArr,
                               sReal const xNum,
                               __global float4 * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xArr[lId] * xNum;
}

__kernel void gMulAddConst(__global sReal const * xArr,
                           sReal const xNumMultiplier,
                           sReal const xNumSum,
                           __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = fma(xArr[lId],
                    xNumMultiplier, xNumSum);
}

__kernel void gMulAddConstManual(__global sReal const * xArr,
                                 sReal const xNumMultiplier,
                                 sReal const xNumSum,
                                 __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xArr[lId] * xNumMultiplier + xNumSum;
}

#pragma mark - Powers

__kernel void gPowFltExp(__global sReal const * xArr,
                         __global sReal * yArr,
                         sReal const xPower) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = pow(xArr[lId], xPower);
}

__kernel void gPowPositiveBase(__global sReal const * xArr,
                               __global sReal * yArr,
                               sIdx const xPower) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = powr(xArr[lId], xPower);
}

__kernel void gPowIntExp(__global sReal const * xArr,
                         __global sReal * yArr,
                         sIdx const xPower) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = pown(xArr[lId], xPower);
}

__kernel void gPowIntExpManual(__global sReal const * xArr,
                               __global sReal * yArr,
                               sIdx const xPower) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = xArr[lId];
    for (sIdx i = 1; i < xPower; i ++) {
        yArr[lId] *= yArr[lId];
    }
}

__kernel void gPowSqrt(__global sReal const * xArr,
                       __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = sqrt(xArr[lId]);
}

__kernel void gPowInverseIntExp(__global sReal const * xArr,
                                __global sReal * yArr,
                                sIdx const xPower) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = rootn(xArr[lId], xPower);
}

__kernel void gPowExp(__global sReal const * xArr,
                      __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = exp(xArr[lId]);
}

#pragma mark - Trigonomtry

__kernel void gTrigCos(__global sReal const * xArr,
                       __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = cos(xArr[lId]);
}

__kernel void gTrigCosNative(__global sReal const * xArr,
                             __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = native_cos(xArr[lId]);
}

__kernel void gTrigCos16(__global sReal16 const * xArr,
                         __global sReal16 * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = cos(xArr[lId]);
}

__kernel void gTrigCosPi(__global sReal const * xArr,
                         __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = cospi(xArr[lId]);
}

__kernel void gTrigCosPiManual(__global sReal const * xArr,
                               __global sReal * yArr) {
    sIdx const lId = get_global_id(0);
    yArr[lId] = cos(xArr[lId] * 3.14159265359f);
}

