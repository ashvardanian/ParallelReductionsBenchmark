// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include "HelperMethods.hpp"
#include "ComputeApp.hpp"
#include "ResourcesManager.hpp"
#include <math.h>

using namespace nAV;

template <typename aFuncRun>
void gForCPUMethod(sBenchDescription xBench, sStr xMethod,
                   SArr<sBenchResult> & yResults,
                   aFuncRun xFuncRun) {
    sBenchResult lResultCPU;
    lResultCPU.mTechnology = "CPU";
    lResultCPU.mDevice = "Intel(R) Core(TM) i7-7820HQ CPU @ 2.90GHz";
    lResultCPU.mTask = xBench;
    lResultCPU.mTask.mCountThreads = 1;
    lResultCPU.mMethod = xMethod;
    lResultCPU.mDurationTotal = GMeasureTime([&] {
        lResultCPU.mCheckSum = xFuncRun();
    });
    lResultCPU.mDurationCompute = lResultCPU.mDurationTotal;
    yResults.push_back(lResultCPU);
}

void nCPU::gBenchReduce(sBenchDescription xBench,
                        SArr<sReal> const & xArrFullCPU,
                        SArr<sBenchResult> & yResults) {
    gForCPUMethod(xBench, "C", yResults, [&] {
        sReal lTotal = 0.0f;
        for (bSize lIdx = 0; lIdx < xBench.mSize; lIdx ++)
            lTotal += xArrFullCPU[lIdx];
        return lTotal;
    });
    gForCPUMethod(xBench, "STL", yResults, [&] {
        return std::accumulate(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, 0.f);
    });
}

void nCPU::gBenchParallel(sBenchDescription xBench,
                          SArr<sReal> const & xArrFullCPU,
                          SArr<sBenchResult> & yResults) {
    constexpr sReal kArgConst = M_PI;
    SArr<sReal> lArrResultCPU(xBench.mSize);
    
    gResetOnCPU(lArrResultCPU);
    gForCPUMethod(xBench, "gArithmAddConst", yResults, [&] {
        std::transform(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, lArrResultCPU.begin(), [](sReal l) {
            return l + kArgConst;
        });
        return lArrResultCPU.front();
    });
    
    gResetOnCPU(lArrResultCPU);
    gForCPUMethod(xBench, "gArithmMulConst", yResults, [&] {
        std::transform(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, lArrResultCPU.begin(), [](sReal l) {
            return l * kArgConst;
        });
        return lArrResultCPU.front();
    });

    gResetOnCPU(lArrResultCPU);
    gForCPUMethod(xBench, "gPowSqrt", yResults, [&] {
        std::transform(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, lArrResultCPU.begin(), [](sReal l) {
            return sqrt(l);
        });
        return lArrResultCPU.front();
    });

    gResetOnCPU(lArrResultCPU);
    gForCPUMethod(xBench, "gPowExp", yResults, [&] {
        std::transform(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, lArrResultCPU.begin(), [](sReal l) {
            return exp(l);
        });
        return lArrResultCPU.front();
    });

    gResetOnCPU(lArrResultCPU);
    gForCPUMethod(xBench, "gTrigCos", yResults, [&] {
        std::transform(xArrFullCPU.cbegin(), xArrFullCPU.cbegin() + xBench.mSize, lArrResultCPU.begin(), [](sReal l) {
            return cos(l);
        });
        return lArrResultCPU.front();
    });
}

