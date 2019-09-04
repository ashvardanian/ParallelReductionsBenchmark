// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include "HelperAliasis.hpp"
#include "HelperMethods.hpp"

#include <arrayfire.h>
#include <af/array.h>

using namespace nAF;

template <typename aFuncRun>
void gForMethod(sBenchDescription xBench, sStr xMethod,
                SArr<sBenchResult> & yResults,
                sStr xDevName,
                aFuncRun xFuncRun) {
    sBenchResult lResultCPU;
    lResultCPU.mTechnology = "ArrayFire";
    lResultCPU.mDevice = xDevName;
    lResultCPU.mTask = xBench;
    lResultCPU.mTask.mSizeBatch = 0;
    lResultCPU.mTask.mCountThreads = 0;
    lResultCPU.mMethod = xMethod;
    lResultCPU.mDurationTotal = GMeasureTime([&] {
        lResultCPU.mCheckSum = xFuncRun();
    });
    lResultCPU.mDurationCompute = lResultCPU.mDurationTotal;
    yResults.push_back(lResultCPU);
}

template <typename aFuncRun>
void gForEachDevice(aFuncRun xFuncRun) {
    for (bInt32 lDevID = 0; lDevID < af::getDeviceCount(); lDevID ++) {
        af::setDevice(lDevID);
        sStr lDev;
        {
            char lName[129] = { 0 };
            af::deviceInfo(lName, nullptr, nullptr, nullptr);
            lDev = lName;
        }
        xFuncRun(lDev);
    }
}

void nAF::gBenchParallel(sBenchDescription xBench,
                         SArr<sReal> const & xArrFullCPU,
                         SArr<sBenchResult> & yResults) {
    SArr<sReal> lArrOnCPU(xBench.mSize);
    gForEachDevice([&](sStr lDev) {
        
        af::array lBuffA { static_cast<dim_t>(xBench.mSize), xArrFullCPU.data() };
        af::array lBuffB { static_cast<dim_t>(xBench.mSize), xArrFullCPU.data() };
        af::array lBuffOnCPU { static_cast<dim_t>(xBench.mSize) };
        
        gForMethod(xBench, "gArithmAddConst", yResults, lDev, [&] {
            lBuffOnCPU = lBuffA + M_PI;
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gArithmMulConst", yResults, lDev, [&] {
            lBuffOnCPU = lBuffA * M_PI;
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gTrigSin", yResults, lDev, [&] {
            lBuffOnCPU = af::sin(lBuffA);
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gTrigCos", yResults, lDev, [&] {
            lBuffOnCPU = af::cos(lBuffA);
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gPowExp", yResults, lDev, [&] {
            lBuffOnCPU = af::exp(lBuffA);
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gPowSqrt", yResults, lDev, [&] {
            lBuffOnCPU = af::sqrt(lBuffA);
            lBuffOnCPU.host(lArrOnCPU.data());
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gReduceSimple", yResults, lDev, [&] {
            return af::sum<bFlt32>(lBuffA);
        });
    });
}

void nAF::gBenchReduce(sBenchDescription xBench,
                       SArr<sReal> const & xArrFullCPU,
                       SArr<sBenchResult> & yResults) {
    SArr<sReal> lArrOnCPU(xBench.mSize);
    gForEachDevice([&](sStr lDev) {
        
        af::array lBuff { static_cast<dim_t>(xBench.mSize), xArrFullCPU.data() };
        gForMethod(xBench, "gReduceSimple", yResults, lDev, [&] {
            return af::sum<bFlt32>(lBuff);
        });
    });
}
