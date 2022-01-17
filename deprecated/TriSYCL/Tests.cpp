// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include <CL/sycl.hpp>
#include <SYCL/sycl.hpp>
#include "HelperAliasis.hpp"
#include "HelperMethods.hpp"
#include "Shared/OpenCL/ComputeApp.hpp"

namespace nSy = cl::sycl;

template <typename aFuncRun>
void gForMethod(sBenchDescription xBench, sStr xMethod,
                   SArr<sBenchResult> & yResults,
                   nSy::device xDev,
                   aFuncRun xFuncRun) {
    sBenchResult lResultCPU;
    lResultCPU.mTechnology = "SyCL";
    lResultCPU.mDevice = "Unknown"; // nCL::sOpenCL::gGetName(xDev.get());
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

void nSyCL::gBenchParallel(sBenchDescription xBench,
                           SArr<sReal> const & xArrFullCPU,
                           SArr<sBenchResult> & yResults) {
    SArr<sReal> lArrOnCPU(xBench.mSize);
    SArr<nSy::device> const lDevs = cl::sycl::device::get_devices();
    for (nSy::device const & lDev : lDevs) {
        if (lDev.is_host())
            continue;
        
        nSy::queue lQueue(lDev);
        nSy::buffer<sReal, 1> lBuffA { xArrFullCPU.data(), xBench.mSize };
        nSy::buffer<sReal, 1> lBuffB { xArrFullCPU.data(), xBench.mSize };
        nSy::buffer<sReal, 1> lBuffOnCPU { lArrOnCPU.data(), xBench.mSize };
        
        gForMethod(xBench, "gArithmAddConst", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                auto lHandleInB = lBuffB.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = lHandleInA[xIdx] + lHandleInB[xIdx];
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gArithmMulConst", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::read_write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                auto lHandleInB = lBuffB.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = lHandleInA[xIdx] * lHandleInB[xIdx];
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gTrigSin", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::read_write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = sin(lHandleInA[xIdx]);
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gTrigCos", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::read_write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = cos(lHandleInA[xIdx]);
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gPowExp", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::read_write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = exp(lHandleInA[xIdx]);
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
        gForMethod(xBench, "gPowSqrt", yResults, lDev, [&] {
            lQueue.submit([&](nSy::handler & xHandle) {
                auto lHandleOut = lBuffOnCPU.get_access<nSy::access::mode::read_write>(xHandle);
                auto lHandleInA = lBuffA.get_access<nSy::access::mode::read>(xHandle);
                xHandle.parallel_for<class nothing>(xBench.mSize, [=] (nSy::id<1> xIdx) {
                    lHandleOut[xIdx] = sqrt(lHandleInA[xIdx]);
                });
            });
            lQueue.wait();
            return lArrOnCPU.front();
        });
    }
}
