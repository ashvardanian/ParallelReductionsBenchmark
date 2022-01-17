// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include "ComputeApp.hpp"
#include "HelperMethods.hpp"
#include "ResourcesManager.hpp"

using namespace nAV;
using namespace nCL;

inline static constexpr auto kFileReduce = pFilePath("Shared/OpenCL/ReduceSum.cl");
inline static constexpr auto kFileParallel = pFilePath("Shared/OpenCL/DataParallel.cl");

template <typename aFuncConfigure, typename aFuncCheck>
void gForEachOpenCLMethod(sBenchDescription xBench, sStr xProgramPath, SArr<sStr> xMethods,
                          SArr<sBenchResult> &xResults, aFuncConfigure xFuncConfigure, aFuncCheck xFuncCheck) {
    sOpenCL lCL;
    auto lDevIDs = lCL.fAddAllDevices();
    for (auto lDevID : lDevIDs) {
        sOpenCL::gPrintSpecs(lDevID);
        auto lDevPtr = lCL.fFindDevice(lDevID);
        auto lProgram = sResourcesManager::gShared().fLoad(xProgramPath);
        for (auto lMethod : xMethods) {
            auto lFunc = lProgram->fMakeFunc(lMethod);
            auto lCompiled = lDevPtr->fCompile(lFunc);

            sBenchResult lResultGPU;
            lResultGPU.mTechnology = "OpenCL";
            lResultGPU.mTask = xBench;
            lResultGPU.mDevice = sOpenCL::gGetSpecStr(lDevID, CL_DEVICE_NAME);
            lResultGPU.mMethod = lMethod;

            auto lAppPtr = lCompiled->fNewTask(lMethod);
            lAppPtr->mWorkDimensions = {xBench.mSize};
            lAppPtr->mGroupDimensions = {xBench.mSizeBatch};
            xFuncConfigure(lAppPtr, lMethod);
            bBool lFailed = false;

            lResultGPU.mDurationTotal = GMeasureTime([&] {
                try {
                    lResultGPU.mDurationCompute = GMeasureTime([&] { lDevPtr->fRunOnAnyQueue(lAppPtr); });
                    lResultGPU.mCheckSum = xFuncCheck(lAppPtr);
                } catch (std::runtime_error const &e) {
                    lFailed = true;
                    printf("%s\n", e.what());
                }
            });

            if (!lFailed)
                xResults.push_back(lResultGPU);
        }
    }
}

void nCL::gBenchReduce(sBenchDescription xBench, SArr<sReal> const &xArrFullCPU, SArr<sBenchResult> &yResults) {
    SArr<sReal> lArrPartialSumsCPU(xBench.mCountThreads);

    gForEachOpenCLMethod(
        xBench, kFileReduce,
        {"reduce_simple", "reduce_w_modulo", "reduce_in_shared", "reduce_w_sequential_addressing", "reduce_bi_step",
         "reduce_unrolled", "reduce_unrolled_fully", "reduce_w_brents_theorem"},
        yResults,
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            gResetOnCPU(lArrPartialSumsCPU);
            cl_int lN = static_cast<cl_int>(xBench.mSize);
            lAppPtr->fReallocVar(0)->fResetConstant(xArrFullCPU.data(), sizeof(sReal) * xBench.mSize);
            lAppPtr->fReallocVar(1)->fResetVariable(lArrPartialSumsCPU.data(),
                                                    sizeof(sReal) * lArrPartialSumsCPU.size());
            lAppPtr->fReallocVar(2)->fResetValue(&lN, sizeof(cl_int));
            lAppPtr->fReallocVar(3)->fResetIndependantBuffer(sizeof(sReal) * xBench.mCountThreads);
        },
        [&](SShared<sTask> lAppPtr) {
            lAppPtr->mVarPtrs[1]->fPullInto(lArrPartialSumsCPU.data(), sizeof(sReal) * lArrPartialSumsCPU.size());
            return std::accumulate(lArrPartialSumsCPU.begin(), lArrPartialSumsCPU.end(), 0);
        });
}

void nCL::gBenchParallel(sBenchDescription xBench, SArr<sReal> const &xArrFullCPU, SArr<sBenchResult> &yResults) {

    SArr<sReal> lArrResultCPU(xBench.mSize);
    gResetOnCPU(lArrResultCPU);

    auto fRunWithConfigurator = [&](SArr<sStr> xMethods, auto xConfigurator, bInt32 xOutputIndex) {
        gForEachOpenCLMethod(
            xBench, kFileParallel, xMethods, yResults,
            [&](SShared<sTask> lAppPtr, sStr lMethod) {
                gResetOnCPU(lArrResultCPU);
                lAppPtr->fReallocVar(0)->fResetConstant(xArrFullCPU.data(), sizeof(sReal) * xBench.mSize);
                lAppPtr->fReallocVar(xOutputIndex)->fResetVariable(lArrResultCPU.data(), sizeof(sReal) * xBench.mSize);
                xConfigurator(lAppPtr, lMethod);
            },
            [&](SShared<sTask> lAppPtr) {
                lAppPtr->mVarPtrs[xOutputIndex]->fPullInto(lArrResultCPU.data(), sizeof(sReal) * xBench.mSize);
                return lArrResultCPU.front();
            });
    };

    fRunWithConfigurator(
        {
            "gArithmAddConst",
            "gArithmMulConst",
        },
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            cl_float lX = static_cast<cl_float>(M_PI);
            lAppPtr->fReallocVar(1)->fResetValue(&lX, sizeof(cl_float));
        },
        2);

    fRunWithConfigurator(
        {
            "gArithmAddArr",
            "gArithmMulArr",
        },
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            lAppPtr->fReallocVar(1)->fResetConstant(xArrFullCPU.data(), sizeof(sReal) * xBench.mSize);
        },
        2);

    fRunWithConfigurator(
        {"gPowSqrt", "gPowExp", "gTrigCos", "gTrigCosNative", // "gTrigCos16",
         "gTrigCosPi", "gTrigCosPiManual"},
        [](SShared<sTask> lAppPtr, sStr lMethod) {}, 1);

    fRunWithConfigurator(
        {"gPowFltExp"},
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            cl_float lX = static_cast<cl_float>(3.f);
            lAppPtr->fReallocVar(2)->fResetValue(&lX, sizeof(cl_float));
        },
        1);

    fRunWithConfigurator(
        {
            "gPowPositiveBase",
            "gPowIntExp",
            "gPowInverseIntExp",
        },
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            cl_int lX = static_cast<cl_int>(12);
            lAppPtr->fReallocVar(2)->fResetValue(&lX, sizeof(cl_float));
        },
        1);

    fRunWithConfigurator(
        {
            "gMulAddConst",
            "gMulAddConstManual",
        },
        [&](SShared<sTask> lAppPtr, sStr lMethod) {
            cl_float lX = static_cast<cl_float>(M_PI);
            lAppPtr->fReallocVar(1)->fResetValue(&lX, sizeof(cl_float));
            lAppPtr->fReallocVar(2)->fResetValue(&lX, sizeof(cl_float));
        },
        3);
}
