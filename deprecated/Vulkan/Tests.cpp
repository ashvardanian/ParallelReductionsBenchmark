// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include "ComputeApp.hpp"
#include "ResourcesManager.hpp"
#include "HelperLodepng.h" // Used for png encoding.
#include <chrono>
#include <thread>

using namespace nAV;
using namespace nVK;
using namespace nFlow;

void nVK::gBenchMandelbrot() {
    
    bSize constexpr lImgChannels = 4;
    bSize constexpr lImgWidth = 3200;
    bSize constexpr lImgHeight = 2400;
    
    auto & lRes = sResourcesManager::gShared();
    auto lSource = lRes.fLoad(pFilePath("Shared/Vulkan/Mandelbrot.spv"));
    sArgDescriptor lArg;
    lArg.mName = "buffer";
    lArg.mPurpose = eArgPurpose::kOutput;
    lArg.mFixedSize = lImgWidth * lImgHeight * lImgChannels * sizeof(bFlt32);
    auto lFunc = lSource->fMakeFunc("main", { lArg });
    
    auto & lVk = sVulkan::gShared();
    auto lDevPtr = lVk.fAddAllDevices().front();
    auto lCompiled = lDevPtr->fCompile(lFunc);
    auto lTaskPtr = lCompiled->fNewTask();
    {
        lTaskPtr->fReallocVar(0);
        lTaskPtr->mWorkDimensions = { lImgWidth, lImgHeight };
    }
    lDevPtr->fRunOnAnyQueue(lTaskPtr);
    {
        SArr<bFlt32> lImgRGBAFloats(lImgWidth * lImgHeight * lImgChannels * sizeof(bFlt32));
        lTaskPtr->mVarPtrs[0]->fPullInto(lImgRGBAFloats.data(), lImgRGBAFloats.size());
        
        // Get the color data from the buffer, and cast it to bytes.
        SArr<bByte> lImgRGBABytes(lImgWidth * lImgHeight * lImgChannels * sizeof(bFlt32));
        std::transform(lImgRGBAFloats.cbegin(), lImgRGBAFloats.cend(), lImgRGBABytes.begin(), [](bFlt32 lV) {
            return bByte { bUInt8(255.0f * lV) };
        });
        
        // Now we save the acquired color data to a .png.
        auto const error = lodepng::encode("mandelbrot.png", (unsigned char *)lImgRGBABytes.data(), lImgWidth, lImgHeight);
        if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
    }
}

template <typename aFuncConfigure, typename aFuncCheck>
void gForEachVulkanMethod(sBenchDescription xBench,
                          SArr<sStr> xProgramPaths,
                          SArr<sStr> xMethodsNames,
                          SArr<sBenchResult> & xResults,
                          aFuncConfigure xFuncConfigure,
                          aFuncCheck xFuncCheck) {
    sVulkan lCL;
    auto lDevPtrs = lCL.fAddAllDevices();
    for (auto lDevPtr : lDevPtrs) {
        
        for (bSize lIdx = 0; lIdx < xProgramPaths.size(); lIdx ++) {
            auto lMethod = xMethodsNames[lIdx];
            auto lPath = xProgramPaths[lIdx];
            auto lProgram = sResourcesManager::gShared().fLoad(lPath);
            SArr<sArgDescriptor> lArgs;
            {
                sArgDescriptor lArgXA;
                lArgXA.mName = "xA";
                lArgXA.mPurpose = eArgPurpose::kInput;
                lArgXA.mFixedSize = 0;
                sArgDescriptor lArgXB;
                lArgXB.mName = "xB";
                lArgXB.mPurpose = eArgPurpose::kInput;
                lArgXB.mFixedSize = 0;
                sArgDescriptor lArgY;
                lArgY.mName = "y";
                lArgY.mPurpose = eArgPurpose::kOutput;
                lArgY.mFixedSize = 0;
                lArgs = { lArgXA, lArgXB, lArgY };
            }
            auto lFunc = lProgram->fMakeFunc("main", lArgs);
            auto lCompiled = lDevPtr->fCompile(lFunc);
            
            sBenchResult lResultGPU;
            lResultGPU.mTechnology = "Vulkan";
            lResultGPU.mTask = xBench;
            lResultGPU.mDevice = lDevPtr->mFeatures.mName;
            lResultGPU.mMethod = lMethod;
            
            auto lAppPtr = lCompiled->fNewTask();
            lAppPtr->mWorkDimensions = { xBench.mSize };
            xFuncConfigure(lAppPtr, lMethod);
            
            lResultGPU.mDurationTotal = GMeasureTime([&] {
                try {
                    lResultGPU.mDurationCompute = GMeasureTime([&] {
                        lDevPtr->fRunOnAnyQueue(lAppPtr);
                    });
                    lResultGPU.mCheckSum = xFuncCheck(lAppPtr);
                } catch (std::runtime_error const & e) {
                    printf("%s\n", e.what());
                }
            });
            
            if (lResultGPU.mCheckSum != 0)
                xResults.push_back(lResultGPU);
        }
    }
}

void nVK::gBenchParallel(sBenchDescription xBench,
                         SArr<sReal> const & xArrFullCPU,
                         SArr<sBenchResult> & yResults) {
    SArr<sReal> lArrResultCPU(xBench.mSize);
    
    gForEachVulkanMethod(xBench, {
        pFilePath("Shared/Vulkan/DataParallel/ArithmMulArr.spv"),
        pFilePath("Shared/Vulkan/DataParallel/ArithmAddArr.spv"),
        pFilePath("Shared/Vulkan/DataParallel/PowSqrt.spv"),
        pFilePath("Shared/Vulkan/DataParallel/PowExp.spv"),
        pFilePath("Shared/Vulkan/DataParallel/TrigCos.spv"),
    }, {
        "gArithmMulArr", "gArithmAddArr",
        "gPowSqrt", "gPowExp",
        "gTrigCos"
    }, yResults, [&](SShared<sTask> lAppPtr, sStr lMethod) {
        lAppPtr->fReallocVar(0)->fResetConstant(xArrFullCPU.data(), sizeof(sReal) * xBench.mSize);
        lAppPtr->fReallocVar(1)->fResetConstant(xArrFullCPU.data(), sizeof(sReal) * xBench.mSize);
        lAppPtr->fReallocVar(2)->fResetVariable(lArrResultCPU.data(), sizeof(sReal) * xBench.mSize);
    }, [&](SShared<sTask> lAppPtr) {
        lAppPtr->mVarPtrs[2]->fPullInto(lArrResultCPU.data(), sizeof(sReal) * xBench.mSize);
        return lArrResultCPU.front();
    });
    
    
}

