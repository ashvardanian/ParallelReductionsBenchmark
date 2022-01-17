// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "Tests.hpp"
#include "HelperAliasis.hpp"
#include "HelperMethods.hpp"

#include <Halide.h>

using namespace nHal;

template <typename aFuncRun>
void gForMethod(sBenchDescription xBench, sStr xMethod,
                SArr<sBenchResult> & yResults,
                sStr xDevName,
                aFuncRun xFuncRun) {
    sBenchResult lResultCPU;
    lResultCPU.mTechnology = "Halide";
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

void nHal::gBenchParallel(sBenchDescription xBench,
                          SArr<sReal> const & xArrFullCPU,
                          SArr<sBenchResult> & yResults) {
    
    
    Halide::Buffer<sReal> lArr {
        const_cast<sReal *>(xArrFullCPU.data()),
        static_cast<bInt32>(xArrFullCPU.size()),
        "xArr"
    };
    Halide::Var lIdx { "lIdx" };
    Halide::Func lMapping;
    
    auto lRunBench = [&](sStr lName, auto lConfig) {
        gForMethod(xBench, lName, yResults, "CompileJIT", [&] {
            lConfig();
            lMapping.compile_jit();
        });
        gForMethod(xBench, lName, yResults, "CompileToC", [&] {
            lConfig();
            lMapping.compile_to_c(lName, { });
        });
        gForMethod(xBench, lName, yResults, "CompileStatic", [&] {
            lConfig();
            lMapping.compile_to_static_library(lName, { });
        });
        gForMethod(xBench, lName, yResults, "CompileAssembly", [&] {
            lConfig();
            lMapping.compile_to_assembly(lName, { });
        });
        gForMethod(xBench, lName, yResults, "CompileBitcode", [&] {
            lConfig();
            lMapping.compile_to_bitcode(lName, { });
        });
        gForMethod(xBench, lName, yResults, "Default", [&] {
            lConfig();

            Halide::Buffer<sReal> lResult = lMapping.realize(static_cast<bInt32>(xBench.mSize));
            SArr<sReal> lResultCPU(lResult.begin(), lResult.end());
        });
        gForMethod(xBench, lName, yResults, "Debug", [&] {
            lConfig();
            lMapping.trace_loads();
            lMapping.trace_realizations();
            lMapping.trace_stores();
            
            Halide::Buffer<sReal> lResult = lMapping.realize(static_cast<bInt32>(xBench.mSize));
            SArr<sReal> lResultCPU(lResult.begin(), lResult.end());
        });
    };
    
    lRunBench("gArithmAddConst", [&] {
        Halide::Expr lExpr = lArr(lIdx);
        lMapping(lIdx) = lExpr + sReal(M_PI);
    });
    lRunBench("gArithmMulConst", [&] {
        lMapping(lIdx) = lArr(lIdx) * sReal(M_PI);
    });
    lRunBench("gArithmAddArr", [&] {
        lMapping(lIdx) = lArr(lIdx) + lArr(lIdx);
    });
    lRunBench("gArithmMulArr", [&] {
        lMapping(lIdx) = lArr(lIdx) * lArr(lIdx);
    });
    lRunBench("gTrigSin", [&] {
        lMapping(lIdx) = Halide::sin(lArr(lIdx));
    });
    lRunBench("gTrigCos", [&] {
        lMapping(lIdx) = Halide::cos(lArr(lIdx));
    });
    lRunBench("gPowExp", [&] {
        lMapping(lIdx) = Halide::exp(lArr(lIdx));
    });
    lRunBench("gPowSqrt", [&] {
        lMapping(lIdx) = Halide::sqrt(lArr(lIdx));
    });
    lRunBench("gTrigCosPiManual", [&] {
        lMapping(lIdx) = Halide::cos(lArr(lIdx) * sReal(M_PI));
    });
    lRunBench("gMulAddConstManual", [&] {
        lMapping(lIdx) = lArr(lIdx) * sReal(M_PI) + sReal(M_PI);
    });
}

