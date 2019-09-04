// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include "HelperAliasis.hpp"

namespace nAV {

    enum class eArgPurpose {
        kUnknown = 0,
        kInput,
        kOutput,
        kInNOut,
        kBuffer
    };
}

namespace nFlow {
    using namespace nAV;
    
    struct sFuncDectriptor;
    struct sArgDescriptor;
    
    struct sExeSource :
    public std::enable_shared_from_this<sExeSource> {
        sStr mContent;
        sStr mPath;
        SArr<SShared<sFuncDectriptor>> mFuncs;
        
        void fInitText();
        SShared<sFuncDectriptor> fMakeFunc(sStr, SArr<sArgDescriptor> = { });
    };

    struct sFuncDectriptor :
    public std::enable_shared_from_this<sFuncDectriptor> {
        SArr<SShared<sArgDescriptor>> mArgs;
        sStr mName;
        SShared<sExeSource> mSourceFile;
    };
    
    /**
     *  Describes the types and properties of arguments to a specific function.
     */
    struct sArgDescriptor :
    public std::enable_shared_from_this<sArgDescriptor> {
        sStr mName;
        sHash32 mType = sHash32 { 0 };
        eArgPurpose mPurpose = eArgPurpose::kUnknown;
        bSize mFixedSize = 0;
    };
    
    /**
     *  Describes a variable being passed between functions.
     *  Those are the names of `sArgDescriptor` within a specific program,
     *  that allow us to build a computational graph.
     */
    struct sVarSymbol {
        sStr mName;
    };
    
    struct sTaskDependancies {
        enum class eStatus {
            kReady,
            kRunning,
            kCompleted,
            kRestructuingItself
        };
        std::atomic<bBool> mCompleted = false;
        SArr<SShared<sTaskDependancies>> mDependencies;
        SArr<SWeak<sTaskDependancies>> mFollowers;
        
        // If only all the `mDependencies` are `mCompleted`.
        bBool fCanStart() const;
        // If only all the `mFollowers` are `mCompleted`.
        bBool fMayStillNeedBuffers() const;
    };
    
}


//namespace nFlow {
//    
//    struct sDataPass { };
//    struct sFunction {
//        
//    };
//    struct sSymbol {
//        sDataPass m;
//    };
//    /**
//     *  The program as intended by user or third party.
//     *  This will be later transformed into `sModelle` IR,
//     *  which will optimize the needless .
//     */
//    struct sModel {
//        SArr<SShared<sTaska>> mTasks;
//        SArr<SShared<sSymbolum>> mSymbols;
//        
//    };
//    
//    struct sVariable {
//        
//    };
//    /**
//     *  Task = Function + Variables;
//     */
//    struct sTask {
//        
//    };
//    /**
//     *  Application = Model + Data;
//     */
//    struct sApplication {
//        
//    };
//    struct sApp {
//        
//    };
//}
