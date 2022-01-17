// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#pragma once
#include <MoltenVK/mvk_vulkan.h>
#include "HelperMethods.hpp"
#include "HelperAliasis.hpp"
#include "ComputeGraph.hpp"

namespace nVK {

    struct sQueue;
    struct sTask;
    struct sTaskSequenceCompute;
    struct sDevice;
    
    // Optimal organization schema.
    // 1. init the devices and queues and command-pools
    // 2. allow submitting operations into device-specific or general queue.
    // 3. once a new operation is mapped to specific device:
    //    a. read the source code into `SArr<sExeSource> mSources` IF MISSING.
    //    b. generate `sExe` and `sExe.mVkDescriptorLayout` and `sExe.mShader` for specific `sDevice` IF MISSING.
    //    c. generate a new `sTask` with custom `VkDescriptorPool` for that specific task.
    //    d. for every variable in `sExe`, allocate a new buffer in `sTask`
    //    e. create semaphores for every waiting process and every following dependency.
    //       they will be use during the construction of the piepeline.
    //       http://ogldev.atspace.co.uk/www/tutorial53/tutorial53.html
    //    f. create 1 operation `VkPipeline` in `sTask`.
    // 4. group all operations for a specific GPU together and submit them into the `sDevice.mQueues.fAny()`.
    //
    // Each file ends with ".<stage>", where <stage> is one of:
    //  .conf   to provide an optional config file that replaces the default configuration
    //  .vert   for a vertex shader
    //  .tesc   for a tessellation control shader
    //  .tese   for a tessellation evaluation shader
    //  .geom   for a geometry shader
    //  .frag   for a fragment shader
    //  .comp   for a compute shader
    struct sFuncCompiled :
    public std::enable_shared_from_this<sFuncCompiled> {
        SShared<sDevice> mDevicePtr;
        SWeak<nFlow::sFuncDectriptor> mDescriptor;
        VkDescriptorSetLayout mVkDescriptorLayout;
        VkShaderModule mShader;
        
        ~sFuncCompiled();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitShader();
        void fInitDescriptorLayout();
        
        SShared<sTask> fNewTask();
    };
    
    struct sVarWBuffer :
    public std::enable_shared_from_this<sVarWBuffer> {
        /**
         *  Buffer is used as a storage buffer.
         *  Its exclusive to a single queue family at a time.
         *  But the buffer doesn't allocate memory for itself,
         *  so we must do that manually.
         */
        VkBuffer mVkBuffer;
        VkDeviceMemory mVkMemory;
        bSize mSize = 0;
        bPtrUntyped mSourceOnHost = nullptr;
        SShared<sTask> mTaskPtr;
        
        ~sVarWBuffer();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitBuffer();
        
        bBool fResetConstant(bPtrUntypedC, bSize);
        bBool fResetVariable(bPtrUntyped, bSize);
        bBool fResetIndependantBuffer(bSize);
        bBool fPullInto(bPtrUntyped, bSize);
        
    private:
        bBool fExchangeWith(bPtrUntyped, bSize, nAV::nDisk::eMemoryAccess);
    };
    
    /**
     *  App is a combination of:
     *  -   executable binary.
     *  -   arguments for it.
     *  -   allocated buffers for its output.
     */
    struct sTask :
    public std::enable_shared_from_this<sTask> {
        /**
         *  A single descriptor represents a single resource,
         *  and several descriptors are organized into descriptor sets,
         *  which are basically just collections of descriptors.
         */
        VkDescriptorSet mVkDescriptorSet;
        /**
         *  Descriptors represent resources in shaders.
         *  They allow us to use things like uniform buffers,
         *  storage buffers and images in GLSL.
         *
         *  We can have 1 instance of `mPoolDescriptors` for multiple `sTask`s,
         *  but it becomes much more complicated.
         */
        VkDescriptorPool mVkDescriptorPool;

        SShared<sFuncCompiled> mFuncPtr;
        SArr<SShared<sVarWBuffer>> mVarPtrs;
        SShared<sTaskSequenceCompute> mGroupPtr;
        SArr<bSize> mWorkDimensions;

        ~sTask();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitDescriptorPool();
        void fInitDescriptorSet();

        SShared<sVarWBuffer> fReallocVar(bSize lIdx);
    };

    struct sQueueFeatures {
        bBool mSupportsGraphics = false;
        bBool mSupportsCompute = false;
    };
    struct sQueueFamily {
        bUInt32 mFamilyID = 0;
        sQueueFeatures mFeatures;
    };
    struct sDeviceFeatures {
        bSize mMemoryTotal = 0;
        bBool mSupportsFlt16 = false;
        SArr<sQueueFamily> mQueueFamilies;
        sStr mName;
    };
    struct sQueue {
        ~sQueue();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitCommandPool();
        void fRunNow(SShared<sTaskSequenceCompute>);
        
        SArr<SShared<sTaskSequenceCompute>> mTasks;
        SShared<sDevice> mDevicePtr;
        VkQueue mVk;
        /**
         *  Groups of queues that have the same capabilities are called families.
         *  When submitting a command buffer, you must specify to which queue in
         *  the family you are submitting to.
         */
        sQueueFamily mFamily;
        /**
         *  The `VkCommandBuffer` is used to record commands,
         *  that will be submitted to a queue. It is specific to "queue-family"!
         *  To allocate such command buffers, we use a command pool.
         */
        VkCommandPool mVkPoolCommands;
    };
    
    struct sDevice :
    public std::enable_shared_from_this<sDevice> {
        /**
         *  Priorities are just a hint to the implementation.
         *  They don't give you any guarantees.
         */
        static constexpr bFlt32 kQueuePriorities[] = { 1.0, 0.8, 0.6 };
        /**
         *  Generally there is a 1-to-1 mapping between logical and physical devices,
         *  but multiple devices can be grouped together, if they support identical
         *  extensions, features, and properties.
         *  It allows them to use the shared address space.
         */
        VkDevice mLogical;
        /**
         *  The physical device is some device (CPU/GPU/FPGA) on
         *  the system that supports usage of Vulkan.
         *  Often, it is simply a graphics card that supports Vulkan.
         */
        SArr<VkPhysicalDevice> mPhysicals;
        /**
         *  Each queue may process work asynchronously to one another.
         *  The commands are stored in a `VkCommandBuffer`, which is given to the queue.
         */
        SArr<SShared<sQueue>> mQueuesCompute;
        /**
         *  None of the vendors have more than 1 hardware graphics queue AFAIK,
         *  only multiple compute queues. AMD has this, while NVidia emulates it in the driver.
         */
        SArr<SShared<sQueue>> mQueuesGraphics;
        
        sDeviceFeatures mFeatures;
        
        ~sDevice();
        void fDealloc();
        void fInitNewQueues(SArr<SPair<sQueueFamily, bSize>> lFamsAndCounts);
        void fInitNewQueues(bSize lCntCompute, bSize lCntGraphics);
        bUInt32 fFindMemoryGroupOfType(bUInt32 lMemoryTypeBits,
                                       VkMemoryPropertyFlags);
        /**
         *  Vulkan doesn’t support any high-level shading language like GLSL or HLSL.
         *  Instead, Vulkan accepts an intermediate format called SPIR-V which any
         *  higher-level language can emit. A buffer filled with data in SPIR-V is
         *  used to create a `VkShaderModule`.
         *  That object represents a piece of shader code, possibly in some partially
         *  compiled form, but it’s not anything the GPU can execute yet. Only when
         *  creating the `VkPipeline` for each shader stage you are going to use (vertex,
         *  tessellation control, tessellation evaluation, geometry, fragment, or compute)
         *  do you specify the ShaderModule plus the name of the entry point function (like “main”).
         */
        SShared<sFuncCompiled> fCompile(SShared<nFlow::sFuncDectriptor>);

        void fRunOnAnyQueue(SShared<sTask> const &);
    };
    
    /**
     *  Vulkan exposes one or more devices,
     *  each of which exposes one or more async queues.
     *  The set of queues supported by a device is partitioned into families.
     *  Each family supports one or more types of functionality and may
     *  contain multiple queues with similar characteristics.
     */
    struct sVulkan {
        
        static sVulkan & gShared() {
            static sVulkan lVk;
            return lVk;
        }
        
        /**
         *  By enabling validation layers, Vulkan will emit warnings if the API
         *  is used incorrectly. We shall enable the layer "VK_LAYER_LUNARG_standard_validation",
         *  which is basically a collection of several useful validation layers.
         *  On Apple devices the only exposed layer is "MoltenVK"!
         */
        constexpr static bool kEnableValidationLayers = false;
        
        /**
         *  Instance is the first object you create. It represents the connection
         *  from your application to the Vulkan runtime and therefore only should
         *  exist once in your application. It also stores all application specific
         *  state required to use Vulkan. Therefore you must specify all layers (like
         *  the Validation Layer) and all extensions you want to enable when creating it.
         */
        VkInstance mVkInstance;
        VkDebugReportCallbackEXT mDebugReportCallback;
        
        SArr<SShared<sDevice>> mDevices;
        SArr<sCStrC> mEnabledLayers;
        SArr<sCStrC> mEnabledExtensions;
        
        /**
         *  Don't create more queues than necessary.
         *  NVidia for example always exposes 16 queues in the presentation/graphics family,
         *  which are multiplexed by the driver into a single hardware queue.
         *  As such, it can cause overhead in the driver if you choose to reserve
         *  more queues than you use.
         *  https://www.reddit.com/r/vulkan/comments/7ynlcl/cost_of_including_queues_in_logical_device/
         */
        bUInt32 mQueuesComputePerDevice = 1;
        bUInt32 mQueuesGraphicsPerDevice = 0;
        
        static void gCheckCode(VkResult);
        
        sVulkan();
        ~sVulkan();
        void fDealloc();
        void fRun(sTask);
        
    private:
        void fInitValidation();
        void fInitInstance();
        void fInitWarningsCallback();
        void fInitDevices();
    public:
        
        SArr<SShared<sDevice>> fAddAllDevices();
        static void gExportDevice(VkPhysicalDevice, sDeviceFeatures &);
        static void gExportComputeQueueFamilies(VkPhysicalDevice, SArr<sQueueFamily> &);

    };

}
