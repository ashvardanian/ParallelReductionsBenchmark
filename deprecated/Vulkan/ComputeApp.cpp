// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "ComputeApp.hpp"
#include "HelperMethods.hpp"

using namespace nAV;
using namespace nVK;

namespace nVK {
    /**
     *  `sTaskSequenceCompute` is the simpler one, because all it supports
     *  is compute-only programs (sometimes called compute shaders).
     *  Computational (non-graphics) pipelines must have just 1 `sTask` in them!
     */
    struct sTaskSequenceCompute :
    public std::enable_shared_from_this<sTaskSequenceCompute> {
        VkPipeline mVkPipeline;
        VkPipelineLayout mVkPipelineLayout;
        SShared<sTask> mTaskPtr;
        VkCommandBuffer mVkCommandBuffer;
        SArr<bSize> mWorkDimensions;
        
        ~sTaskSequenceCompute();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitPipeline();
        void fInitCommandBuffer(sQueue &);
    };
    
    /**
     *  `sTaskSequenceGraphics` is much more complex than `sTaskSequenceCompute`,
     *  because it encompasses all the parameters like vertex, fragment, geometry,
     *  compute and tessellation where applicable, plus things like vertex attributes,
     *  primitive topology, backface culling, and blending mode, to name just a few.
     *
     *  All those parameters that used to be separate settings in much older
     *  graphics APIs (DirectX 9, OpenGL), were later grouped into a smaller
     *  number of state objects as the APIs progressed (DirectX 10 and 11) and
     *  must now be baked into a single big, immutable object with today’s
     *  modern APIs like Vulkan.
     *  For each different set of parameters needed during rendering you must
     *  create a new `VkPipeline`. You can then set it as the current active `VkPipeline`
     *  in a `VkCommandBuffer` by calling the function `vkCmdBindPipeline`.
     */
    struct sTaskSequenceGraphics :
    public std::enable_shared_from_this<sTaskSequenceGraphics> {
        VkPipeline mVkPipeline;
        VkPipelineLayout mVkPipelineLayout;
        SArr<SShared<sTask>> mAppPtrs;
        SArr<VkSemaphore> mVkSemaphores;
        
        ~sTaskSequenceGraphics();
        void fDealloc();
        VkDevice fDeviceID() const;
        void fInitPipeline();
    };
}

void sDevice::fInitNewQueues(SArr<SPair<sQueueFamily, bSize>> lFamsWanted) {
    sDevice & lDevice = *this;
    sVulkan & lAPI = sVulkan::gShared();
    
    SArr<VkDeviceQueueCreateInfo> lQueueCreateInfos;
    for (auto & [lFam, lCnt] : lFamsWanted) {
        if (!lCnt)
            continue;
        VkDeviceQueueCreateInfo lQueueCreateInfo = { };
        lQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        lQueueCreateInfo.queueFamilyIndex = lFam.mFamilyID;
        lQueueCreateInfo.queueCount = static_cast<bUInt32>(lCnt);
        lQueueCreateInfo.pQueuePriorities = kQueuePriorities;
        lQueueCreateInfos.push_back(lQueueCreateInfo);
    }
    if (lQueueCreateInfos.empty())
        return;
    
    VkDeviceCreateInfo lDeviceCreateInfo = { };
    VkPhysicalDeviceFeatures lDeviceFeatures = { };
    lDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    lDeviceCreateInfo.enabledExtensionCount = 0;
    lDeviceCreateInfo.enabledLayerCount = static_cast<bUInt32>(lAPI.mEnabledLayers.size());
    lDeviceCreateInfo.ppEnabledLayerNames = lAPI.mEnabledLayers.data();
    lDeviceCreateInfo.pQueueCreateInfos = lQueueCreateInfos.data();
    lDeviceCreateInfo.queueCreateInfoCount = static_cast<bUInt32>(lQueueCreateInfos.size());
    lDeviceCreateInfo.pEnabledFeatures = &lDeviceFeatures;
    sVulkan::gCheckCode(vkCreateDevice(lDevice.mPhysicals.front(),
                                       &lDeviceCreateInfo, NULL,
                                       &lDevice.mLogical));
    
    for (auto const & [lFam, lCnt] : lFamsWanted) {
        for (bUInt32 lQIdx = 0; lQIdx < lCnt; lQIdx ++) {
            VkQueue lQNew = { };
            vkGetDeviceQueue(lDevice.mLogical, lFam.mFamilyID,
                             lQIdx, &lQNew);

            auto lQ = std::make_shared<sQueue>();
            lQ->mVk = lQNew;
            lQ->mFamily = lFam;
            lQ->mDevicePtr = shared_from_this();
            if (lFam.mFeatures.mSupportsCompute)
                mQueuesCompute.push_back(lQ);
            if (lFam.mFeatures.mSupportsGraphics)
                mQueuesGraphics.push_back(lQ);
            lQ->fInitCommandPool();
        }
    }
}

void sDevice::fInitNewQueues(bSize lCntCompute, bSize lCntGraphics) {
    sDevice & lDevice = *this;
    auto const & lFams = lDevice.mFeatures.mQueueFamilies;
    for (auto & lFam : lFams) {
        if (lFam.mFeatures.mSupportsCompute &&
            lFam.mFeatures.mSupportsGraphics) {
            fInitNewQueues({ { lFam, lCntCompute + lCntGraphics } });
            lCntCompute = 0;
            lCntGraphics = 0;
            return;
        } else if (lFam.mFeatures.mSupportsCompute) {
            fInitNewQueues({ { lFam, lCntCompute } });
            lCntCompute = 0;
        } else if (lFam.mFeatures.mSupportsGraphics) {
            fInitNewQueues({ { lFam, lCntGraphics } });
            lCntGraphics = 0;
        }
    }
}

bUInt32 sDevice::fFindMemoryGroupOfType(bUInt32 lMemoryTypeBits,
                                        VkMemoryPropertyFlags lProperties) {
    sDevice & lDevice = *this;
    VkPhysicalDeviceMemoryProperties lMemoryProperties = { };
    vkGetPhysicalDeviceMemoryProperties(lDevice.mPhysicals.front(), &lMemoryProperties);
    
    for (bUInt32 i = 0; i < lMemoryProperties.memoryTypeCount; i ++) {
        if ((lMemoryTypeBits & (1 << i)) &&
            ((lMemoryProperties.memoryTypes[i].propertyFlags & lProperties) == lProperties))
            return i;
    }
    return std::numeric_limits<bUInt32>::max();
}

SShared<sFuncCompiled> sDevice::fCompile(SShared<nFlow::sFuncDectriptor> lSource) {
    auto lResult = std::make_shared<sFuncCompiled>();
    lResult->mDescriptor = lSource;
    lResult->mDevicePtr = shared_from_this();
    lResult->fInitShader();
    lResult->fInitDescriptorLayout();
    return lResult;
}

void sDevice::fRunOnAnyQueue(SShared<sTask> const & xTaskPtr) {
    auto lPipePtr = std::make_shared<sTaskSequenceCompute>();
    auto lQueuePtr = mQueuesCompute.front();
    {
        xTaskPtr->fInitDescriptorSet();
        lPipePtr->mTaskPtr = xTaskPtr;
    }
    lPipePtr->mWorkDimensions = xTaskPtr->mWorkDimensions;
    lPipePtr->fInitPipeline();
    lPipePtr->fInitCommandBuffer(*lQueuePtr);
    lQueuePtr->fRunNow(lPipePtr);
}

void sVarWBuffer::fInitBuffer() {
    sVarWBuffer & lVar = *this;
    auto lDevicePtr = mTaskPtr->mFuncPtr->mDevicePtr;
    
    if (lVar.mVkBuffer)
        return;
    
    VkBufferCreateInfo lBufferCreateInfo = { };
    lBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    lBufferCreateInfo.size = lVar.mSize;
    lBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    lBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sVulkan::gCheckCode(vkCreateBuffer(fDeviceID(), &lBufferCreateInfo, NULL, &lVar.mVkBuffer));
    
    VkMemoryRequirements lMemoryRequirements = { };
    vkGetBufferMemoryRequirements(fDeviceID(), lVar.mVkBuffer, &lMemoryRequirements);
    
    /// We want to be able to read the buffer memory from the GPU to the CPU
    /// with `vkMapMemory`, so we set `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.
    constexpr auto kCoherent = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    /// With it memory written by the device(GPU) will be easily
    /// visible to the host(CPU), without having to call any extra
    /// flushing commands. So mainly for convenience, we set this flag.
    constexpr auto kVisible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    
    VkMemoryAllocateInfo lAllocateInfo = { };
    lAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    lAllocateInfo.allocationSize = lMemoryRequirements.size;
    lAllocateInfo.memoryTypeIndex = lDevicePtr->fFindMemoryGroupOfType(lMemoryRequirements.memoryTypeBits,
                                                                       kCoherent | kVisible);
    
    sVulkan::gCheckCode(vkAllocateMemory(fDeviceID(), &lAllocateInfo, NULL, &lVar.mVkMemory));
    sVulkan::gCheckCode(vkBindBufferMemory(fDeviceID(), lVar.mVkBuffer, lVar.mVkMemory, 0));
}

bBool sVarWBuffer::fExchangeWith(bPtrUntyped xStart, bSize xLen,
                                 nAV::nDisk::eMemoryAccess xDir) {
    fInitBuffer();
    bPtrUntyped lMappedMemoryPtr = NULL;
    // Map the buffer memory, so that we can read from it on the CPU.
    sVulkan::gCheckCode(vkMapMemory(fDeviceID(), mVkMemory, 0,
                                    xLen, 0, &lMappedMemoryPtr));
    if (xDir == nDisk::eMemoryAccess::kRead) {
        std::memcpy(xStart, lMappedMemoryPtr, xLen);
    } else {
        std::memcpy(lMappedMemoryPtr, xStart, xLen);
    }
    // Done reading, so unmap to free-up virtual address space.
    vkUnmapMemory(fDeviceID(), mVkMemory);
    return true;
}

bBool sVarWBuffer::fPullInto(bPtrUntyped xData, bSize xLen) {
    return fExchangeWith(xData, xLen, nAV::nDisk::eMemoryAccess::kRead);
}

bBool sVarWBuffer::fResetConstant(bPtrUntypedC xData, bSize xLen) {
    mSize = xLen;
    // The `mVkCommandBuffer` is part of `sTaskSequenceCompute`
    //    mSourceOnHost = (bPtrUntyped)xData;
    //    return true;
    return fExchangeWith((bPtrUntyped)xData, xLen, nAV::nDisk::eMemoryAccess::kWrite);
}

bBool sVarWBuffer::fResetVariable(bPtrUntyped xData, bSize xLen) {
    mSize = xLen;
    //    mSourceOnHost = xData;
    //    return true;
    return fExchangeWith(xData, xLen, nAV::nDisk::eMemoryAccess::kWrite);
}

bBool sVarWBuffer::fResetIndependantBuffer(bSize xLen) {
    mSize = xLen;
    mSourceOnHost = nullptr;
    return true;
}

void sFuncCompiled::fInitDescriptorLayout() {
    auto lDescriptorPtr = mDescriptor.lock();
    
    SArr<VkDescriptorSetLayoutBinding> lBindings;
    for (bUInt32 lIdx = 0; lIdx < lDescriptorPtr->mArgs.size(); lIdx ++) {
        VkDescriptorSetLayoutBinding lDescriptorSetLayoutBinding = { };
        lDescriptorSetLayoutBinding.binding = lIdx;
        lDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lDescriptorSetLayoutBinding.descriptorCount = 1;
        lDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        lBindings.push_back(lDescriptorSetLayoutBinding);
    }
    
    VkDescriptorSetLayoutCreateInfo lDescriptorSetLayoutCreateInfo = { };
    lDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lDescriptorSetLayoutCreateInfo.bindingCount = static_cast<bUInt32>(lBindings.size());
    lDescriptorSetLayoutCreateInfo.pBindings = lBindings.data();
    
    sVulkan::gCheckCode(vkCreateDescriptorSetLayout(fDeviceID(),
                                                    &lDescriptorSetLayoutCreateInfo,
                                                    NULL, &mVkDescriptorLayout));
}

SShared<sTask> sFuncCompiled::fNewTask() {
    auto lDescriptorPtr = mDescriptor.lock();
    auto lResult = std::make_shared<sTask>();
    lResult->mFuncPtr = shared_from_this();
    lResult->mVarPtrs.resize(lDescriptorPtr->mArgs.size());
    lResult->fInitDescriptorPool();
    return lResult;
}

SShared<sVarWBuffer> sTask::fReallocVar(bSize lIdx) {
    auto & lTask = *this;
    auto lDescriptorPtr = lTask.mFuncPtr->mDescriptor.lock();
    auto lPtr = std::make_shared<sVarWBuffer>();
    lPtr->mTaskPtr = lTask.shared_from_this();
    lPtr->mSize = lDescriptorPtr->mArgs[lIdx]->mFixedSize;
    lTask.mVarPtrs[lIdx] = lPtr;
    return lPtr;
}

void sTask::fInitDescriptorPool() {
    sTask & lTask = *this;
    auto lExeFilePtr = mFuncPtr->mDescriptor.lock();
    
    VkDescriptorPoolSize lDescriptorPoolSize = { };
    lDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lDescriptorPoolSize.descriptorCount = static_cast<bUInt32>(lExeFilePtr->mArgs.size());
    
    VkDescriptorPoolCreateInfo lDescriptorPoolCreateInfo = { };
    lDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    lDescriptorPoolCreateInfo.maxSets = 1;
    lDescriptorPoolCreateInfo.poolSizeCount = 1;
    lDescriptorPoolCreateInfo.pPoolSizes = &lDescriptorPoolSize;
    
    sVulkan::gCheckCode(vkCreateDescriptorPool(fDeviceID(),
                                               &lDescriptorPoolCreateInfo, NULL,
                                               &lTask.mVkDescriptorPool));
}

void sTask::fInitDescriptorSet() {
    sTask & lTask = *this;
    auto lExeFilePtr = mFuncPtr->mDescriptor.lock();
    auto & lVarsExpected = lExeFilePtr->mArgs;
    
    VkDescriptorSetAllocateInfo lDescriptorSetAllocateInfo = { };
    lDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    lDescriptorSetAllocateInfo.descriptorPool = lTask.mVkDescriptorPool;
    lDescriptorSetAllocateInfo.descriptorSetCount = 1;
    lDescriptorSetAllocateInfo.pSetLayouts = &lTask.mFuncPtr->mVkDescriptorLayout;
    
    sVulkan::gCheckCode(vkAllocateDescriptorSets(fDeviceID(),
                                                 &lDescriptorSetAllocateInfo,
                                                 &lTask.mVkDescriptorSet));
    
    SArr<VkDescriptorBufferInfo> lDescriptorBufferInfos(lVarsExpected.size());
    for (bSize i = 0; i < lVarsExpected.size(); i ++) {
        SShared<sVarWBuffer> & lVarPtr = mVarPtrs[i];
        lVarPtr->fInitBuffer();
        VkDescriptorBufferInfo lDescriptorBufferInfo = { };
        lDescriptorBufferInfo.buffer = lVarPtr->mVkBuffer;
        lDescriptorBufferInfo.offset = 0;
        lDescriptorBufferInfo.range = lVarPtr->mSize;
        lDescriptorBufferInfos[i] = lDescriptorBufferInfo;
    }
    
    VkWriteDescriptorSet lWriteDescriptorSet = { };
    lWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    lWriteDescriptorSet.dstSet = lTask.mVkDescriptorSet;
    lWriteDescriptorSet.dstBinding = 0;
    lWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lWriteDescriptorSet.descriptorCount = static_cast<bUInt32>(lDescriptorBufferInfos.size());
    lWriteDescriptorSet.pBufferInfo = lDescriptorBufferInfos.data();
    
    vkUpdateDescriptorSets(fDeviceID(),
                           1, &lWriteDescriptorSet, 0, NULL);
}

void sFuncCompiled::fInitShader() {
    auto lDescriptorPtr = mDescriptor.lock();
    auto lDevicePtr = mDevicePtr;
    
    VkShaderModuleCreateInfo lCreateInfo = { };
    lCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    lCreateInfo.pCode = (bUInt32 *)lDescriptorPtr->mSourceFile->mContent.c_str();
    lCreateInfo.codeSize = lDescriptorPtr->mSourceFile->mContent.size();
    
    sVulkan::gCheckCode(vkCreateShaderModule(fDeviceID(),
                                             &lCreateInfo, NULL, &mShader));
}

void sTaskSequenceCompute::fInitPipeline() {
    VkPipelineLayoutCreateInfo lPipelineLayoutCreateInfo = { };
    lPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lPipelineLayoutCreateInfo.setLayoutCount = 1;
    lPipelineLayoutCreateInfo.pSetLayouts = &mTaskPtr->mFuncPtr->mVkDescriptorLayout;
    sVulkan::gCheckCode(vkCreatePipelineLayout(fDeviceID(),
                                               &lPipelineLayoutCreateInfo,
                                               NULL, &mVkPipelineLayout));
    
    VkPipelineShaderStageCreateInfo lShaderStageCreateInfo = { };
    lShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    lShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    lShaderStageCreateInfo.module = mTaskPtr->mFuncPtr->mShader;
    lShaderStageCreateInfo.pName = "main";
    
    VkComputePipelineCreateInfo lPipelineCreateInfo = { };
    lPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    lPipelineCreateInfo.stage = lShaderStageCreateInfo;
    lPipelineCreateInfo.layout = mVkPipelineLayout;
    
    sVulkan::gCheckCode(vkCreateComputePipelines(fDeviceID(),
                                                 VK_NULL_HANDLE,
                                                 1, &lPipelineCreateInfo,
                                                 NULL, &mVkPipeline));
}

void sQueue::fInitCommandPool() {
    VkCommandPoolCreateInfo lCommandPoolCreateInfo = { };
    lCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    lCommandPoolCreateInfo.flags = 0;
    // All command buffers allocated from this command pool,
    // must be submitted to queues of this family ONLY.
    lCommandPoolCreateInfo.queueFamilyIndex = mFamily.mFamilyID;
    sVulkan::gCheckCode(vkCreateCommandPool(fDeviceID(),
                                            &lCommandPoolCreateInfo,
                                            NULL, &mVkPoolCommands));
}

void sTaskSequenceCompute::fInitCommandBuffer(sQueue & lQ) {
    
    VkCommandBufferAllocateInfo lCommandBufferAllocateInfo = { };
    lCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    lCommandBufferAllocateInfo.commandPool = lQ.mVkPoolCommands;
    lCommandBufferAllocateInfo.commandBufferCount = 1;
    // If the command buffer is primary, it can be directly submitted to queues.
    // A secondary buffer has to be called from some primary command buffer,
    // and cannot be directly submitted to a queue.
    // To keep things simple, we use a primary command buffer.
    lCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    sVulkan::gCheckCode(vkAllocateCommandBuffers(fDeviceID(), &lCommandBufferAllocateInfo, &mVkCommandBuffer));
    
    VkCommandBufferBeginInfo lBeginInfo = { };
    lBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // This buffer is only submitted and used once in this application.
    lBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    sVulkan::gCheckCode(vkBeginCommandBuffer(mVkCommandBuffer, &lBeginInfo));
    
    
    // Wait for some events:
    // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/vkCmdWaitEvents.html
    //    for (auto lVarPtr : mTaskPtr->mVarPtrs) {
    //        if (!lVarPtr->mSourceOnHost)
    //            continue;
    //        // Not recommended for large data pieces:
    //        vkCmdUpdateBuffer(mVkCommandBuffer, lVarPtr->mVkBuffer, 0,
    //                          lVarPtr->mSize, lVarPtr->mSourceOnHost);
    //        // Copy from another device:
    //        // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/vkCmdCopyBuffer.html
    //        // For small fixed-len variables use:
    //        // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/vkCmdPushConstants.html
    //    }

    // We need to bind a pipeline, AND a descriptor set before we dispatch.
    // The validation layer will NOT give warnings if you forget these,
    // so be very careful not to forget them.
    vkCmdBindPipeline(mVkCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mVkPipeline);
    vkCmdBindDescriptorSets(mVkCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            mVkPipelineLayout, 0, 1, &mTaskPtr->mVkDescriptorSet, 0, NULL);
    
    // Calling `vkCmdDispatch` basically starts the compute pipeline,
    // and executes the compute shader.
    // The number of workgroups is specified in the arguments.
    vkCmdDispatch(mVkCommandBuffer,
                  static_cast<bUInt32>(mWorkDimensions.size() > 0 ? mWorkDimensions[0] : 0),
                  static_cast<bUInt32>(mWorkDimensions.size() > 1 ? mWorkDimensions[1] : 1),
                  static_cast<bUInt32>(mWorkDimensions.size() > 2 ? mWorkDimensions[2] : 1));
    sVulkan::gCheckCode(vkEndCommandBuffer(mVkCommandBuffer));
}

void sQueue::fRunNow(SShared<sTaskSequenceCompute> lTaskPtr) {
    
    VkSubmitInfo lSubmitInfo = { };
    lSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    lSubmitInfo.commandBufferCount = 1;
    lSubmitInfo.pCommandBuffers = &lTaskPtr->mVkCommandBuffer;
    
    VkFence lFence = { };
    VkFenceCreateInfo lFenceCreateInfo = { };
    lFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    lFenceCreateInfo.flags = 0;
    sVulkan::gCheckCode(vkCreateFence(fDeviceID(), &lFenceCreateInfo, NULL, &lFence));
    
    // Submit the command buffer on the queue, at the same time giving a fence.
    sVulkan::gCheckCode(vkQueueSubmit(mVk, 1, &lSubmitInfo, lFence));
    // The command will not have finished executing until the fence is signalled.
    // So we wait here. We will directly after this read our buffer from the GPU,
    // and we will not be sure that the command has finished executing unless we wait for the fence.
    sVulkan::gCheckCode(vkWaitForFences(fDeviceID(), 1, &lFence, VK_TRUE, 100000000000));
    vkDestroyFence(fDeviceID(), lFence, NULL);
}

#pragma mark - Preparation

VKAPI_ATTR VKAPI_CALL
VkBool32 gDebugReportCallback(VkDebugReportFlagsEXT,
                              VkDebugReportObjectTypeEXT,
                              bUInt64 lObject,
                              bSize lLocation,
                              bInt32 lCode,
                              sCStrC lLayerPrefix,
                              sCStrC lMessage,
                              bPtrUntyped lUserData) {
    
    printf("Debug Report: %s: %s\n", lLayerPrefix, lMessage);
    return VK_FALSE;
}

void sVulkan::gCheckCode(VkResult lResult) {
    if (lResult != VK_SUCCESS)
        gThrow("");
}

void sVulkan::fInitValidation() {
    if (!kEnableValidationLayers)
        return;
    
    bUInt32 lLayerCount;
    vkEnumerateInstanceLayerProperties(&lLayerCount, NULL);
    SArr<VkLayerProperties> lLayerProps(lLayerCount);
    vkEnumerateInstanceLayerProperties(&lLayerCount, lLayerProps.data());
    
    bBool foundLayer = false;
    for (VkLayerProperties lProp : lLayerProps) {
        if (strcmp("VK_LAYER_LUNARG_standard_validation", lProp.layerName) == 0) {
            foundLayer = true;
            break;
        }
    }
    if (!foundLayer)
        gThrow("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
    mEnabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
    
    bUInt32 lExtCount;
    vkEnumerateInstanceExtensionProperties(NULL, &lExtCount, NULL);
    SArr<VkExtensionProperties> lExtProperties(lExtCount);
    vkEnumerateInstanceExtensionProperties(NULL, &lExtCount, lExtProperties.data());
    
    bBool foundExtension = false;
    for (VkExtensionProperties lProp : lExtProperties) {
        if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, lProp.extensionName) == 0) {
            foundExtension = true;
            break;
        }
    }
    
    if (!foundExtension)
        gThrow("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
    mEnabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
}

void sVulkan::fInitInstance() {
    VkApplicationInfo lAppInfo = { };
    lAppInfo.apiVersion = VK_API_VERSION_1_0;
    lAppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    lAppInfo.pApplicationName = "UNUM.Net";
    lAppInfo.applicationVersion = 0;
    lAppInfo.pEngineName = "UNUM.APUs";
    lAppInfo.engineVersion = 0;
    
    VkInstanceCreateInfo lCreateInfo = { };
    lCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    lCreateInfo.flags = 0;
    lCreateInfo.pApplicationInfo = &lAppInfo;
    lCreateInfo.enabledLayerCount = static_cast<bUInt32>(mEnabledLayers.size());
    lCreateInfo.ppEnabledLayerNames = mEnabledLayers.data();
    lCreateInfo.enabledExtensionCount = static_cast<bUInt32>(mEnabledExtensions.size());
    lCreateInfo.ppEnabledExtensionNames = mEnabledExtensions.data();
    
    sVulkan::gCheckCode(vkCreateInstance(&lCreateInfo, NULL,
                                         &mVkInstance));
}

void sVulkan::fInitWarningsCallback() {
    if (!kEnableValidationLayers)
        return;
    
    VkDebugReportCallbackCreateInfoEXT lCreateInfo = { };
    lCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    lCreateInfo.pfnCallback = gDebugReportCallback;
    lCreateInfo.flags = (VK_DEBUG_REPORT_ERROR_BIT_EXT |
                         VK_DEBUG_REPORT_WARNING_BIT_EXT |
                         VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT);
    
    using sExt = PFN_vkCreateDebugReportCallbackEXT;
    auto lExt = (sExt)vkGetInstanceProcAddr(mVkInstance, "vkCreateDebugReportCallbackEXT");
    if (!lExt)
        gThrow("Could not load vkCreateDebugReportCallbackEXT");
    sVulkan::gCheckCode(vkCreateDebugReportCallbackEXT(mVkInstance,
                                                       &lCreateInfo, NULL,
                                                       &mDebugReportCallback));
}

void sVulkan::fInitDevices() {
    bUInt32 lDeviceCount;
    vkEnumeratePhysicalDevices(mVkInstance, &lDeviceCount, NULL);
    if (!lDeviceCount)
        gThrow("could not find a device with vulkan support");
    SArr<VkPhysicalDevice> lDevices(lDeviceCount);
    vkEnumeratePhysicalDevices(mVkInstance, &lDeviceCount, lDevices.data());
    
    // TODO: Group devices by their features into a single logical device.
    
    mDevices.clear();
    mDevices.resize(lDeviceCount);
    for (bSize i = 0; i < lDeviceCount; i ++) {
        auto & lDeviceFull = mDevices[i];
        lDeviceFull = std::make_shared<sDevice>();
        lDeviceFull->mPhysicals = { lDevices[i] };
        gExportDevice(lDevices[i], lDeviceFull->mFeatures);
        
        auto & lQs = lDeviceFull->mFeatures.mQueueFamilies;
        gExportComputeQueueFamilies(lDevices[i], lQs);
        if (lQs.empty())
            gThrow("Couldn't find a queue family that supports operations!");
        
        lDeviceFull->fInitNewQueues(mQueuesComputePerDevice, mQueuesGraphicsPerDevice);
    }
}


void sVulkan::gExportDevice(VkPhysicalDevice lDevice,
                            sDeviceFeatures & lStats) {
    VkPhysicalDeviceFeatures lDeviceFeatures = { };
    VkPhysicalDeviceProperties lDeviceProperties = { };
    VkPhysicalDeviceMemoryProperties lMemoryProperties = { };
    vkGetPhysicalDeviceFeatures(lDevice, &lDeviceFeatures);
    vkGetPhysicalDeviceProperties(lDevice, &lDeviceProperties);
    vkGetPhysicalDeviceMemoryProperties(lDevice, &lMemoryProperties);
    
    for (bSize i = 0; i < lMemoryProperties.memoryHeapCount; i ++) {
        lStats.mMemoryTotal += lMemoryProperties.memoryHeaps[i].size;
    }
    lStats.mName = lDeviceProperties.deviceName;
}

void sVulkan::gExportComputeQueueFamilies(VkPhysicalDevice lDevice,
                                          SArr<sQueueFamily> & lQueues) {
    bUInt32 lQueueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(lDevice, &lQueueFamilyCount, NULL);
    SArr<VkQueueFamilyProperties> lQueueFamilies(lQueueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(lDevice, &lQueueFamilyCount, lQueueFamilies.data());
    
    for (bUInt32 i = 0; i < lQueueFamilies.size(); i ++) {
        VkQueueFamilyProperties props = lQueueFamilies[i];
        if (!props.queueCount)
            continue;
        sQueueFamily lQueue;
        lQueue.mFamilyID = i;
        lQueue.mFeatures.mSupportsGraphics = (props.queueFlags & VK_QUEUE_GRAPHICS_BIT);
        lQueue.mFeatures.mSupportsCompute = (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
        lQueues.push_back(lQueue);
    }
}

#pragma mark - Cleaning memory

void sDevice::fDealloc() {
    vkDestroyDevice(mLogical, NULL);
}
void sVulkan::fDealloc() {
    if (kEnableValidationLayers) {
        using sExt = PFN_vkDestroyDebugReportCallbackEXT;
        auto lFunc = (sExt)vkGetInstanceProcAddr(mVkInstance, "vkDestroyDebugReportCallbackEXT");
        if (!lFunc)
            gThrow("Could not load vkDestroyDebugReportCallbackEXT");
        lFunc(mVkInstance, mDebugReportCallback, NULL);
    }
    vkDestroyInstance(mVkInstance, NULL);
}
void sQueue::fDealloc() {
    vkDestroyCommandPool(fDeviceID(), mVkPoolCommands, NULL);
}
void sTaskSequenceCompute::fDealloc() {
    vkDestroyPipelineLayout(fDeviceID(), mVkPipelineLayout, NULL);
    vkDestroyPipeline(fDeviceID(), mVkPipeline, NULL);
}
void sTask::fDealloc() {
    vkDestroyDescriptorPool(fDeviceID(), mVkDescriptorPool, NULL);
}
void sFuncCompiled::fDealloc() {
    vkDestroyShaderModule(fDeviceID(), mShader, NULL);
    vkDestroyDescriptorSetLayout(fDeviceID(), mVkDescriptorLayout, NULL);
}
void sVarWBuffer::fDealloc() {
    vkFreeMemory(fDeviceID(), mVkMemory, NULL);
    vkDestroyBuffer(fDeviceID(), mVkBuffer, NULL);
}

#pragma mark - Destructos

sDevice::~sDevice() {
    mQueuesCompute.clear();
    mQueuesGraphics.clear();
    fDealloc();
}

sTask::~sTask() {
    mVarPtrs.clear();
    fDealloc();
}

sTaskSequenceCompute::~sTaskSequenceCompute() {
    mTaskPtr->fDealloc();
    fDealloc();
}

sVarWBuffer::~sVarWBuffer() {
    if (mTaskPtr)
        fDealloc();
}
sFuncCompiled::~sFuncCompiled() { fDealloc(); }
sVulkan::~sVulkan() { fDealloc(); }
sQueue::~sQueue() { fDealloc(); }

#pragma mark - API

sVulkan::sVulkan() {
    auto & lVk = *this;
    lVk.fInitValidation();
    lVk.fInitInstance();
    lVk.fInitWarningsCallback();
}

SArr<SShared<sDevice>> sVulkan::fAddAllDevices() {
    fInitDevices();
    return mDevices;
}

#pragma mark - Device Accessors

VkDevice sFuncCompiled::fDeviceID() const {
    return mDevicePtr->mLogical;
}
VkDevice sVarWBuffer::fDeviceID() const {
    return mTaskPtr->fDeviceID();
}
VkDevice sTask::fDeviceID() const {
    return mFuncPtr->fDeviceID();
}
VkDevice sTaskSequenceCompute::fDeviceID() const {
    return mTaskPtr->fDeviceID();
}
VkDevice sQueue::fDeviceID() const {
    return mDevicePtr->mLogical;
}

