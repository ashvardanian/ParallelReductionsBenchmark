// Project: SandboxGPUs.
// Author: Ashot Vardanian.
// Created: 04/09/2019.
// Copyright: Check "License" file.
//

#include "HelperAliasis.hpp"
#include "Tests.hpp"
#include <algorithm>
#include <CL/sycl.hpp>
#include <SYCL/sycl.hpp>

namespace nSy = cl::sycl;
namespace syi = nSy::info;

std::map<syi::device_type, sStr> device_type_representation {
    {syi::device_type::cpu, "CPU"},
    {syi::device_type::gpu, "GPU"},
    {syi::device_type::accelerator, "Accelerator"},
    {syi::device_type::custom, "Custom"},
    {syi::device_type::automatic, "Automatic"},
    {syi::device_type::host, "Host"},
    {syi::device_type::all, "All"}
};

std::map<syi::global_mem_cache_type, sStr> global_mem_cache_type_representation {
    {syi::global_mem_cache_type::none, "None"},
    {syi::global_mem_cache_type::read_only, "Read-only"},
    {syi::global_mem_cache_type::write_only, "Write-only"}
};


std::map<syi::local_mem_type, sStr> local_mem_type_representation {
    {syi::local_mem_type::none, "None"},
    {syi::local_mem_type::local, "Local"},
    {syi::local_mem_type::global, "Global"}
};

std::map<syi::fp_config, sStr> fp_config_representation {
    { syi::fp_config::denorm, "denorm"},
    { syi::fp_config::inf_nan, "inf_nan"},
    { syi::fp_config::round_to_nearest, "round_to_nearest"},
    { syi::fp_config::round_to_zero, "round_to_zero"},
    { syi::fp_config::round_to_inf, "round_to_inf"},
    { syi::fp_config::fma, "fma"},
    { syi::fp_config::correctly_rounded_divide_sqrt, "correctly_rounded_divide_sqrt"},
    { syi::fp_config::soft_float, "soft_float" }
};

std::map<syi::execution_capability, sStr> exec_capability_representation{
    {syi::execution_capability::exec_kernel, "exec_kernel"},
    {syi::execution_capability::exec_native_kernel, "exec_native_kernel"}
};


std::map<syi::partition_property, sStr>
partition_property_representation {
    {syi::partition_property::no_partition, "no_partition"},
    {syi::partition_property::partition_equally, "partition_equally"},
    {syi::partition_property::partition_by_counts, "partition_by_counts"},
    {syi::partition_property::partition_by_affinity_domain, "partition_by_affinity_domain"}
};

std::map<syi::partition_affinity_domain, sStr>
partition_affinity_domain_representation {
    {syi::partition_affinity_domain::not_applicable, "not_applicable"},
    {syi::partition_affinity_domain::numa, "numa"},
    {syi::partition_affinity_domain::L4_cache, "L4_cache"},
    {syi::partition_affinity_domain::L3_cache, "L3_cache"},
    {syi::partition_affinity_domain::L2_cache, "L2_cache"},
    {syi::partition_affinity_domain::L1_cache, "L1_cache"},
    {syi::partition_affinity_domain::next_partitionable, "next_partitionable"}
};

std::ostream & operator<<(std::ostream& lhs, const nSy::id<3>& idx) {
    lhs << idx[0] << " " << idx[1] << " " << idx[2];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::device_type dtype) {
    lhs << device_type_representation[dtype];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::global_mem_cache_type cache_type) {
    lhs << global_mem_cache_type_representation[cache_type];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::local_mem_type local_type) {
    lhs << local_mem_type_representation[local_type];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::fp_config fpconfig) {
    lhs << fp_config_representation[fpconfig];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::execution_capability ecap) {
    lhs << exec_capability_representation[ecap];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::partition_property pprop) {
    lhs << partition_property_representation[pprop];
    return lhs;
}

std::ostream& operator<<(std::ostream& lhs, syi::partition_affinity_domain domain) {
    lhs << partition_affinity_domain_representation[domain];
    return lhs;
}

template <class T>
std::ostream & operator << (std::ostream& lhs, const SArr<T> & rhs) {
    for (bSize i = 0; i < rhs.size(); ++i) {
        lhs << rhs[i];
        if (i != rhs.size()-1)
            lhs << ", ";
    }
    return lhs;
}

#define PRINT_PLATFORM_PROPERTY(plat, prop) \
std::cout << #prop << ": " \
<< plat.get_info<syi::platform::prop>() << std::endl;

#define PRINT_DEVICE_PROPERTY(dev, prop) \
std::cout << #prop << ": " \
<< dev.get_info<syi::device::prop>() << std::endl;

void nSyCL::gPrint() {
    
    SArr<nSy::device> devices = nSy::device::get_devices();
    for(const auto & dev : devices) {
        std::cout << "***************************************" << std::endl;
        std::cout << "           Platform:                   " << std::endl;
        std::cout << "***************************************" << std::endl;
        
        nSy::platform plat = dev.get_platform();
        
        PRINT_PLATFORM_PROPERTY(plat, name);
        PRINT_PLATFORM_PROPERTY(plat, vendor);
        PRINT_PLATFORM_PROPERTY(plat, version);
        PRINT_PLATFORM_PROPERTY(plat, profile);
        PRINT_PLATFORM_PROPERTY(plat, extensions);
        
        std::cout << "============ Found device: =============" << std::endl;
        PRINT_DEVICE_PROPERTY(dev, vendor);
        PRINT_DEVICE_PROPERTY(dev, name);
        PRINT_DEVICE_PROPERTY(dev, driver_version);
        PRINT_DEVICE_PROPERTY(dev, profile);
        PRINT_DEVICE_PROPERTY(dev, version);
        PRINT_DEVICE_PROPERTY(dev, opencl_c_version);
        PRINT_DEVICE_PROPERTY(dev, extensions);
        PRINT_DEVICE_PROPERTY(dev, device_type);
        PRINT_DEVICE_PROPERTY(dev, vendor_id);
        PRINT_DEVICE_PROPERTY(dev, max_compute_units);
        PRINT_DEVICE_PROPERTY(dev, max_work_item_dimensions);
        PRINT_DEVICE_PROPERTY(dev, max_work_item_sizes);
        PRINT_DEVICE_PROPERTY(dev, max_work_group_size);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_char);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_short);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_int);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_long);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_float);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_double);
        PRINT_DEVICE_PROPERTY(dev, preferred_vector_width_half);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_char);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_short);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_int);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_long);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_float);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_double);
        PRINT_DEVICE_PROPERTY(dev, native_vector_width_half);
        PRINT_DEVICE_PROPERTY(dev, max_clock_frequency);
        PRINT_DEVICE_PROPERTY(dev, address_bits);
        PRINT_DEVICE_PROPERTY(dev, max_mem_alloc_size);
        PRINT_DEVICE_PROPERTY(dev, image_support);
        PRINT_DEVICE_PROPERTY(dev, max_read_image_args);
        PRINT_DEVICE_PROPERTY(dev, max_write_image_args);
        PRINT_DEVICE_PROPERTY(dev, image2d_max_height);
        PRINT_DEVICE_PROPERTY(dev, image2d_max_width);
        PRINT_DEVICE_PROPERTY(dev, image3d_max_height);
        PRINT_DEVICE_PROPERTY(dev, image3d_max_width);
        PRINT_DEVICE_PROPERTY(dev, image3d_max_depth);
        PRINT_DEVICE_PROPERTY(dev, image_max_buffer_size);
        PRINT_DEVICE_PROPERTY(dev, image_max_array_size);
        PRINT_DEVICE_PROPERTY(dev, max_samplers);
        PRINT_DEVICE_PROPERTY(dev, max_parameter_size);
        PRINT_DEVICE_PROPERTY(dev, mem_base_addr_align);
        
        PRINT_DEVICE_PROPERTY(dev, half_fp_config);
        PRINT_DEVICE_PROPERTY(dev, single_fp_config);
        PRINT_DEVICE_PROPERTY(dev, double_fp_config);
        PRINT_DEVICE_PROPERTY(dev, global_mem_cache_type);
        PRINT_DEVICE_PROPERTY(dev, global_mem_cache_line_size);
        PRINT_DEVICE_PROPERTY(dev, global_mem_cache_size);
        PRINT_DEVICE_PROPERTY(dev, global_mem_size);
        PRINT_DEVICE_PROPERTY(dev, max_constant_buffer_size);
        PRINT_DEVICE_PROPERTY(dev, max_constant_args);
        PRINT_DEVICE_PROPERTY(dev, local_mem_type);
        PRINT_DEVICE_PROPERTY(dev, local_mem_size);
        PRINT_DEVICE_PROPERTY(dev, error_correction_support);
        PRINT_DEVICE_PROPERTY(dev, host_unified_memory);
        PRINT_DEVICE_PROPERTY(dev, profiling_timer_resolution);
        PRINT_DEVICE_PROPERTY(dev, is_endian_little);
        PRINT_DEVICE_PROPERTY(dev, is_available);
        PRINT_DEVICE_PROPERTY(dev, is_compiler_available);
        PRINT_DEVICE_PROPERTY(dev, is_linker_available);
        PRINT_DEVICE_PROPERTY(dev, execution_capabilities);
        PRINT_DEVICE_PROPERTY(dev, queue_profiling);
        PRINT_DEVICE_PROPERTY(dev, built_in_kernels);
        
        
        PRINT_DEVICE_PROPERTY(dev, printf_buffer_size);
        PRINT_DEVICE_PROPERTY(dev, preferred_interop_user_sync);
        PRINT_DEVICE_PROPERTY(dev, partition_max_sub_devices);
        
        PRINT_DEVICE_PROPERTY(dev, partition_properties);
        PRINT_DEVICE_PROPERTY(dev, partition_affinity_domains);
        PRINT_DEVICE_PROPERTY(dev, partition_type_property);
        PRINT_DEVICE_PROPERTY(dev, partition_type_affinity_domain);
        PRINT_DEVICE_PROPERTY(dev, reference_count);
        
        std::cout << std::endl << std::endl;
    }
}
