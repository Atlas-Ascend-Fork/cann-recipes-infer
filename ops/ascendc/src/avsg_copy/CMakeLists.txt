# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

add_ops_compile_options(
        OP_NAME AvsgCopy
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnExc PRIVATE
        op_host/avsg_copy_def.cpp
)

target_sources(opapi PRIVATE
        op_api/aclnn_avsg_copy.cpp
        op_api/avsg_copy.cpp
)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
        ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc
)

target_sources(optiling PRIVATE
        op_host/avsg_copy_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/avsg_copy_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/avsg_copy_proto.cpp
)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_avsg_copy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/avsg_copy.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
