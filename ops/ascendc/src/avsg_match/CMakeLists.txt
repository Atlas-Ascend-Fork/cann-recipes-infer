# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

add_ops_compile_options(
        OP_NAME AvsgMatch
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

set(avsg_match_depends transformer/attention/avsg_match PARENT_SCOPE)

target_sources(op_host_aclnnExc PRIVATE
        op_host/avsg_match_def.cpp
)

target_sources(opapi PRIVATE
        op_api/aclnn_avsg_match.cpp
        op_api/avsg_match.cpp
)

# The ACLNN wrapper uses l0op::ViewCopy from the toolkit's libopapi.
target_link_libraries(opapi PRIVATE -lopapi)

target_include_directories(opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api
        ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/pkg_inc
)

target_sources(optiling PRIVATE
        op_host/avsg_match_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/avsg_match_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/avsg_match_proto.cpp
)

install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_avsg_match.h
        ${CMAKE_CURRENT_SOURCE_DIR}/op_api/avsg_match.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR}
)
