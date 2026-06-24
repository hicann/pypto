/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file function_cached_attribute.h
 * \brief
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>

#include "tilefwk/symbolic_scalar.h"
#include "interface/utils/entry_registrar.h"

namespace npu::tile_fwk {
namespace dynamic {
class DevAscendFunction;
}

class Function;

struct RebuildableAttributeBase {
    virtual void Rebuild(Function *func);
    virtual void Reset(void *data);
    virtual ~RebuildableAttributeBase() = default;
};

class RebuildableAttributeManager {
public:
    static EntryRegistrarGroup &GetRegistrarGroup();

    static RebuildableAttributeManager &GetInstance();

    template<typename T>
    T *GetAttr(Function *func)
    {
        if (attrDict_.find(func) == attrDict_.end()) {
            InitAttrsForFunc(func);
        }
        std::shared_ptr<RebuildableAttributeBase> attrBase = attrDict_[func][typeid(T).name()];
        std::shared_ptr<T> attr = std::static_pointer_cast<T>(attrBase);
        return attr.get();
    }

    template<typename T>
    void ResetAttr(Function *func, void *data)
    {
        GetAttr<T>(func)->Reset(data);
    }

    template<typename T>
    void BuildAttr(Function *func)
    {
        GetAttr<T>(func)->Build(func);
    }

    void InitAttr(Function *func, const std::string &name, std::shared_ptr<RebuildableAttributeBase> base)
    {
        attrDict_[func][name] = base;
    }

    RebuildableAttributeManager() = default;

private:
    void InitAttrsForFunc(Function *func);

    std::unordered_map<Function *, std::unordered_map<std::string, std::shared_ptr<RebuildableAttributeBase>>> attrDict_;
};

struct RebuildableAttrInitContext {
    RebuildableAttributeManager *manager;
    Function *func;
};

#define RBUILDABLE_ATTRIBUTE_REGISTER(TyAttr) \
    static void Entry##TyAttr(void *data) { \
        auto ctx = reinterpret_cast<RebuildableAttrInitContext *>(data); \
        auto ptr = std::make_shared<TyAttr>(); \
        auto base = std::static_pointer_cast<RebuildableAttributeBase>(ptr); \
        const std::string name = typeid(TyAttr).name(); \
        ctx->manager->InitAttr(ctx->func, name, base); \
    } \
    static EntryRegistrarNode node(RebuildableAttributeManager::GetRegistrarGroup(), Entry##TyAttr, #TyAttr);

struct WorkspaceDesc {
    struct WorkspaceConfig {
        uint64_t innerSpilledRecyclePeriod;
        uint64_t unrollStitchCount;
        uint64_t actualStitchCount;
        uint64_t parallelism;
    } config;

    struct WorkspacePlatform {
        uint64_t aicoreCount;
    } platform;

    struct WorkspacePerRootFunctionDesc {
        Function *func{nullptr};
        std::string devFuncName;
        uint64_t unroll{0};

        uint64_t rootInnerSpilledRawMem{0};
        uint64_t leafPerCoreSpilledMem{0};
        uint64_t rootTotalExclusiveOutcastRawMem{0};

        uint64_t rootInnerSpilledMem{0};
        uint64_t rootTotalExclusiveOutcastMem{0};

        uint64_t rootMaxExclusiveOutcastMem{0};
        int64_t rootMaxExclusiveOutcastIdx{0};
    };

    std::vector<WorkspacePerRootFunctionDesc> rootFuncDescList;

    uint64_t maxRootInnerSpilledMem{0};
    uint64_t maxLeafPerCoreSpilledMem{0};
    uint64_t maxRootTotalExclusiveOutcastMem{0};

    uint64_t maxStaticOutcastMem{0};
    SymbolicScalar maxDynamicAssembleOutcastMem;

    uint64_t totalExclusiveOutcastSlot{0};
    uint64_t totalAssembleOutcastSlot{0};
    uint64_t devTaskBoundaryOutcastNum{0};
    uint64_t devTaskInnerTemporalOutcastNum{0};

    struct WorkspaceCellMatch {
        uint64_t dynamicCellMatchSlotNum{0};
        SymbolicScalar maxDynamicCellMatchTableMem;
    } cellMatch;
};

struct RebuildableWorkspaceDesc : RebuildableAttributeBase {
    void Reset(void *data) override
    {
        desc = *static_cast<WorkspaceDesc *>(data);
    }

    WorkspaceDesc desc;

    uint64_t GetSizeForCheckOnly(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const;

    std::string PrettyDumpSize(uint64_t maxDynamicAssembleOutcastMem, uint64_t debugSize) const;
};

}
