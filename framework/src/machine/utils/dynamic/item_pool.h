/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file item_pool.h
 * \brief
 */

#pragma once

#include "machine/utils/dynamic/allocator/allocators.h"

namespace npu::tile_fwk::dynamic {

static constexpr int64_t itemPoolInvalidIndex = -1;

template <typename T = uint32_t, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL,
    typename WsAllocator_T = WsMetadataAllocator>
class ItemPool {
public:
    static constexpr int64_t createdIndex = -2;
public:
    struct ItemBlock {
        char buf[sizeof(T)];
        int64_t freeListNextIndex;
    };

public:
    ItemPool() = default;
    ItemPool(WsAllocator_T &allocator, size_t count) {
        Init(allocator, count);
    }

    ~ItemPool() {
        if (allocation_) {
            // Call destructor on alive items
            ItemBlock *itemBase = &ItemAt();
            for (size_t i = 0; i < count_; i++) {
                if (itemBase[i].freeListNextIndex == createdIndex) {
                    (reinterpret_cast<T *>(itemBase + i))->~T();
                }
            }

            if (!allocator_) {
                DEV_ERROR("allocation_ is nullptr\n");
            }
            DEV_ASSERT(allocator_);
            allocator_->Deallocate(allocation_);
        }
    }

    void Init(WsAllocator_T &allocator, size_t count) {
        if (allocator_) {
            DEV_ERROR("allocation_ is not nullptr\n");
        }
        DEV_ASSERT(!allocator_);
        allocator_ = &allocator;
        count_ = count;
        allocation_ = allocator_->template Allocate<ItemBlock>(count_, category);
        ItemBlock *itemBase = &ItemAt();
        for (size_t i = 0; i < count_; i++) {
            AppendFreeList(itemBase + i);
        }
    }

    template <typename ...Args>
    T *Create(Args &&...args) {
        if (freeListHeadIndex_ == itemPoolInvalidIndex) {
            DEV_ERROR("freeListHeadIndex_=%ld, itemPoolInvalidIndex_=%ld\n", freeListHeadIndex_, itemPoolInvalidIndex);
        }
        DEV_ASSERT(freeListHeadIndex_ != itemPoolInvalidIndex);
        ItemBlock *item = &ItemAt(freeListHeadIndex_);
        freeListHeadIndex_ = item->freeListNextIndex;
        item->freeListNextIndex = createdIndex;

        T *newItem = reinterpret_cast<T *>(item->buf);
        new(newItem) T(std::forward<Args>(args)...);
        return newItem;
    }

    template <typename ...Args>
    int64_t Allocate(Args &&...args) {
        T *item = Create(args...);
        return reinterpret_cast<ItemBlock *>(item) - &ItemAt(0);
    }

    void Destroy(T *item) {
        item->~T();
        ItemBlock *block = (ItemBlock *)item;
        AppendFreeList(block);
    }

    T &At(int64_t index) { return *reinterpret_cast<T *>(allocation_.As<ItemBlock>()[index].buf); }

    void DestroyAt(int64_t index) {
        Destroy(&At(index));
    }
private:
    inline void AppendFreeList(ItemBlock *block) {
        block->freeListNextIndex = freeListHeadIndex_;
        freeListHeadIndex_ = block - &ItemAt(0);
    }

    ItemBlock &ItemAt(int64_t index = 0) { return allocation_.As<ItemBlock>()[index]; }

private:
    WsAllocator_T *allocator_{nullptr};
    WsAllocation allocation_;
    size_t count_;
    int64_t freeListHeadIndex_{itemPoolInvalidIndex};
};

} // namespace npu::tile_fwk::dynamic