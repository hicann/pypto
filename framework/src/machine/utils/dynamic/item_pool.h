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
#include <limits>

namespace npu::tile_fwk::dynamic {

using ItemPoolIter = int64_t;

static constexpr ItemPoolIter ITEM_POOL_INVALID_INDEX = std::numeric_limits<int64_t>::max();
static constexpr ItemPoolIter ITEM_POOL_NON_FREE_INDEX = std::numeric_limits<int64_t>::max() - 1;

template <typename T, typename WsAllocator_T = WsMetadataAllocator>
class ItemPool {
public:
    struct ItemBlock {
        char buf[sizeof(T)];
        ItemPoolIter freeListNextIndex;

        T &Item() {
            return *reinterpret_cast<T *>(buf);
        }

        const T &Item() const {
            return *reinterpret_cast<const T *>(buf);
        }
    };

public:
    ItemPool() = default;
    ItemPool(WsAllocator_T &allocator, size_t count, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL) {
        Init(allocator, count, category);
    }

    ~ItemPool() {
        if (allocation_) {
            // Call destructor on alive items
            ItemBlock *itemBase = &ItemAt(0);
            for (size_t i = 0; i < count_; i++) {
                if (itemBase[i].freeListNextIndex == ITEM_POOL_NON_FREE_INDEX) {
                    (reinterpret_cast<T *>(itemBase + i))->~T();
                }
            }

            DEV_ASSERT(allocator_);
            allocator_->Deallocate(allocation_);
        }
    }

    void Init(WsAllocator_T &allocator, size_t count, WsMemCategory category = WsMemCategory::UNCLASSIFIED_ITEMPOOL) {
        DEV_ASSERT_MSG(!allocator_, "ItemPool has been initialized already");
        allocator_ = &allocator;
        count_ = count;
        allocation_ = allocator_->template Allocate<ItemBlock>(count_, category);
        ItemBlock *itemBase = &ItemAt(0);
        for (size_t i = 0; i < count_; i++) {
            AppendFreeList(itemBase + i);
        }

        category_ = category;
        freeCount_ = count_;
    }

    template <typename ...Args>
    T *Create(Args &&...args) {
        DEV_ASSERT_MSG(freeListHeadIndex_ != ITEM_POOL_INVALID_INDEX,
            "Available items: %zu/%zu", freeCount_, count_);
        ItemBlock *item = &ItemAt(freeListHeadIndex_);
        freeListHeadIndex_ = item->freeListNextIndex;
        item->freeListNextIndex = ITEM_POOL_NON_FREE_INDEX;
        freeCount_--;

        T *newItem = reinterpret_cast<T *>(item->buf);
        new(newItem) T(std::forward<Args>(args)...);
        return newItem;
    }

    template <typename ...Args>
    ItemPoolIter Allocate(Args &&...args) {
        T *item = Create(args...);
        return reinterpret_cast<ItemBlock *>(item) - &ItemAt(0);
    }

    void Destroy(T *item) {
        item->~T();
        ItemBlock *block = (ItemBlock *)item;
        DEV_ASSERT_MSG(block->freeListNextIndex == ITEM_POOL_NON_FREE_INDEX,
                       "Double free detected in ItemPool");
        AppendFreeList(block);
        freeCount_++;
    }

    T &At(ItemPoolIter index) {
        return ItemAt(index).Item();
    }

    void DestroyAt(ItemPoolIter index) {
        Destroy(&At(index));
    }

    size_t FreeItemNum() const {
        return freeCount_;
    }

private:
    inline void AppendFreeList(ItemBlock *block) {
        block->freeListNextIndex = freeListHeadIndex_;
        freeListHeadIndex_ = block - &ItemAt(0);
    }

    inline ItemBlock &ItemAt(ItemPoolIter index) {
        DEV_ASSERT_MSG(index >= 0 && static_cast<size_t>(index) < count_,
            "Index %" PRId64 " out of range [0, %zu)", index, count_);
        return allocation_.As<ItemBlock>()[index];
    }

private:
    WsMemCategory category_{WsMemCategory::UNCLASSIFIED_ITEMPOOL};
    WsAllocator_T *allocator_{nullptr};
    WsAllocation allocation_;
    size_t count_;
    size_t freeCount_;
    ItemPoolIter freeListHeadIndex_{ITEM_POOL_INVALID_INDEX};
};

} // namespace npu::tile_fwk::dynamic