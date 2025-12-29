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
 * \file device_channel.h
 * \brief
 */

#pragma once

#include <stdint.h>
#include <atomic>
#include <mutex>
#include <vector>
#include "machine/utils/barrier.h"
#include "securec.h"
#include "machine/utils/device_log.h"
#include "tilefwk/error.h"

#define ALIGN_UP_PTR(ptr, al) \
  (reinterpret_cast<decltype(ptr)>( \
    ( \
      reinterpret_cast<uintptr_t>(ptr) + static_cast<uintptr_t>((al) - 1) \
    ) & ~static_cast<uintptr_t>((al) - 1) \
  ))
constexpr int64_t CACHE_LINE_SIZE = 64;
constexpr int64_t INVALID_SLOT_ID = -1;

namespace npu::tile_fwk::dynamic {

class SpinLock {
public:
    void lock() {
        while (flag.test_and_set(std::memory_order_acquire))
            cpuRelax();
    }
    void unlock() { flag.clear(std::memory_order_release); }

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

enum StatusType { IDLE, START, ACKED, FIN };

struct Slot {
    volatile uint64_t status;
    volatile uint64_t taskId;
    volatile uint64_t taskData;
};

class DeviceTaskSender {
public:
    void init(uint8_t *addr, int size) {
        uint8_t *naddr = ALIGN_UP_PTR(addr, CACHE_LINE_SIZE);
        nextSlot_ = reinterpret_cast<int32_t *>(naddr); // nextSlot_ takes a whole cache line
        size -= naddr - addr + CACHE_LINE_SIZE;
        nrSlot_ = size / sizeof(struct Slot);
        slots_ = reinterpret_cast<Slot*>(naddr + CACHE_LINE_SIZE);
        ASSERT(nrSlot_ > 0 && "size too small")<<"nrSlot_= "<<nrSlot_<<", expected >0";
        DEV_DEBUG("nrSlot addr %p %ld \n", nextSlot_, nrSlot_);

        reset();
    }

    void reset() {
        SetNextSlot(INVALID_SLOT_ID);
        status_.resize(nrSlot_, 0);
        memset_s(slots_, nrSlot_ * sizeof(Slot), 0, nrSlot_ * sizeof(Slot));
    }

    int send(int64_t taskId, int64_t taskData) {
        int slotId = INVALID_SLOT_ID;
        while (!trySend(taskId, taskData, slotId)) {
            cpuRelax();
        }
        return slotId;
    }

    bool trySend(int64_t taskId, int64_t taskData, int &slot) {
        slot = *nextSlot_;

        // current still not read
        if (slot != INVALID_SLOT_ID && slots_[slot].status == START) {
            return false;
        }

        slot = slotAlloc();
        if (slot != -1) {
            DEV_DEBUG("req taskId %ld, slot %d\n", taskId, slot);
            sendTask(slot, taskId, taskData);
            return true;
        }
        return false;
    }

    void sync(int slotId) {
        ASSERT(slotId < nrSlot_)<<"slotId="<<slotId<<" >= nrSlot_="<<nrSlot_;
        while (!trySync(slotId)) {
            cpuRelax();
        }
        std::lock_guard<SpinLock> lock(lock_);
        slots_[slotId].status = IDLE;
        status_[slotId] = 0;
    }

    bool trySync(int slotId) {
        volatile int *status = &status_[slotId];
        if (*status == 0) {
            return true;
        }
        return slots_[slotId].status == FIN;
    }

private:
    int slotAlloc() {
        std::lock_guard<SpinLock> lock(lock_);
        for (int i = 0; i < nrSlot_; i++) {
            if (status_[i] == 0) {
                DEV_DEBUG("alloc slot %d\n", i);
                status_[i] = 1;
                return i;
            }
        }
        return -1;
    }

    void sendTask(int32_t n, int64_t taskId, int64_t taskData) const {
        slots_[n].taskId = static_cast<uint64_t>(taskId);
        slots_[n].taskData = static_cast<uint64_t>(taskData);
        slots_[n].status = START;
        SetNextSlot(n);
    }

    void SetNextSlot(int32_t slot) const {
        wmb();
        *nextSlot_ = slot;
    }

private:
    volatile int32_t *nextSlot_;
    SpinLock lock_;
    Slot *slots_;
    int64_t nrSlot_;
    std::vector<int> status_;
};

class DeviceTaskReceiver {
public:
    void init(uint8_t *addr, int size) {
        uint8_t *naddr = ALIGN_UP_PTR(addr, CACHE_LINE_SIZE);
        nextSlot_ = reinterpret_cast<int32_t *>(naddr); // nextSlot_ takes a whole cache line
        size -= naddr - addr + CACHE_LINE_SIZE;
        nrSlot_ = size / sizeof(struct Slot);
        slots_ = reinterpret_cast<Slot*>(naddr + CACHE_LINE_SIZE);
        ASSERT(nrSlot_ > 0 && "size too small")<<", nrSlot_="<<nrSlot_<<", expected >0";
        DEV_DEBUG("nrSlot addr %p %ld\n", nextSlot_, nrSlot_);
    }

    void Finish(int64_t slot) {
        ASSERT(slot < nrSlot_)<<"slot="<<slot<<" >= nrSlot_="<<nrSlot_;

        DEV_DEBUG("fin taskId %lx slot %ld\n", slots_[slot].taskId, slot);
        std::lock_guard<SpinLock> lock(lock_);
        slots_[slot].status = FIN;
    }

    int recv(int64_t &taskId, int64_t &taskData) {
        int slot = INVALID_SLOT_ID;
        while (!TryRecv(taskId, taskData, slot)) {
            cpuRelax();
        }
        return slot;
    }

    bool TryRecv(int64_t &taskId, int64_t &taskData, int &slot) {
        slot = *nextSlot_;
        if (slot == INVALID_SLOT_ID) {
            return false;
        }

        std::lock_guard<SpinLock> lock(lock_);
        if (slots_[slot].status == START) {
            taskId = static_cast<int64_t>(slots_[slot].taskId);
            taskData = static_cast<int64_t>(slots_[slot].taskData);
            DEV_DEBUG("ack taskId %ld, slot %d\n", taskId, slot);
            slots_[slot].status = ACKED; // next slot will be updated after acked
            return true;
        }
        return false;
    }

private:
    volatile int32_t *nextSlot_;
    SpinLock lock_;
    Slot *slots_;
    int64_t nrSlot_;
};
} // namespace npu::tile_fwk
