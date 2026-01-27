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
 * \file device_perf.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <string>
#include "tilefwk/aicpu_common.h"

namespace npu::tile_fwk::dynamic {
struct PerfettoMgr {
    static const int MAX_THEAD_NUM = 200;
    static const int TRUNK_SIZE = 32768;
    static const int MAX_EVT_DEPTH = 64;

    struct Record {
        int type;
        int tid;
        uint64_t start;
        uint64_t end;
        uint32_t index;
        uint32_t pIndex;
        std::string name = "-";
    };

    template <typename T, int N>
    struct Array {
        size_t used{0};
        T data[N];

        inline void Push(const T &t) { data[used++] = t; }
        inline void Pop() { used--; }
        inline bool Full() { return used == N; }
        inline bool Empty() { return used == 0; }
        inline T &Top() { return data[used - 1]; }
        inline T *Alloc() { return &data[used++]; }
    };

    using Trunk = Array<Record, TRUNK_SIZE>;
    using EvtStack = Array<Record *, MAX_EVT_DEPTH>;

    Record *allocRecord(int tid) {
        if (trunk_[tid] == nullptr || trunk_[tid]->Full()) {
            trunk_[tid] = new Trunk;
            mutex_.lock();
            trunks_.push_back(trunk_[tid]);
            mutex_.unlock();
        }
        return trunk_[tid]->Alloc();
    }

    void PerfBegin(int type, int tid) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto &stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->start = GetCycles();
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
        stack.Push(r);
#endif
        (void)type;
        (void)tid;
    }

    void PerfEnd(int type, int tid) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto &stack = evtStack[tid];
        auto r = stack.Top();
        r->end = GetCycles();
        stack.Pop();
#endif
        (void)type;
        (void)tid;
    }

    void PerfEvent(int type, int tid, uint64_t start, uint64_t end, std::string name) {
#if defined(CONFIG_PERFETTO) && CONFIG_PERFETTO
        auto r = allocRecord(tid);
        auto &stack = evtStack[tid];
        r->type = type;
        r->tid = tid;
        r->name = name;
        r->start = start;
        r->end = end;
        r->index = evtIndex++;
        r->pIndex = stack.Empty() ? -1 : stack.Top()->index;
#endif
        (void)type;
        (void)tid;
        (void)start;
        (void)end;
        (void)name;
    }

    static PerfettoMgr &Instance() {
        static PerfettoMgr recorder;
        return recorder;
    }

    void Dump(const std::string &file) {
        std::ofstream os(file);
        for (auto &trunk : trunks_) {
            for (size_t i = 0; i < trunk->used; i++) {
                auto &r = trunk->data[i];
                os << PerfEventName[r.type] << " ";
                os << r.name << " ";
                os << r.index << " ";
                os << r.start << " ";
                os << r.end << " ";
                os << r.tid << ";";
                os << r.pIndex << " ";
                os << std::endl;
            }
        }
    }

private:
    PerfettoMgr() = default;

private:
    EvtStack evtStack[MAX_THEAD_NUM];
    Trunk *trunk_[MAX_THEAD_NUM] = {nullptr};

    std::mutex mutex_;
    std::vector<Trunk *> trunks_;
    std::atomic<int32_t> evtIndex;
};
struct PerfEvtMgr {
    struct Counter {
        int64_t start;
        int64_t total;
        int64_t count;
    };

    bool GetIsOpenProf() {
        return isOpenProf_;
    }

    void SetIsOpenProf(bool isOpenProf, uint64_t aicpuPerf = 0) {
        isOpenProf_ = isOpenProf;
        aicpuPref_ = (MetricPerf*)aicpuPerf;
    }

    void PerfBegin(int type) {
        counters[type].start = static_cast<int64_t>(GetCycles());
    }

    void PerfEnd(int type) {
        auto &c = counters[type];
        c.count++;
        c.total += static_cast<int64_t>(GetCycles() - c.start);
    }

    static PerfEvtMgr &Instance() {
        static PerfEvtMgr recorder;
        return recorder;
    }

    static void RepeatPuts(char c, size_t count) {
        char buf[80];
        for (size_t i = 0; i < count; i++) {
            buf[i] = c;
        }
        buf[count] = '\0';
        DEV_ERROR("%s.", buf);
    }

    void Dump() {
        uint64_t freq = GetFreq();
        static constexpr size_t SHEET_WIDTH = 40 + 3 + 10 + 3 + 10 + 3 + 10;

        RepeatPuts('=', SHEET_WIDTH);
        DEV_ERROR("%40s | %10s | %10s | %10s.", "EventType", "Count", "Total(us)", "Avg(us)");
        RepeatPuts('-', SHEET_WIDTH);

        for (int i = 0; i < PERF_EVT_MAX; i++) {
            auto evt = counters[i];
            if (evt.count != 0) {
                uint64_t total = evt.total * NSEC_PER_SEC / freq / NSEC_PER_USEC;
                float avg = static_cast<float>(total / evt.count);
                DEV_ERROR("%-40s | %10ld | %10lu | %10.1f.", PerfEventName[i], evt.count, total, avg);
            }
        }

        RepeatPuts('=', SHEET_WIDTH);
    }

    void PerfTrace(uint32_t type, uint32_t tid, uint64_t cycle) {
        if (tid >= MAX_USED_AICPU_NUM) {
            return;
        }
        if (PerfTraceIsDevTask[type] && DEVTASK_PERF_ARRY_INDEX(type) < DEVTASK_PERF_TYPE_NUM) {
            auto &cnt = perfTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)];
            if (cnt < PERF_TRACE_COUNT_DEVTASK_MAX_NUM) {
                perfTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][cnt++] =
                    cycle == 0 ? static_cast<uint64_t>(GetCycles()) : cycle;
                if (aicpuPref_ != nullptr) {
                    uint8_t devCnt = aicpuPref_->perfAicpuTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)];
                    aicpuPref_->perfAicpuTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)] += 1;
                    aicpuPref_->perfAicpuTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][devCnt] = perfTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][devCnt];
                }
            }
            return;
        }
        perfTrace[tid][type] = cycle == 0 ? static_cast<uint64_t>(GetCycles()) : cycle;
        if (aicpuPref_ != nullptr) {
            aicpuPref_->perfAicpuTrace[tid][type] = perfTrace[tid][type];
        }
    }

    void DumpPerfTraceCore(std::ostringstream &oss, uint32_t scheCpuNum) {
        auto devTaskPerfFormatFunc = [this](std::ostringstream &osStr, uint32_t tid, uint32_t type) -> void {
            for (uint32_t i = 0; i < perfTraceDevTaskCnt[tid][DEVTASK_PERF_ARRY_INDEX(type)]; i++) {
                if (type == PERF_TRACE_DEV_TASK_SEND_FIRST_CALLOP_TASK) {
                    osStr << "{\"name\":\"" << PerfTraceName[type] << "\",";
                } else {
                    osStr << "{\"name\":\"" << PerfTraceName[type] << "(" << i << ")\",";
                }
                osStr << "\"end\":" << perfTraceDevTask[tid][DEVTASK_PERF_ARRY_INDEX(type)][i] << "},";
            }
        };

        uint64_t freq = GetFreq() / (NSEC_PER_SEC / NSEC_PER_USEC);
        uint32_t usedAicpuNum = scheCpuNum + MAX_OTHER_AICPU_NUM;
        for (uint32_t tid = 0 ; tid < usedAicpuNum; tid++) {
            std::string coreType = "\"AICPU\"";
            if (tid == 0) {
                coreType = "\"AICPU-CTRL\"";
            } else if (tid <= scheCpuNum) {
                coreType = "\"AICPU-SCHED\"";
            }
            oss << "{\"blockIdx\":" << tid << ",\"coreType\":" << coreType << ",\"freq\":"<< freq <<",\"tasks\":[";
            for (uint32_t type = 0; type < PERF_TRACE_MAX; type++) {
                if (PerfTraceIsDevTask[type]) {
                    devTaskPerfFormatFunc(oss, tid, type);
                    continue;
                }
                if (perfTrace[tid][type] == 0) {
                    continue;
                }
                oss << "{\"name\":\"" << PerfTraceName[type] << "\",\"end\":" << perfTrace[tid][type]
                    << "}" << (type == PERF_TRACE_MAX - 1 ? "" : ",");
            }
            oss << "]}" << (tid == usedAicpuNum - 1 ? "" : ",");
        }
    }

    void DumpPerfTrace(uint32_t scheCpuNum, std::string file = "") {
        (void)file;
        (void)scheCpuNum;
#if ENABLE_PERF_TRACE
        std::ostringstream oss;
        DumpPerfTraceCore(oss, scheCpuNum);
        const std::string& str = oss.str();
        uint32_t totalLength = str.length();
        uint32_t startPos = 0;
        uint32_t batchSize = 600;
        while (startPos < totalLength) {
            uint32_t endPos = std::min(startPos + batchSize, totalLength);
            std::string batch = str.substr(startPos, endPos - startPos);
            DEV_ERROR("tile_fwk aicpu prof:%s", batch.c_str());
            startPos = endPos;
        }

        if (file != "") {
            std::ofstream os(file);
            os << "[";
            os << oss.str();
            os << "]";
        }
        ResetPerfTrace();
#endif
        return;
    }

private:
    PerfEvtMgr() {
#if ENABLE_PERF_EVT
        memset_s(counters, sizeof(counters), 0, sizeof(counters));
#endif
#if ENABLE_PERF_TRACE
        ResetPerfTrace();
#endif
    };

    void ResetPerfTrace() {
        memset_s(perfTrace, sizeof(perfTrace), 0, sizeof(perfTrace));
        memset_s(perfTraceDevTask, sizeof(perfTraceDevTask), 0, sizeof(perfTraceDevTask));
        memset_s(perfTraceDevTaskCnt, sizeof(perfTraceDevTaskCnt), 0, sizeof(perfTraceDevTaskCnt));
    }

private:
    Counter counters[PERF_EVT_MAX];
    uint64_t perfTrace[MAX_USED_AICPU_NUM][PERF_TRACE_MAX] = {{0}};
    uint64_t perfTraceDevTask[MAX_USED_AICPU_NUM][DEVTASK_PERF_TYPE_NUM][PERF_TRACE_COUNT_DEVTASK_MAX_NUM] = {{{0}}};
    uint8_t perfTraceDevTaskCnt[MAX_USED_AICPU_NUM][DEVTASK_PERF_TYPE_NUM] = {{0}};
    bool isOpenProf_{false};
    MetricPerf *aicpuPref_{nullptr};
};

inline void PerfBegin(int type) {
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfBegin(type);
#else
    (void)type;
#endif
}

inline void PerfEnd(int type) {
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfEnd(type);
#else
  (void)type;
#endif
}

inline void PerfMtBegin(int type, int tid) {
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfBegin(type + tid);
#else
  (void)type;
  (void)tid;
#endif
}

inline void PerfMtEnd(int type, int tid) {
#if ENABLE_PERF_EVT
    PerfEvtMgr::Instance().PerfEnd(type + tid);
#else
  (void)type;
  (void)tid;
#endif
}

inline void PerfMtEvent(int type, int tid, uint64_t start, uint64_t end, std::string name = "-") {
    if (PerfEvtEnable[type]) {
        PerfettoMgr::Instance().PerfEvent(type, tid, start, end, name);
    }
}

inline void PerfMtTrace(uint32_t type, uint32_t tid, uint64_t cycle = 0) {
    (void)type;
    (void)tid;
    (void)cycle;
    if (unlikely(ENABLE_PERF_TRACE == 1 || PerfEvtMgr::Instance().GetIsOpenProf())) {
        PerfEvtMgr::Instance().PerfTrace(type, tid, cycle);
    }
}

struct AutoScopedPerf {
    explicit AutoScopedPerf(int type) : type_(type) { PerfBegin(type); }
    ~AutoScopedPerf() { PerfEnd(type_); }
    int type_;
};
}
