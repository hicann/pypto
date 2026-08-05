#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rebuild PyPTO DeviceTask, CoreTask and MIX lifecycles from trace logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPORT_NAME = "trace_report.txt"
LOG_SUFFIXES = {".log", ".txt", ".out"}
TASK_INIT = 0xFFFFFFFF

TAG_PATTERN = re.compile(r"#trace\.([A-Za-z0-9_.]+):")
FIELD_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)=(-?(?:0[xX][0-9A-Fa-f]+|\d+)|[^\s\"]+)"
)
TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\.(\d{3})\.(\d{3})"
)
DEVTASK_BUILT_PATTERN = re.compile(r"\bdtaskId\s+(?:=\s*)?(\d+)")

CORE_TYPE_NAMES = {0: "AIV", 1: "AIC"}
MIX_TYPE_NAMES = {0: "UNKNOWN", 1: "1C1V", 2: "1C2V"}
MIX_ROLE_NAMES = {0: "AIC", 1: "AIV0", 2: "AIV1"}
DFX_STAGE_NAMES = {
    1: "HANDSHAKE_START",
    2: "HANDSHAKE_END",
    3: "CORE_EXIT",
    4: "GET_NEXT_TASK_STOP",
    5: "PRE_EXEC_COREFUNC_KERNEL",
    6: "FINISH_EXEC_COREFUNC_KERNEL",
    7: "FINISH_PIPE_SYNC",
    8: "FINISH_CUR_TASK",
    9: "GET_PARALLEL_DEVTASK_TIMEOUT",
    10: "GET_NEXT_TASK_TIMEOUT",
    11: "WAVE_TIMEOUT",
    12: "GET_HIGH_REG_TIMEOUT",
    13: "UPDATE_PARALLEL_DEVTASK_TIMEOUT",
    14: "UPDATE_PARALLEL_DEVTASK_ID_TIMEOUT",
    15: "RUN_LEAFTASK_TIMEOUT",
    16: "GET_FUNCDATA_STOP",
}


def parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value, 0)
    except ValueError:
        return default


def parse_timestamp(line: str) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.search(line)
    if match is None:
        return None
    timestamp = f"{match.group(1)}.{match.group(2)}{match.group(3)}"
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d-%H:%M:%S.%f")
    except ValueError:
        return None


def format_timestamp(value: Optional[datetime]) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")


def format_duration(start: Optional[datetime], end: Optional[datetime]) -> str:
    if start is None or end is None:
        return "-"
    duration_ms = (end - start).total_seconds() * 1000
    return f"{duration_ms:.3f} ms"


@dataclass
class TraceEvent:
    tag: str
    fields: Dict[str, str]
    timestamp: Optional[datetime]
    source: Path
    line_number: int
    input_order: int
    sequence: int = 0
    round_id: int = 0

    def get_int(self, name: str, default: Optional[int] = None) -> Optional[int]:
        return parse_int(self.fields.get(name), default)

    @property
    def reference(self) -> str:
        return f"{self.source.name}:{self.line_number}"


@dataclass
class DeviceTaskRecord:
    round_id: int
    task_id: int
    built_events: List[TraceEvent] = field(default_factory=list)
    start_events: List[TraceEvent] = field(default_factory=list)
    end_events: List[TraceEvent] = field(default_factory=list)
    abnormal_end_events: List[TraceEvent] = field(default_factory=list)
    core_counts: Set[int] = field(default_factory=set)
    finished_counts: Set[int] = field(default_factory=set)
    is_last: bool = False

    @property
    def start_time(self) -> Optional[datetime]:
        values = [event.timestamp for event in self.start_events if event.timestamp is not None]
        return min(values) if values else None

    @property
    def end_time(self) -> Optional[datetime]:
        events = self.end_events or self.abnormal_end_events
        values = [event.timestamp for event in events if event.timestamp is not None]
        return max(values) if values else None

    @property
    def start_sequence(self) -> Optional[int]:
        return min((event.sequence for event in self.start_events), default=None)

    @property
    def end_sequence(self) -> Optional[int]:
        events = self.end_events or self.abnormal_end_events
        return max((event.sequence for event in events), default=None)

    @property
    def expected_core_count(self) -> Optional[int]:
        if len(self.core_counts) != 1:
            return None
        return next(iter(self.core_counts))

    @property
    def finished_core_count(self) -> Optional[int]:
        if self.abnormal_end_events and not self.end_events:
            return max(self.finished_counts, default=None)
        if len(self.finished_counts) != 1:
            return None
        return next(iter(self.finished_counts))

    @property
    def has_execution_events(self) -> bool:
        return bool(self.start_events or self.end_events or self.abnormal_end_events)

    def add(self, event: TraceEvent) -> None:
        if event.tag == "dtask.built":
            self.built_events.append(event)
        elif event.tag == "dtask.start":
            self.start_events.append(event)
            core_count = event.get_int("coreFunctionCnt")
            if core_count is not None:
                self.core_counts.add(core_count)
            self.is_last = self.is_last or event.get_int("isLast", 0) == 1
        elif event.tag == "dtask.end":
            self.end_events.append(event)
            core_count = event.get_int("coreFunctionCnt")
            finished_count = event.get_int("finishedFunctionCnt")
            if core_count is not None:
                self.core_counts.add(core_count)
            if finished_count is not None:
                self.finished_counts.add(finished_count)
        elif event.tag == "dtask.abnormalend":
            self.abnormal_end_events.append(event)
            finished_count = event.get_int("finishedFunctionCnt")
            if finished_count is not None:
                self.finished_counts.add(finished_count)

    def structural_issues(self) -> List[str]:
        issues = []
        if not self.start_events:
            issues.append("缺少 DeviceTask start")
        if self.abnormal_end_events:
            pass
        elif not self.end_events:
            issues.append("缺少 DeviceTask end")
        if len(self.core_counts) > 1:
            issues.append(f"coreFunctionCnt 不一致: {sorted(self.core_counts)}")
        if self.end_events and len(self.finished_counts) > 1:
            issues.append(f"finishedFunctionCnt 不一致: {sorted(self.finished_counts)}")
        expected = self.expected_core_count
        finished = self.finished_core_count
        if expected is not None and finished is not None and expected != finished:
            suffix = "（异常结束）" if self.abnormal_end_events else ""
            issues.append(f"CoreTask 未全部执行结束: {finished}/{expected}{suffix}")
        elif self.abnormal_end_events:
            issues.append("DeviceTask 异常结束")
        start_tids = {
            event.get_int("tid") for event in self.start_events if event.get_int("tid") is not None
        }
        end_tids = {
            event.get_int("tid") for event in self.end_events if event.get_int("tid") is not None
        }
        missing_end_tids = sorted(start_tids - end_tids)
        if self.end_events and missing_end_tids:
            issues.append(f"调度线程缺少 end: {missing_end_tids}")
        return issues


@dataclass
class CoreTaskLifecycle:
    round_id: int
    dev_task_id: int
    task_id: int
    first_batch_resolves: List[TraceEvent] = field(default_factory=list)
    runtime_resolves: List[TraceEvent] = field(default_factory=list)
    sends: List[TraceEvent] = field(default_factory=list)
    acks: List[TraceEvent] = field(default_factory=list)
    finishes: List[TraceEvent] = field(default_factory=list)
    mix_resolved: bool = False

    def add(self, event: TraceEvent) -> None:
        if event.tag == "ltask.resolve":
            if event.get_int("firstBatch", 0) == 1:
                self.first_batch_resolves.append(event)
            else:
                self.runtime_resolves.append(event)
        elif event.tag == "ltask.send":
            self.sends.append(event)
        elif event.tag == "ltask.ack":
            self.acks.append(event)
        elif event.tag == "ltask.finish":
            self.finishes.append(event)

    @property
    def resolved(self) -> bool:
        return bool(self.first_batch_resolves or self.runtime_resolves or self.mix_resolved)

    @property
    def scheduler_ids(self) -> List[int]:
        result = {
            event.get_int("tid")
            for event in self.sends + self.acks + self.finishes
            if event.get_int("tid") is not None
        }
        return sorted(result)

    @property
    def core_names(self) -> List[str]:
        result = set()
        for event in self.sends:
            core_idx = event.get_int("coreIdx")
            core_type = event.get_int("coreType")
            if core_idx is not None:
                result.add(f"{CORE_TYPE_NAMES.get(core_type, f'TYPE{core_type}')}[{core_idx}]")
        for event in self.acks + self.finishes:
            core_idx = event.get_int("coreIdx")
            if core_idx is not None:
                result.add(f"CORE[{core_idx}]")
        return sorted(result)

    @property
    def leaf_hash(self) -> Optional[int]:
        for event in self.sends:
            value = event.get_int("leafHash")
            if value is not None:
                return value
        return None

    def issues(self) -> List[str]:
        issues = []
        if not self.resolved:
            issues.append("缺少 resolve")
        if not self.sends:
            issues.append("缺少 send")
        if not self.finishes:
            issues.append("缺少 finish")
        if len(self.sends) > 1:
            issues.append(f"send 重复 {len(self.sends)} 次")
        if len(self.finishes) > 1:
            issues.append(f"finish 重复 {len(self.finishes)} 次")
        if len(self.runtime_resolves) > 1:
            issues.append(f"运行时 resolve 重复 {len(self.runtime_resolves)} 次")
        if len(self.acks) > 1:
            issues.append(f"ack 重复 {len(self.acks)} 次")

        send_cores = {
            event.get_int("coreIdx") for event in self.sends if event.get_int("coreIdx") is not None
        }
        ack_cores = {
            event.get_int("coreIdx") for event in self.acks if event.get_int("coreIdx") is not None
        }
        finish_cores = {
            event.get_int("coreIdx") for event in self.finishes if event.get_int("coreIdx") is not None
        }
        all_cores = send_cores | ack_cores | finish_cores
        if len(all_cores) > 1:
            issues.append(f"send/ack/finish 核不一致: {sorted(all_cores)}")

        first_resolve = self._first_time(self.first_batch_resolves + self.runtime_resolves)
        first_send = self._first_time(self.sends)
        first_ack = self._first_time(self.acks)
        first_finish = self._first_time(self.finishes)
        if first_resolve is not None and first_send is not None and first_send < first_resolve:
            issues.append("send 早于 resolve")
        if first_ack is not None and first_send is not None and first_ack < first_send:
            issues.append("ack 早于 send")
        if first_finish is not None and first_send is not None and first_finish < first_send:
            issues.append("finish 早于 send")
        return issues

    @staticmethod
    def _first_time(events: Iterable[TraceEvent]) -> Optional[datetime]:
        values = [event.timestamp for event in events if event.timestamp is not None]
        return min(values) if values else None

    def stage_text(self) -> str:
        ack_text = "✓" if self.acks else "-"
        return (
            f"resolve={'✓' if self.resolved else '✗'} "
            f"send={'✓' if self.sends else '✗'} "
            f"ack={ack_text} "
            f"finish={'✓' if self.finishes else '✗'}"
        )

    def references(self) -> str:
        parts = []
        stage_events = [
            ("resolve", self.first_batch_resolves + self.runtime_resolves),
            ("send", self.sends),
            ("ack", self.acks),
            ("finish", self.finishes),
        ]
        for stage, events in stage_events:
            if events:
                parts.append(f"{stage}={events[0].reference}")
        return ", ".join(parts) if parts else "-"


@dataclass
class MixWrapRecord:
    round_id: int
    dev_task_id: int
    wrap_id: int
    mix_type: int = 0
    direct_send: Optional[bool] = None
    tasks: Dict[int, int] = field(default_factory=dict)
    cores: Dict[int, int] = field(default_factory=dict)
    resolved_roles: Set[int] = field(default_factory=set)
    sent_roles: Set[int] = field(default_factory=set)
    finished_roles: Set[int] = field(default_factory=set)
    release_events: List[TraceEvent] = field(default_factory=list)
    events: List[TraceEvent] = field(default_factory=list)

    @property
    def all_finish(self) -> bool:
        return bool(self.expected_roles) and self.expected_roles <= self.finished_roles

    @property
    def expected_roles(self) -> Set[int]:
        if self.mix_type == 1:
            return {0, 1}
        if self.mix_type == 2:
            return {0, 1, 2}
        return set(self.tasks)

    def update_task(self, role: int, task_id: Optional[int]) -> None:
        if task_id is not None and task_id != 0:
            self.tasks[role] = task_id

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)
        mix_type = event.get_int("mixType")
        if mix_type is not None:
            self.mix_type = mix_type

        if event.tag == "mix.resolve":
            self.direct_send = event.get_int("directSend", 0) == 1
            self.update_task(0, event.get_int("aicTaskId"))
            self.update_task(1, event.get_int("aiv0TaskId"))
            self.update_task(2, event.get_int("aiv1TaskId"))
            self.resolved_roles.update(self.expected_roles)
        elif event.tag == "mix.wrapcreate":
            self.update_task(0, event.get_int("taskAic"))
            self.update_task(1, event.get_int("taskAiv0"))
            self.update_task(2, event.get_int("taskAiv1"))
        elif event.tag == "mix.wrapresolve":
            role = event.get_int("wrapAicoreIdx")
            if role is not None:
                self.update_task(role, event.get_int("taskId"))
                self.resolved_roles.add(role)
            if self.direct_send is None:
                self.direct_send = False
        elif event.tag == "mix.send":
            self.update_task(0, event.get_int("taskAic"))
            self.update_task(1, event.get_int("taskAiv0"))
            self.update_task(2, event.get_int("taskAiv1"))
            core_fields = {0: "aicIdx", 1: "aivIdx0", 2: "aivIdx1"}
            for role in self.expected_roles:
                core = event.get_int(core_fields[role])
                if core is not None:
                    self.cores[role] = core
                self.sent_roles.add(role)
        elif event.tag == "mix.sendsingle":
            role = event.get_int("wrapAicoreIdx")
            if role is not None:
                self.update_task(role, event.get_int("taskId"))
                core = event.get_int("core")
                if core is not None:
                    self.cores[role] = core
                self.sent_roles.add(role)
        elif event.tag == "mix.wrapfinish":
            task_id = event.get_int("finishTaskId")
            self._mark_finished_by_task_or_core(task_id, event.get_int("core"))
        elif event.tag == "mix.release":
            self.release_events.append(event)
            self._mark_finished_by_task_or_core(None, event.get_int("core"))

    def _mark_finished_by_task_or_core(
        self, task_id: Optional[int], core_idx: Optional[int]
    ) -> None:
        for role, known_task in self.tasks.items():
            if task_id is not None and known_task == task_id:
                self.finished_roles.add(role)
            elif core_idx is not None and self.cores.get(role) == core_idx:
                self.finished_roles.add(role)

    def mark_coretask_finish(self, task_id: int) -> None:
        self._mark_finished_by_task_or_core(task_id, None)

    @property
    def complete(self) -> bool:
        expected = self.expected_roles
        if not expected:
            return False
        return expected <= self.finished_roles

    def issues(self) -> List[str]:
        issues = []
        expected = self.expected_roles
        if not expected:
            issues.append("mixType 未知")
            return issues
        missing_tasks = sorted(expected - set(self.tasks))
        if missing_tasks:
            issues.append(
                "缺少角色任务: " + ", ".join(MIX_ROLE_NAMES[role] for role in missing_tasks)
            )
        missing_sends = sorted(expected - self.sent_roles)
        if missing_sends:
            issues.append(
                "未下发角色: " + ", ".join(MIX_ROLE_NAMES[role] for role in missing_sends)
            )
        missing_finishes = sorted(expected - self.finished_roles)
        if missing_finishes:
            issues.append(
                "未完成角色: " + ", ".join(MIX_ROLE_NAMES[role] for role in missing_finishes)
            )
        return issues


@dataclass
class AnalysisResult:
    events: List[TraceEvent]
    source_files: List[Path]
    rounds: Dict[int, Dict[str, List[TraceEvent]]]
    arbitrations: List[TraceEvent]
    handshakes: List[TraceEvent]
    device_tasks: Dict[Tuple[int, int], DeviceTaskRecord]
    core_tasks: Dict[Tuple[int, int, int], CoreTaskLifecycle]
    mix_wraps: Dict[Tuple[int, int, int], MixWrapRecord]
    aicore_statuses: List[TraceEvent]
    queue_snapshots: List[TraceEvent]


def discover_log_files(log_dir: Path) -> List[Path]:
    output_path = (log_dir / REPORT_NAME).resolve()
    result = []
    for path in log_dir.rglob("*"):
        if not path.is_file() or path.resolve() == output_path:
            continue
        if path.suffix.lower() in LOG_SUFFIXES:
            result.append(path)
    return sorted(result)


def parse_log_files(paths: Iterable[Path]) -> Tuple[List[TraceEvent], List[Path]]:
    events = []
    source_files = []
    input_order = 0
    for path in paths:
        file_has_trace = False
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                tag_match = TAG_PATTERN.search(line)
                if tag_match is None:
                    continue
                file_has_trace = True
                fields = {match.group(1): match.group(2) for match in FIELD_PATTERN.finditer(line)}
                tag = tag_match.group(1)
                if tag == "dtask.built" and "dtaskId" not in fields:
                    built_match = DEVTASK_BUILT_PATTERN.search(line)
                    if built_match is not None:
                        fields["dtaskId"] = built_match.group(1)
                events.append(
                    TraceEvent(
                        tag=tag,
                        fields=fields,
                        timestamp=parse_timestamp(line),
                        source=path,
                        line_number=line_number,
                        input_order=input_order,
                    )
                )
                input_order += 1
        if file_has_trace:
            source_files.append(path)

    events.sort(
        key=lambda event: (
            event.timestamp is None,
            event.timestamp or datetime.max,
            event.input_order,
        )
    )
    current_round = 0
    for sequence, event in enumerate(events, 1):
        event.sequence = sequence
        if event.tag == "round.start":
            current_round = event.get_int("round", current_round) or current_round
        event.round_id = current_round
    return events, source_files


def build_analysis(events: List[TraceEvent], source_files: List[Path]) -> AnalysisResult:
    rounds: Dict[int, Dict[str, List[TraceEvent]]] = defaultdict(lambda: defaultdict(list))
    arbitrations = []
    handshakes = []
    device_tasks: Dict[Tuple[int, int], DeviceTaskRecord] = {}
    core_tasks: Dict[Tuple[int, int, int], CoreTaskLifecycle] = {}
    mix_events = []
    aicore_statuses = []
    queue_snapshots = []

    for event in events:
        rounds[event.round_id][event.tag].append(event)
        if event.tag == "arbitration":
            arbitrations.append(event)
        elif event.tag == "handshake":
            handshakes.append(event)
        elif event.tag.startswith("dtask."):
            field_name = "dtaskId" if event.tag == "dtask.built" else "taskId"
            task_id = event.get_int(field_name)
            if task_id is None:
                continue
            key = (event.round_id, task_id)
            record = device_tasks.setdefault(key, DeviceTaskRecord(event.round_id, task_id))
            record.add(event)
        elif event.tag.startswith("ltask."):
            dev_task_id = event.get_int("dtaskId")
            task_id = event.get_int("task")
            if dev_task_id is None or task_id is None:
                continue
            key = (event.round_id, dev_task_id, task_id)
            record = core_tasks.setdefault(
                key, CoreTaskLifecycle(event.round_id, dev_task_id, task_id)
            )
            record.add(event)
        elif event.tag.startswith("mix."):
            mix_events.append(event)
        elif event.tag == "aicore.status":
            aicore_statuses.append(event)
        elif event.tag == "queue":
            queue_snapshots.append(event)

    mix_wraps: Dict[Tuple[int, int, int], MixWrapRecord] = {}
    for event in mix_events:
        wrap_id = event.get_int("wrapId")
        if wrap_id is None:
            continue
        dev_task_id = find_serial_device_task(event, device_tasks)
        if dev_task_id is None:
            continue
        key = (event.round_id, dev_task_id, wrap_id)
        record = mix_wraps.setdefault(
            key, MixWrapRecord(event.round_id, dev_task_id, wrap_id)
        )
        record.add(event)

    mix_task_keys = {}
    for record in mix_wraps.values():
        for task_id in record.tasks.values():
            key = (record.round_id, record.dev_task_id, task_id)
            mix_task_keys[key] = record
            lifecycle = core_tasks.get(key)
            if lifecycle is not None:
                lifecycle.mix_resolved = True

    for key, lifecycle in core_tasks.items():
        mix_record = mix_task_keys.get(key)
        if mix_record is not None and lifecycle.finishes:
            mix_record.mark_coretask_finish(lifecycle.task_id)

    return AnalysisResult(
        events=events,
        source_files=source_files,
        rounds=dict(rounds),
        arbitrations=arbitrations,
        handshakes=handshakes,
        device_tasks=device_tasks,
        core_tasks=core_tasks,
        mix_wraps=mix_wraps,
        aicore_statuses=aicore_statuses,
        queue_snapshots=queue_snapshots,
    )


def find_serial_device_task(
    event: TraceEvent, device_tasks: Dict[Tuple[int, int], DeviceTaskRecord]
) -> Optional[int]:
    candidates = []
    for (round_id, task_id), record in device_tasks.items():
        if round_id != event.round_id or record.start_sequence is None:
            continue
        end_sequence = record.end_sequence or sys.maxsize
        if record.start_sequence <= event.sequence <= end_sequence:
            candidates.append(task_id)
    if len(candidates) == 1:
        return candidates[0]
    round_tasks = [
        task_id for (round_id, task_id) in device_tasks if round_id == event.round_id
    ]
    if len(round_tasks) == 1:
        return round_tasks[0]
    return None


def handshake_summary(
    arbitrations: List[TraceEvent], handshakes: List[TraceEvent]
) -> Dict[int, Tuple[int, int]]:
    expected = {0: set(), 1: set()}
    succeeded = {0: set(), 1: set()}
    for event in arbitrations:
        aic_start = event.get_int("aicStart")
        aic_end = event.get_int("aicEnd")
        aiv_start = event.get_int("aivStart")
        aiv_end = event.get_int("aivEnd")
        if aic_start is not None and aic_end is not None:
            expected[1].update(range(aic_start, aic_end))
        if aiv_start is not None and aiv_end is not None:
            expected[0].update(range(aiv_start, aiv_end))
    for event in handshakes:
        if event.get_int("success", 0) != 1:
            continue
        core_type = event.get_int("type")
        core = event.get_int("core")
        if core_type in succeeded and core is not None:
            succeeded[core_type].add(core)
    return {
        core_type: (len(succeeded[core_type]), len(expected[core_type]))
        for core_type in (1, 0)
    }


def device_task_core_records(
    result: AnalysisResult, round_id: int, task_id: int
) -> List[CoreTaskLifecycle]:
    records = [
        record
        for (core_round, dev_task_id, _), record in result.core_tasks.items()
        if core_round == round_id and dev_task_id == task_id
    ]
    return sorted(records, key=lambda record: record.task_id)


def queued_core_records(records: Iterable[CoreTaskLifecycle]) -> List[CoreTaskLifecycle]:
    """Return tasks logged by PushReadyTask (firstBatch=0)."""
    return [record for record in records if record.runtime_resolves]


def first_batch_core_records(records: Iterable[CoreTaskLifecycle]) -> List[CoreTaskLifecycle]:
    """Return tasks logged by firstBatch resolve (firstBatch=1)."""
    return [record for record in records if record.first_batch_resolves]


def mix_core_records(records: Iterable[CoreTaskLifecycle]) -> List[CoreTaskLifecycle]:
    """Return tasks logged by mix resolve (mix_resolved)."""
    return [record for record in records if record.mix_resolved]


def executed_device_task_items(
    result: AnalysisResult,
) -> List[Tuple[Tuple[int, int], DeviceTaskRecord]]:
    return [
        (key, record)
        for key, record in sorted(result.device_tasks.items())
        if record.has_execution_events or device_task_core_records(result, *key)
    ]


def built_device_task_count(result: AnalysisResult) -> int:
    return sum(len(record.built_events) for record in result.device_tasks.values())


def control_core_ranges(
    result: AnalysisResult, round_id: int, prefer_snapshot: bool = False
) -> List[Tuple[int, int, int, int, int, Optional[int]]]:
    tags = result.rounds.get(round_id, {})
    source_events = tags.get("ctrlcore", []) if prefer_snapshot else []
    if not source_events:
        source_events = tags.get("arbitration", [])

    latest_by_scheduler = {}
    for event in source_events:
        scheduler = event.get_int("schedIdx")
        if scheduler is not None:
            latest_by_scheduler[scheduler] = event

    ranges = []
    for scheduler, event in sorted(latest_by_scheduler.items()):
        aic_start = event.get_int("aicStart")
        aic_end = event.get_int("aicEnd")
        aiv_start = event.get_int("aivStart")
        aiv_end = event.get_int("aivEnd")
        if None in (aic_start, aic_end, aiv_start, aiv_end):
            continue
        ranges.append(
            (
                scheduler,
                aic_start,
                aic_end,
                aiv_start,
                aiv_end,
                event.get_int("ctrlCoreDisabled"),
            )
        )
    return ranges


def collect_overall_issues(result: AnalysisResult) -> List[str]:
    issues = []
    for round_id, tags in result.rounds.items():
        error_codes = sorted(
            {
                event.get_int("ret")
                for event in tags.get("round.end", [])
                if event.get_int("ret") not in (None, 0)
            }
        )
        for error_code in error_codes:
            issues.append(f"Round {round_id} 返回错误 ret={error_code}")
    for key, device_task in executed_device_task_items(result):
        for issue in device_task.structural_issues():
            issues.append(f"DeviceTask round={key[0]} taskId={key[1]}: {issue}")
        core_records = device_task_core_records(result, *key)
        abnormal_count = sum(
            bool(record.issues())
            for record in queued_core_records(core_records)
            + first_batch_core_records(core_records)
            + mix_core_records(core_records)
        )
        if abnormal_count:
            issues.append(f"DeviceTask round={key[0]} taskId={key[1]}: {abnormal_count} 个异常 CoreTask")
    return issues


@dataclass
class RuleResult:
    name: str
    description: str
    passed: bool
    details: str = ""


class Rule:
    name: str = ""
    description: str = ""

    def check(self, result: AnalysisResult) -> RuleResult:
        raise NotImplementedError


class SchedulerCoreRangeRule(Rule):
    name = "调度核范围校验"
    description = "线程下发的 coreIdx 必须在自身仲裁管理的核范围内"

    def check(self, result: AnalysisResult) -> RuleResult:
        ranges: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        for event in result.arbitrations:
            sched = event.get_int("schedIdx")
            if sched is None:
                continue
            aic_start = event.get_int("aicStart")
            aic_end = event.get_int("aicEnd")
            aiv_start = event.get_int("aivStart")
            aiv_end = event.get_int("aivEnd")
            if None in (aic_start, aic_end, aiv_start, aiv_end):
                continue
            ranges[(event.round_id, sched)] = (aic_start, aic_end, aiv_start, aiv_end)

        violations: List[str] = []
        checked = 0
        for lifecycle in result.core_tasks.values():
            for send in lifecycle.sends:
                tid = send.get_int("tid")
                core_idx = send.get_int("coreIdx")
                core_type = send.get_int("coreType")
                if tid is None or core_idx is None or core_type is None:
                    continue
                key = (send.round_id, tid)
                if key not in ranges:
                    continue
                aic_start, aic_end, aiv_start, aiv_end = ranges[key]
                checked += 1
                if core_type == 1:
                    if not (aic_start <= core_idx < aic_end):
                        violations.append(
                            f"tid={tid} 向 AIC[{core_idx}] 下发任务，"
                            f"但管理范围为 [{aic_start}, {aic_end})"
                        )
                elif core_type == 0:
                    if not (aiv_start <= core_idx < aiv_end):
                        violations.append(
                            f"tid={tid} 向 AIV[{core_idx}] 下发任务，"
                            f"但管理范围为 [{aiv_start}, {aiv_end})"
                        )

        if violations:
            shown = "; ".join(violations[:5])
            suffix = f" 等 {len(violations)} 条" if len(violations) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"校验 {checked} 次，违规 {len(violations)} 次: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class ThreadIdxUniqueRule(Rule):
    name = "threadIdx 唯一性校验"
    description = "所有线程的 threadIdx 不能有重复"

    def check(self, result: AnalysisResult) -> RuleResult:
        tid_to_thread_idx: Dict[int, int] = {}
        for event in result.arbitrations:
            tid = event.get_int("tid")
            thread_idx = event.get_int("threadIdx")
            if tid is None or thread_idx is None:
                continue
            tid_to_thread_idx[tid] = thread_idx

        if not tid_to_thread_idx:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="无 arbitration 事件，跳过",
            )

        idx_to_tids: Dict[int, List[int]] = defaultdict(list)
        for tid, thread_idx in tid_to_thread_idx.items():
            idx_to_tids[thread_idx].append(tid)

        duplicates = {idx: tids for idx, tids in idx_to_tids.items() if len(tids) > 1}
        if duplicates:
            parts = [
                f"threadIdx={idx} 被 tid={tids} 共用"
                for idx, tids in sorted(duplicates.items())
            ]
            shown = "; ".join(parts[:5])
            suffix = f" 等 {len(duplicates)} 个" if len(duplicates) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"共 {len(tid_to_thread_idx)} 个线程，重复 threadIdx {len(duplicates)} 个: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class ArbitrationNumConsistentRule(Rule):
    name = "arbitrationNum 一致性校验"
    description = "所有线程的 arbitratedScheNum 必须相同"

    def check(self, result: AnalysisResult) -> RuleResult:
        tid_to_num: Dict[int, int] = {}
        for event in result.arbitrations:
            tid = event.get_int("tid")
            num = event.get_int("arbitratedScheNum")
            if tid is None or num is None:
                continue
            tid_to_num[tid] = num

        if not tid_to_num:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="无 arbitration 事件，跳过",
            )

        unique_nums = set(tid_to_num.values())
        if len(unique_nums) > 1:
            detail_map: Dict[int, List[int]] = defaultdict(list)
            for tid, num in tid_to_num.items():
                detail_map[num].append(tid)
            parts = [
                f"arbitratedScheNum={num} (tid={tids})"
                for num, tids in sorted(detail_map.items())
            ]
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"共 {len(tid_to_num)} 个线程，存在 {len(unique_nums)} 种值: {'; '.join(parts)}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class CoreIdRangeRule(Rule):
    name = "coreId 范围校验"
    description = "所有 coreId 应该在 0 - 108 之间"

    def check(self, result: AnalysisResult) -> RuleResult:
        min_core_id = 0
        max_core_id = 108
        seen: Set[int] = set()
        for lifecycle in result.core_tasks.values():
            for event in lifecycle.sends + lifecycle.acks + lifecycle.finishes:
                core_idx = event.get_int("coreIdx")
                if core_idx is not None:
                    seen.add(core_idx)
        for event in result.aicore_statuses:
            core = event.get_int("core")
            if core is not None:
                seen.add(core)

        if not seen:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="未发现 coreId，跳过",
            )
        out_of_range = sorted(c for c in seen if not (min_core_id <= c <= max_core_id))
        if out_of_range:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"共 {len(seen)} 个 coreId，越界 {len(out_of_range)} 个: {out_of_range[:10]}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class CoreTaskDuplicateRule(Rule):
    name = "CoreTask 重复事件校验"
    description = "同一 CoreTask 的 send/finish/ack/runtime resolve 不应重复出现"

    def check(self, result: AnalysisResult) -> RuleResult:
        total = 0
        violations: List[str] = []
        for lifecycle in result.core_tasks.values():
            if not lifecycle.resolved:
                continue
            total += 1
            parts = []
            if len(lifecycle.sends) > 1:
                parts.append(f"send×{len(lifecycle.sends)}")
            if len(lifecycle.finishes) > 1:
                parts.append(f"finish×{len(lifecycle.finishes)}")
            if len(lifecycle.acks) > 1:
                parts.append(f"ack×{len(lifecycle.acks)}")
            if len(lifecycle.runtime_resolves) > 1:
                parts.append(f"resolve×{len(lifecycle.runtime_resolves)}")
            if parts:
                violations.append(
                    f"task=0x{lifecycle.task_id:X} ({', '.join(parts)})"
                )

        if not total:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="无已 resolve 的 CoreTask，跳过",
            )
        if violations:
            shown = "; ".join(violations[:5])
            suffix = f" 等 {len(violations)} 个" if len(violations) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"校验 {total} 个 CoreTask，重复 {len(violations)} 个: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class CoreReadyCntRangeRule(Rule):
    name = "Run/PendReadyCnt 计数校验 1"
    description = "runReady/pendReady 计数不应超过管理的核总数"

    def check(self, result: AnalysisResult) -> RuleResult:
        checked = 0
        violations: List[str] = []
        for round_id, tags in result.rounds.items():
            for event in tags.get("ctrlcore", []):
                aic_start = event.get_int("aicStart")
                aic_end = event.get_int("aicEnd")
                aiv_start = event.get_int("aivStart")
                aiv_end = event.get_int("aivEnd")
                if None in (aic_start, aic_end, aiv_start, aiv_end):
                    continue
                aic_num = aic_end - aic_start
                aiv_num = aiv_end - aiv_start
                run_ready_aic = event.get_int("runReadyAic", 0) or 0
                run_ready_aiv = event.get_int("runReadyAiv", 0) or 0
                pend_ready_aic = event.get_int("pendReadyAic", 0) or 0
                pend_ready_aiv = event.get_int("pendReadyAiv", 0) or 0
                checked += 1
                if run_ready_aic > aic_num:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"runReadyAic={run_ready_aic} > AIC 核数 {aic_num}"
                    )
                if run_ready_aiv > aiv_num:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"runReadyAiv={run_ready_aiv} > AIV 核数 {aiv_num}"
                    )
                if pend_ready_aic > aic_num:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"pendReadyAic={pend_ready_aic} > AIC 核数 {aic_num}"
                    )
                if pend_ready_aiv > aiv_num:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"pendReadyAiv={pend_ready_aiv} > AIV 核数 {aiv_num}"
                    )

        if not checked:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="仅异常时检测",
            )
        if violations:
            shown = "; ".join(violations[:5])
            suffix = f" 等 {len(violations)} 条" if len(violations) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"校验 {checked} 次快照，违规 {len(violations)} 条: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class CoreReadyCntBalanceRule(Rule):
    name = "Run/PendReadyCnt 计数校验 2"
    description = "pendReady 计数应不小于 runReady 计数"

    def check(self, result: AnalysisResult) -> RuleResult:
        checked = 0
        violations: List[str] = []
        for round_id, tags in result.rounds.items():
            for event in tags.get("ctrlcore", []):
                run_ready_aic = event.get_int("runReadyAic")
                pend_ready_aic = event.get_int("pendReadyAic")
                run_ready_aiv = event.get_int("runReadyAiv")
                pend_ready_aiv = event.get_int("pendReadyAiv")
                if None in (run_ready_aic, pend_ready_aic, run_ready_aiv, pend_ready_aiv):
                    continue
                checked += 1
                if pend_ready_aic < run_ready_aic:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"pendReadyAic={pend_ready_aic} < runReadyAic={run_ready_aic}"
                    )
                if pend_ready_aiv < run_ready_aiv:
                    violations.append(
                        f"round={round_id} schedIdx={event.get_int('schedIdx')} "
                        f"pendReadyAiv={pend_ready_aiv} < runReadyAiv={run_ready_aiv}"
                    )

        if not checked:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="仅异常时检测",
            )
        if violations:
            shown = "; ".join(violations[:5])
            suffix = f" 等 {len(violations)} 条" if len(violations) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"校验 {checked} 次快照，违规 {len(violations)} 条: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


class CoreTaskPendingRunningRule(Rule):
    name = "核任务运行态一致性校验"
    description = "runningId/pendingId 应与 send 记录匹配，且两者不能同时为非 INIT"

    TASK_BITS = 16
    FUNC_BITS = 10
    TASK_FUNC_MASK = (1 << (TASK_BITS + FUNC_BITS)) - 1

    def _task_func_id(self, task_id: Optional[int]) -> Optional[int]:
        if task_id is None or task_id == TASK_INIT:
            return None
        return task_id & self.TASK_FUNC_MASK

    def check(self, result: AnalysisResult) -> RuleResult:
        if not result.aicore_statuses:
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=True,
                details="仅异常时检测",
            )

        core_sent_tasks: Dict[int, Set[int]] = {}
        for lifecycle in result.core_tasks.values():
            for send in lifecycle.sends:
                core_idx = send.get_int("coreIdx")
                task_id = send.get_int("task")
                if core_idx is not None and task_id is not None:
                    core_sent_tasks.setdefault(core_idx, set()).add(self._task_func_id(task_id))

        checked = 0
        violations: List[str] = []
        for event in result.aicore_statuses:
            core = event.get_int("core")
            if core is None:
                continue
            running = event.get_int("runningId", TASK_INIT)
            pending = event.get_int("pendingId", TASK_INIT)
            checked += 1
            running_active = running is not None and running != TASK_INIT
            pending_active = pending is not None and pending != TASK_INIT

            if running_active and pending_active:
                running_id = self._task_func_id(running)
                pending_id = self._task_func_id(pending)
                if running_id == pending_id:
                    violations.append(
                        f"core={core} running 和 pending 同时有效且 task+func 相同 "
                        f"(running=0x{running:X} pending=0x{pending:X})"
                    )
            if running_active:
                running_id = self._task_func_id(running)
                sent_tasks = core_sent_tasks.get(core, set())
                if running_id is not None and sent_tasks and running_id not in sent_tasks:
                    violations.append(
                        f"core={core} running taskFunc={running_id} (0x{running:X}) "
                        f"不在 send 记录中 {sorted(sent_tasks)[:5]}"
                    )
            if pending_active:
                pending_id = self._task_func_id(pending)
                sent_tasks = core_sent_tasks.get(core, set())
                if pending_id is not None and sent_tasks and pending_id not in sent_tasks:
                    violations.append(
                        f"core={core} pending taskFunc={pending_id} (0x{pending:X}) "
                        f"不在 send 记录中 {sorted(sent_tasks)[:5]}"
                    )

        if violations:
            shown = "; ".join(violations[:5])
            suffix = f" 等 {len(violations)} 条" if len(violations) > 5 else ""
            return RuleResult(
                name=self.name,
                description=self.description,
                passed=False,
                details=f"校验 {checked} 个核状态，违规 {len(violations)} 条: {shown}{suffix}",
            )
        return RuleResult(
            name=self.name,
            description=self.description,
            passed=True,
            details="通过",
        )


RULES: List[Rule] = [
    SchedulerCoreRangeRule(),
    ThreadIdxUniqueRule(),
    ArbitrationNumConsistentRule(),
    CoreIdRangeRule(),
    CoreTaskDuplicateRule(),
    CoreReadyCntRangeRule(),
    CoreReadyCntBalanceRule(),
    CoreTaskPendingRunningRule(),
]


def run_rules(result: AnalysisResult) -> List[RuleResult]:
    return [rule.check(result) for rule in RULES]


def build_abnormal_summary(
    result: AnalysisResult,
    execution_items: List[Tuple[Tuple[int, int], DeviceTaskRecord]],
) -> List[str]:
    lines: List[str] = ["  异常总结:"]
    lines.append(
        f"    构建/执行 DeviceTask: {built_device_task_count(result)}/{len(execution_items)}"
    )

    abnormal_devtask_infos = []
    for idx, (key, device_task) in enumerate(execution_items, 1):
        core_records = device_task_core_records(result, *key)
        has_structural = bool(device_task.structural_issues())
        has_abnormal_core = any(
            record.issues() for record in queued_core_records(core_records) + first_batch_core_records(core_records)
        )
        if has_structural or has_abnormal_core:
            expected = device_task.expected_core_count or len(core_records)
            unfinished = sum(1 for record in core_records if not record.finishes)
            abnormal_devtask_infos.append((idx, key, expected, unfinished))
    if abnormal_devtask_infos:
        parts = [
            f"第 {idx} 个 (round={key[0]}, taskId={key[1]}), CoreTask 总数={total}, 未完成={unfinished}"
            for idx, key, total, unfinished in abnormal_devtask_infos
        ]
        lines.append(f"    异常 DeviceTask: {'; '.join(parts)}")

    latest_status_by_core: Dict[int, TraceEvent] = {}
    stats_by_sched: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for event in result.aicore_statuses:
        core = event.get_int("core")
        if core is not None:
            latest_status_by_core[core] = event
    if latest_status_by_core:
        all_ctrlcores = []
        for round_tags in result.rounds.values():
            all_ctrlcores.extend(round_tags.get("ctrlcore", []))
        stats_by_sched: Dict[int, Dict[str, Tuple[int, int]]] = {}
        for event in result.aicore_statuses:
            core = event.get_int("core")
            if core is None:
                continue
            running = event.get_int("runningId", TASK_INIT)
            pending = event.get_int("pendingId", TASK_INIT)
            running_active = running is not None and running != TASK_INIT
            pending_active = pending is not None and pending != TASK_INIT
            is_pending = pending_active and not running_active
            is_running = running_active
            core_type = event.get_int("type")
            for sched_event in all_ctrlcores:
                sched = sched_event.get_int("schedIdx")
                if sched is None:
                    continue
                aic_start = sched_event.get_int("aicStart")
                aic_end = sched_event.get_int("aicEnd")
                aiv_start = sched_event.get_int("aivStart")
                aiv_end = sched_event.get_int("aivEnd")
                if None in (aic_start, aic_end, aiv_start, aiv_end):
                    continue
                in_range = (core_type == 1 and aic_start <= core < aic_end) or \
                           (core_type == 0 and aiv_start <= core < aiv_end)
                if not in_range:
                    continue
                stats = stats_by_sched.setdefault(sched, {"idle": (0, 0), "pending": (0, 0), "running": (0, 0)})
                type_idx = 0 if core_type == 1 else 1
                state = "running" if is_running else ("pending" if is_pending else "idle")
                a, b = stats[state]
                stats[state] = (a + 1, b) if type_idx == 0 else (a, b + 1)
                break
        lines.append("    Core Status:")
        for sched in sorted(stats_by_sched):
            stats = stats_by_sched[sched]
            idle_aic, idle_aiv = stats["idle"]
            pend_aic, pend_aiv = stats["pending"]
            run_aic, run_aiv = stats["running"]
            lines.append(
                f"      Scheduler {sched}: "
                f"Idle Core: AIC={idle_aic}, AIV={idle_aiv}  "
                f"Pending Core: AIC={pend_aic}, AIV={pend_aiv}  "
                f"Running Core: AIC={run_aic}, AIV={run_aiv}"
            )

    unsent_count = sum(
        1
        for record in result.core_tasks.values()
        if (record.first_batch_resolves or record.runtime_resolves) and not record.sends
    )
    if result.queue_snapshots:
        queue_by_sched: Dict[int, Dict[str, int]] = {}
        for event in result.queue_snapshots:
            sched = event.get_int("schedIdx", 0)
            name = event.fields.get("name", "-")
            size = event.get_int("size", 0) or 0
            queue_by_sched.setdefault(sched, {})[name] = queue_by_sched.setdefault(sched, {}).get(name, 0) + size
        for sched in sorted(queue_by_sched):
            q = queue_by_sched[sched]
            lines.append(
                f"    Scheduler {sched}: AIC QUE SIZE: {q.get('readyAicCoreFunctionQue', 0)}, "
                f"AIV QUE SIZE: {q.get('readyAivCoreFunctionQue', 0)}, "
                f"MIX QUE SIZE: {q.get('readyWrapCoreFunctionQue', 0)}, "
                f"未下发task={unsent_count}"
            )
    else:
        lines.append(f"    队列中未下发 task 数: {unsent_count}")

    if result.mix_wraps:
        total_mix = len(result.mix_wraps)
        completed_mix = sum(1 for r in result.mix_wraps.values() if r.complete)
        unsent_mix = sum(1 for r in result.mix_wraps.values() if not r.sent_roles and r.resolved_roles)
        unfinished_mix = sum(1 for r in result.mix_wraps.values() if r.sent_roles and not r.complete)
        lines.append(
            f"    MIX wrap: 总数={total_mix}, 完成={completed_mix}, "
            f"未下发={unsent_mix}, 未完成={unfinished_mix}"
        )

    incomplete_coretask = sum(
        1
        for record in result.core_tasks.values()
        if record.resolved and record.issues()
    )
    if incomplete_coretask:
        lines.append(f"    CoreTask 生命周期不完整: {incomplete_coretask}")

    if stats_by_sched:
        total_idle = sum(s["idle"][0] + s["idle"][1] for s in stats_by_sched.values())
        total_active = sum(
            s["pending"][0] + s["pending"][1] + s["running"][0] + s["running"][1]
            for s in stats_by_sched.values()
        )
        total_que = 0
        if result.queue_snapshots:
            que_names = (
                "readyAicCoreFunctionQue", "readyAivCoreFunctionQue", "readyWrapCoreFunctionQue",
            )
            total_que = sum(
                (event.get_int("size", 0) or 0)
                for event in result.queue_snapshots
                if event.fields.get("name") in que_names
            )
        if total_que > 0 and total_idle == 0 and total_active == 0:
            lines.append("    总结: 队列有任务但无空闲核可下发")
        elif total_que > 0 and total_idle > 0:
            lines.append("    总结: 队列有任务且有空闲核可下发")
        elif total_que == 0 and total_idle == 0 and total_active > 0:
            lines.append("    总结: 无任务可下发且无空闲核，仅有核在执行中")
        elif total_que == 0 and total_idle > 0 and total_active > 0:
            lines.append("    总结: 无任务可下发，有空闲核且有核在执行中")
        elif total_que == 0 and total_idle == 0 and total_active == 0 and incomplete_coretask > 0:
            lines.append("    总结: 无任务可下发且无核在执行，CoreTask 生命周期不完整")
        elif total_que == 0 and total_idle > 0 and total_active == 0 and incomplete_coretask > 0:
            lines.append("    总结: 无任务可下发且有空闲核，CoreTask 生命周期不完整")
        elif total_que == 0 and total_idle > 0 and total_active == 0 and incomplete_coretask == 0:
            lines.append("    总结: 无任务可下发，有空闲核，无异常 CoreTask")

    return lines


def render_report(result: AnalysisResult, log_dir: Path) -> str:
    lines: List[str] = []
    overall_issues = collect_overall_issues(result)
    execution_items = executed_device_task_items(result)
    queued_tasks = queued_core_records(result.core_tasks.values())
    first_batch_tasks = first_batch_core_records(result.core_tasks.values())
    mix_tasks = mix_core_records(result.core_tasks.values())
    abnormal_core_tasks = [
        record for record in queued_tasks + first_batch_tasks + mix_tasks if record.issues()
    ]
    abnormal_mix_wraps = [
        record for record in result.mix_wraps.values() if record.issues()
    ]

    built_count = built_device_task_count(result)

    lines.extend(
        [
            "PyPTO Trace 任务生命周期报告",
            "=" * 88,
            f"日志目录      : {log_dir}",
            f"Trace 文件数  : {len(result.source_files)}",
            f"Trace 事件数  : {len(result.events)}",
            f"执行结果      : {'异常' if overall_issues else '正常'}",
            f"构建 DeviceTask: {built_count}",
            f"执行 DeviceTask: {len(execution_items)}",
            f"首批 CoreTask : {len(first_batch_tasks)}",
            f"入队 CoreTask : {len(queued_tasks) + len(mix_tasks)}",
            f"异常 CoreTask : {len(abnormal_core_tasks)}",
        ]
    )
    if result.mix_wraps:
        lines.append(
            f"MIX wrap      : {len(result.mix_wraps)}，异常 {len(abnormal_mix_wraps)}"
        )
    lines.append("")

    if result.source_files:
        lines.append("输入文件")
        lines.append("-" * 88)
        for path in result.source_files:
            lines.append(f"  {path}")
        lines.append("")

    round_ids = sorted(result.rounds)
    if round_ids:
        lines.append("1. Launch Round")
        lines.append("-" * 88)
        for round_id in round_ids:
            tags = result.rounds[round_id]
            starts = tags.get("round.start", [])
            ends = tags.get("round.end", [])
            ctrl_starts = [e for e in starts if e.get_int("tid") is None]
            ctrl_ends = [e for e in ends if e.get_int("tid") is None]
            sche_starts = [e for e in starts if e.get_int("tid") is not None]
            sche_ends = [e for e in ends if e.get_int("tid") is not None]
            sche_tids = sorted(
                {
                    event.get_int("tid")
                    for event in sche_starts + sche_ends
                    if event.get_int("tid") is not None
                }
            )
            lines.append(f"  Round {round_id}:")
            lines.append(
                f"    Ctrl  : start={len(ctrl_starts)}, end={len(ctrl_ends)}"
            )
            lines.append(
                f"    Sche  : start={len(sche_starts)}, end={len(sche_ends)}, tid={sche_tids or '-'}"
            )
        lines.append("")

    if result.handshakes:
        lines.append("2. 握手统计")
        lines.append("-" * 88)
        summary = handshake_summary(result.arbitrations, result.handshakes)
        aic_success = summary[1][0]
        aiv_success = summary[0][0]
        lines.append(
            f"  握手成功数: AIC={aic_success}, AIV={aiv_success}"
        )
        arbitration_ranges = {}
        for event in result.arbitrations:
            arbitration_ranges.setdefault(event.round_id, []).append(event)
        for round_id in sorted(arbitration_ranges):
            lines.append(f"  Round {round_id} 仲裁控核范围:")
            for scheduler, aic_start, aic_end, aiv_start, aiv_end, _ in control_core_ranges(
                result, round_id
            ):
                lines.append(
                    f"    Scheduler {scheduler}: "
                    f"AIC=[{aic_start}, {aic_end}) count={aic_end - aic_start}, "
                    f"AIV=[{aiv_start}, {aiv_end}) count={aiv_end - aiv_start}"
                )
        lines.append("")

    lines.append("3. DeviceTask 与 CoreTask 生命周期")
    lines.append("-" * 88)
    if not execution_items:
        lines.append("  未发现 DeviceTask Trace。")
    for key, device_task in execution_items:
        core_records = device_task_core_records(result, *key)
        task_queued_records = queued_core_records(core_records)
        task_first_batch_records = first_batch_core_records(core_records)
        task_mix_records = mix_core_records(core_records)
        abnormal_records = [
            record for record in task_queued_records + task_first_batch_records + task_mix_records
            if record.issues()
        ]
        sent_count = sum(bool(record.sends) for record in task_queued_records)
        ack_count = sum(bool(record.acks) for record in task_queued_records)
        finish_count = sum(bool(record.finishes) for record in task_queued_records)
        fb_sent_count = sum(bool(record.sends) for record in task_first_batch_records)
        fb_finish_count = sum(bool(record.finishes) for record in task_first_batch_records)
        mix_sent_count = sum(bool(record.sends) for record in task_mix_records)
        mix_finish_count = sum(bool(record.finishes) for record in task_mix_records)
        expected = device_task.expected_core_count
        finished = device_task.finished_core_count
        status = "异常" if device_task.structural_issues() or abnormal_records else "正常"
        start_tids = sorted(
            {
                event.get_int("tid")
                for event in device_task.start_events
                if event.get_int("tid") is not None
            }
        )
        end_tids = sorted(
            {
                event.get_int("tid")
                for event in device_task.end_events
                if event.get_int("tid") is not None
            }
        )
        lines.extend(
            [
                f"  Round {device_task.round_id} / DeviceTask {device_task.task_id}: {status}",
                f"    时间        : {format_timestamp(device_task.start_time)} -> "
                f"{format_timestamp(device_task.end_time)} "
                f"({format_duration(device_task.start_time, device_task.end_time)})",
                f"    调度线程    : start={start_tids}, end={end_tids}",
                f"    DevTask CoreTask 计数 : finished={finished if finished is not None else '-'} / "
                f"expected={expected if expected is not None else '-'}",
                f"    首批 CoreTask: unique={len(task_first_batch_records)}, "
                f"sent={fb_sent_count}, finished={fb_finish_count}",
                f"    入队 CoreTask: unique={len(task_queued_records)}, "
                f"sent={sent_count}, ack={ack_count}, finished={finish_count}, "
                f"abnormal={len(abnormal_records)}",
                f"    MIX CoreTask : unique={len(task_mix_records)}, "
                f"sent={mix_sent_count}, finished={mix_finish_count}",
            ]
        )
        core_ranges = control_core_ranges(
            result, device_task.round_id, prefer_snapshot=bool(device_task.abnormal_end_events)
        )
        if core_ranges:
            total_aic = sum(aic_end - aic_start for _, aic_start, aic_end, _, _, _ in core_ranges)
            total_aiv = sum(aiv_end - aiv_start for _, _, _, aiv_start, aiv_end, _ in core_ranges)
            lines.append(
                f"    控核统计    : AIC={total_aic}, AIV={total_aiv}, total={total_aic + total_aiv}"
            )
            for scheduler, aic_start, aic_end, aiv_start, aiv_end, disabled in core_ranges:
                switch_text = ""
                if disabled is not None:
                    switch_text = f", ctrlCoreDisabled={disabled}"
                lines.append(
                    f"      Scheduler {scheduler}: "
                    f"AIC=[{aic_start}, {aic_end}) count={aic_end - aic_start}, "
                    f"AIV=[{aiv_start}, {aiv_end}) count={aiv_end - aiv_start}"
                    f"{switch_text}"
                )
        for issue in device_task.structural_issues():
            lines.append(f"    DeviceTask 异常: {issue}")
    lines.append("")

    lines.append("4. 异常 CoreTask")
    lines.append("-" * 88)
    if not abnormal_core_tasks:
        lines.append("  无。所有 CoreTask 均具备 resolve + send + finish 生命周期。")
    else:
        grouped_records = defaultdict(list)
        for record in abnormal_core_tasks:
            grouped_records[(record.round_id, record.dev_task_id)].append(record)
        for (round_id, dev_task_id), records in sorted(grouped_records.items()):
            lines.append(f"  Round {round_id} / DeviceTask {dev_task_id}:")
            for record in sorted(records, key=lambda item: item.task_id):
                leaf_hash = record.leaf_hash
                leaf_text = f"0x{leaf_hash:X}" if leaf_hash is not None else "-"
                lines.append(
                    f"    CoreTask {record.task_id} (0x{record.task_id:X}) : "
                    f"leafHash={leaf_text}  "
                    f"{record.stage_text()}  Scheduler: {record.scheduler_ids or '-'} "
                    f"Core: {record.core_names or '-'}  "
                    f"异常: {'；'.join(record.issues())}"
                )
    lines.append("")

    lines.append("5. MIX wrap 统计")
    lines.append("-" * 88)
    if result.mix_wraps:
        complete_count = sum(record.complete for record in result.mix_wraps.values())
        lines.append(
            f"  总数={len(result.mix_wraps)}，完成={complete_count}，"
            f"异常={len(abnormal_mix_wraps)}"
        )
        for record in sorted(
            result.mix_wraps.values(),
            key=lambda item: (item.round_id, item.dev_task_id, item.wrap_id),
        ):
            if record.complete and not record.issues():
                continue
            state = "异常"
            path = (
                "direct"
                if record.direct_send is True
                else "readyQueue"
                if record.direct_send is False
                else "unknown"
            )
            task_text = ", ".join(
                f"{MIX_ROLE_NAMES[role]}={task_id}"
                for role, task_id in sorted(record.tasks.items())
            )
            core_text = ", ".join(
                f"{MIX_ROLE_NAMES[role]}={core}"
                for role, core in sorted(record.cores.items())
            )
            lines.extend(
                [
                    f"  Round {record.round_id} / DeviceTask {record.dev_task_id} / "
                    f"Wrap {record.wrap_id}: {state}",
                    f"    类型/路径 : {MIX_TYPE_NAMES.get(record.mix_type, record.mix_type)} / {path}",
                    f"    Tasks     : {task_text or '-'}",
                    f"    Cores     : {core_text or '-'}",
                    f"    状态      : sent=[{', '.join(MIX_ROLE_NAMES[r] for r in sorted(record.sent_roles))}], "
                    f"finished=[{', '.join(MIX_ROLE_NAMES[r] for r in sorted(record.finished_roles))}], "
                    f"allFinish={int(record.all_finish)}",
                ]
            )
            issues = record.issues()
            if issues:
                lines.append(f"    异常原因  : {'；'.join(issues)}")
    else:
        lines.append("  日志中未找到 mix 任务 trace")
    lines.append("")

    if abnormal_core_tasks and result.aicore_statuses:
        lines.append("6. 异常时 AICore 快照")
        lines.append("-" * 88)
        abnormal_cores = {
            event.get_int("coreIdx")
            for record in abnormal_core_tasks
            for event in record.sends + record.acks + record.finishes
            if event.get_int("coreIdx") is not None
        }
        for event in sorted(result.aicore_statuses, key=lambda e: e.get_int("core") or 0):
            core = event.get_int("core")
            if core not in abnormal_cores:
                continue
            raw_status = event.get_int("aicoreStatus", 0) or 0
            stage = raw_status & 0xFFFFFFFF
            stage_task = raw_status >> 32
            lines.append(
                f"  {CORE_TYPE_NAMES.get(event.get_int('type'), 'CORE')}[{core}]: "
                f"running={event.get_int('runningId')}, pending={event.get_int('pendingId')}, "
                f"stage={stage}({DFX_STAGE_NAMES.get(stage, 'UNKNOWN')}), "
                f"stageTask={stage_task if stage_task else '-'}, "
                f"lastword={event.get_int('lastwordStatus')}, "
                f"finishedReg={event.get_int('finishedTaskReg')}"
            )
        lines.append("")

    if overall_issues:
        lines.append("结论")
        lines.append("-" * 88)
        for issue in overall_issues:
            lines.append(f"  - {issue}")
        lines.append("")
        lines.extend(build_abnormal_summary(result, execution_items))
    else:
        lines.extend(
            [
                "结论",
                "-" * 88,
                "  本次执行正常：DeviceTask 计数一致，所有运行时入队 CoreTask 生命周期完整。",
            ]
        )
        if result.mix_wraps:
            lines.append("  MIX wrap 生命周期统计正常。")

    rule_results = run_rules(result)
    if rule_results:
        lines.append("")
        lines.append("规则校验")
        lines.append("-" * 88)
        for rr in rule_results:
            status = "[✓]" if rr.passed else "[✗]"
            detail_text = f" — {rr.details}" if rr.details else ""
            lines.append(f"  {status} {rr.name}: {rr.description}{detail_text}")
    lines.append("")
    return "\n".join(lines)


def parse_directory(log_dir: Path) -> Tuple[AnalysisResult, str]:
    log_files = discover_log_files(log_dir)
    events, source_files = parse_log_files(log_files)
    if not events:
        raise ValueError(f"目录中未发现 #trace.* 日志: {log_dir}")
    result = build_analysis(events, source_files)
    return result, render_report(result, log_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="解析日志目录中的 PyPTO Trace，并生成单个任务生命周期报告。"
    )
    parser.add_argument("log_dir", help="日志目录；脚本会递归扫描其中的 .log/.txt/.out 文件")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        parser.error(f"日志目录不存在或不是目录: {log_dir}")

    try:
        _, report = parse_directory(log_dir)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    output_path = log_dir / REPORT_NAME
    output_path.write_text(report, encoding="utf-8")
    print(f"Trace report generated: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
