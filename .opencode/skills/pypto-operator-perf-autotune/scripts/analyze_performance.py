#!/usr/bin/env python3
"""
PyPTO 算子性能分析脚本

分析泳道图数据文件，生成性能统计报告。

使用方法:
    python3 analyze_performance.py <output_dir>

参数:
    output_dir: output/output_* 目录路径
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.swimlane_file = self.output_dir / "merged_swimlane.json"
        self.trace_file = self.output_dir / "machine_runtime_operator_trace.json"
        self.bubble_file = self.output_dir / "bubble_analysis.log"
        
    def analyze_bubble_log(self) -> Dict[str, Any]:
        """分析气泡日志"""
        if not self.bubble_file.exists():
            return {}
        
        with open(self.bubble_file, 'r') as f:
            content = f.read()
        
        result = {
            'threads': [],
            'total_wait_time': 0.0,
            'total_work_time': 0.0
        }
        
        lines = content.split('\n')
        for line in lines:
            if 'Execute task num' in line:
                thread_name = line.split('[')[1].split(']')[0]
                result['threads'].append({
                    'name': thread_name,
                    'work_time': 0.0,
                    'wait_time': 0.0,
                    'wait_schedule': 0.0,
                    'wait_predecessor': 0.0
                })
            elif 'Core Total Work Time' in line:
                result['threads'][-1]['work_time'] = float(line.split(':')[1].strip())
                result['total_work_time'] += result['threads'][-1]['work_time']
            elif 'Total Wait Time' in line:
                result['threads'][-1]['wait_time'] = float(line.split(':')[1].strip())
                result['total_wait_time'] += result['threads'][-1]['wait_time']
            elif 'Wait Schedule Time' in line:
                result['threads'][-1]['wait_schedule'] = float(line.split(':')[1].strip())
            elif 'Wait Predecessor Time' in line:
                result['threads'][-1]['wait_predecessor'] = float(line.split(':')[1].strip())
        
        return result

    def analyze_swimlane(self) -> Dict[str, Any]:
        """分析泳道图数据"""
        if not self.swimlane_file.exists():
            return {}
        
        with open(self.swimlane_file, 'r') as f:
            data = json.load(f)
        
        result = {
            'tasks': [],
            'total_duration': 0.0,
            'peak_memory': 0,
            'memory_events': []
        }
        
        for event in data.get('traceEvents', []):
            if event.get('ph') == 'X' and 'dur' in event:
                dur = event.get('dur', 0)
                name = event.get('name', '')
                args = event.get('args', {})
                
                if 'execution-hint' in args:
                    exec_hint = args['execution-hint']
                    avg_time = self._extract_value(exec_hint, 'Average Execution Time')
                    max_time = self._extract_value(exec_hint, 'Max Execution Time')
                    min_time = self._extract_value(exec_hint, 'Min Execution Time')
                    
                    result['tasks'].append({
                        'name': name,
                        'duration': dur,
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'min_time': min_time
                    })
                    result['total_duration'] += dur
                
                if 'OOO_Mem_Usage(UB)' in name:
                    mem_usage = args.get('/byte', 0)
                    result['peak_memory'] = max(result['peak_memory'], mem_usage)
                    result['memory_events'].append({
                        'name': name,
                        'memory': mem_usage
                    })
        
        return result
    
    def analyze_trace(self) -> Dict[str, Any]:
        """分析性能追踪数据"""
        if not self.trace_file.exists():
            return {}
        
        with open(self.trace_file, 'r') as f:
            data = json.load(f)
        
        result = {
            'stages': {},
            'total_time': 0.0
        }
        
        for event in data.get('traceEvents', []):
            if event.get('ph') == 'X' and 'dur' in event:
                name = event.get('name', '')
                dur = event.get('dur', 0)
                cat = event.get('cat', '')
                
                if 'AICPU-CTRL' in cat:
                    if name not in result['stages']:
                        result['stages'][name] = 0.0
                    result['stages'][name] += dur
                    result['total_time'] += dur
        
        return result
    
    def _extract_value(self, text: str, key: str) -> float:
        """从文本中提取数值"""
        for line in text.split('\n'):
            if key in line:
                try:
                    return float(line.split(':')[1].strip())
                except:
                    pass
        return 0.0
    
    def generate_report(self) -> str:
        """生成性能报告"""
        bubble_data = self.analyze_bubble_log()
        swimlane_data = self.analyze_swimlane()
        trace_data = self.analyze_trace()
        
        report = []
        report.append("=" * 80)
        report.append("PyPTO 算子性能分析报告")
        report.append("=" * 80)
        report.append("")
        
        # 线程性能
        if bubble_data.get('threads'):
            report.append("## 线程性能")
            report.append("")
            report.append("| 线程 | 工作时间 | 等待时间 | 等待调度 | 等待前驱 | 利用率 |")
            report.append("|------|---------|---------|---------|---------|--------|")
            for thread in bubble_data['threads']:
                total_time = thread['work_time'] + thread['wait_time']
                utilization = (thread['work_time'] / total_time * 100) if total_time > 0 else 0
                report.append(f"| {thread['name']} | {thread['work_time']:.2f}us | {thread['wait_time']:.2f}us |"
                    f" {thread['wait_schedule']:.2f}us | {thread['wait_predecessor']:.2f}us | {utilization:.1f}% |")
            report.append("")
        
        # 任务执行
        if swimlane_data.get('tasks'):
            report.append("## 任务执行性能")
            report.append("")
            avg_duration = sum(t['duration'] for t in swimlane_data['tasks']) / len(swimlane_data['tasks'])
            report.append(f"- 总任务数: {len(swimlane_data['tasks'])}")
            report.append(f"- 平均执行时间: {avg_duration:.2f}us")
            report.append(f"- 总执行时间: {swimlane_data['total_duration']:.2f}us")
            report.append(f"- 峰值内存使用: {swimlane_data['peak_memory']} bytes")
            report.append("")
        
        # 控制开销
        if trace_data.get('stages'):
            report.append("## 控制开销")
            report.append("")
            report.append("| 阶段 | 时间 | 占比 |")
            report.append("|------|------|------|")
            for stage, time in sorted(trace_data['stages'].items(), key=lambda x: x[1], reverse=True):
                ratio = (time / trace_data['total_time'] * 100) if trace_data['total_time'] > 0 else 0
                report.append(f"| {stage} | {time:.2f}us | {ratio:.1f}% |")
            report.append(f"| **总计** | **{trace_data['total_time']:.2f}us** | **100%** |")
            report.append("")
        
        # 性能评级
        report.append("## 性能评级")
        report.append("")
        
        if bubble_data.get('threads'):
            avg_utilization = sum(t['work_time'] / (t['work_time'] + t['wait_time']) * 100 
                for t in bubble_data['threads'] if (t['work_time'] + t['wait_time']) > 0) / len(bubble_data['threads'])
            report.append(f"- 线程平均利用率: {avg_utilization:.1f}%")
        
        if swimlane_data.get('tasks') and trace_data.get('total_time'):
            compute_time = swimlane_data['total_duration']
            control_time = trace_data['total_time']
            control_ratio = (control_time / (compute_time + control_time) * 100)
                if (compute_time + control_time) > 0 else 0
            report.append(f"- 控制开销占比: {control_ratio:.1f}%")
        
        report.append("")
        report.append("=" * 80)
        
        return '\n'.join(report)


def main():
    if len(sys.argv) < 2:
        print("使用方法: python3 analyze_performance.py <output_dir>")
        print("示例: python3 analyze_performance.py output/output_20260214_152549_401503_511667")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"错误: 目录不存在: {output_dir}")
        sys.exit(1)
    
    analyzer = PerformanceAnalyzer(output_dir)
    report = analyzer.generate_report()
    print(report)


if __name__ == "__main__":
    main()
