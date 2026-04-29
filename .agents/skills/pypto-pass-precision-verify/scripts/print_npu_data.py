#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Print NPU Data Tool
用于在已知目标Op后，打印上板数据（tensor数据、shape值、offset值）
"""

import os
import sys
import re
import json
import logging
import shutil
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PrintConfig:
    """打印配置"""
    print_type: str = "GM"
    dtype: str = "float"
    end_offset: int = 63
    start_offset: int = 0
    insert_pos: str = "kernel_start"


class PrintNPUDataTool:
    def __init__(self, work_path: str, pypto_root: str = "."):
        self.work_path = Path(work_path)
        self.pypto_root = Path(pypto_root)
        self.output_dir = self.work_path / "output"
        self.log_dir = self.work_path / "log" / "debug"
        
        self.config_file = self.pypto_root / "framework/src/interface/configs/tile_fwk_config.json"
        self.print_header = self.pypto_root / "framework/src/interface/machine/device/tilefwk/aicore_print.h"
        
        self.cce_files: List[Path] = []
    
    @staticmethod
    def parse_log(log_file: Path) -> Dict:
        """解析日志获取打印数据"""
        if not log_file or not log_file.exists():
            return {}
        
        content = log_file.read_text(errors='ignore')
        data = {
            'tensor_data': [],
            'shape_data': [],
            'other_logs': []
        }
        
        lines = content.split('\n')
        for line in lines:
            if 'tensor data, range=' in line:
                data['tensor_data'].append(line)
            elif 'shape' in line and ('dim' in line or '[' in line):
                data['shape_data'].append(line)
            elif 'AiCorePrintShape' in line or 'AiCoreLogF' in line or 'DumpAicoreLog' in line:
                data['other_logs'].append(line)
        
        return data
    
    @staticmethod
    def find_kernel_functions(cce_content: str) -> List[Dict]:
        """解析CCE中的kernel函数结构"""
        kernels = []
        
        pattern = r'\[aicore\]\s+void\s+(\w+)\s*\((.*?)\)\s*\{'
        
        for match in re.finditer(pattern, cce_content):
            func_name = match.group(1)
            params = match.group(2)
            start_pos = match.start()
            
            brace_count = 0
            end_pos = start_pos
            for i in range(start_pos, len(cce_content)):
                if cce_content[i] == '{':
                    brace_count += 1
                elif cce_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            
            kernels.append({
                'name': func_name,
                'start': start_pos,
                'end': end_pos,
                'params': params
            })
        
        return kernels
    
    @staticmethod
    def parse_shape_variables(content: str) -> List[str]:
        """解析 CCE 中的 shape 变量"""
        pattern = r'int64_t\s+(\w+_dim_\d+)\s*='
        shape_vars = re.findall(pattern, content)
        return shape_vars
        
    def check_env(self) -> bool:
        """检查必要的文件和目录"""
        if not self.work_path.exists():
            logger.error(f"错误: 工作目录不存在: {self.work_path}")
            return False
        return True
    
    def enable_print_switch(self):
        """开启打印开关"""
        logger.info("=== 开启打印开关 ===")
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                if 'global' not in config:
                    config['global'] = {}
                if 'codegen' not in config['global']:
                    config['global']['codegen'] = {}
                codegen = config['global']['codegen']
                
                codegen['fixed_output_path'] = True
                codegen['force_overwrite'] = False
                codegen['parallel_compile'] = 1
                
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"  [√] 已修改: {self.config_file}")
            except Exception as e:
                logger.error(f"  [×] 修改配置失败: {e}")
        
        if self.print_header.exists():
            try:
                content = self.print_header.read_text()
                if '#define ENABLE_AICORE_PRINT 0' in content:
                    content = content.replace('#define ENABLE_AICORE_PRINT 0', '#define ENABLE_AICORE_PRINT 1')
                    self.print_header.write_text(content)
                    logger.info(f"  [√] 已修改: {self.print_header}")
                elif 'ENABLE_AICORE_PRINT 1' in content:
                    logger.info(f"  [√] 已开启: {self.print_header}")
            except Exception as e:
                logger.error(f"  [×] 修改头文件失败: {e}")
        logger.info("")
        
    def rebuild_pypto(self):
        """重新编译pypto"""
        logger.info("=== 重新编译 PyPTO ===")
        os.chdir(self.pypto_root)
        logger.info(f"  执行: {sys.executable} -m pip install . -v")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", ".", "-v"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("  [√] 编译成功\n")
        else:
            logger.error(f"  [×] 编译失败")
            logger.error(f"  stderr: {result.stderr[:500]}")
        return result.returncode == 0
    
    def find_cce_files(self) -> List[Path]:
        """查找CCE文件（支持 .cce 和 .cpp 格式）"""
        kernel_dir = self.work_path / "kernel_aicore"
        if kernel_dir.exists():
            self.cce_files = sorted(kernel_dir.glob("*.cpp"), key=lambda x: x.stat().st_mtime)
        else:
            cce_files = list(self.output_dir.rglob("*.cce"))
            cpp_files = list(self.output_dir.rglob("*_aiv.cpp"))
            self.cce_files = sorted(cce_files + cpp_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"  找到 {len(self.cce_files)} 个CCE文件:")
        for i, f in enumerate(self.cce_files[:20]):
            logger.info(f"    [{i}] {f.name}")
        if len(self.cce_files) > 20:
            logger.info(f"    ... 共 {len(self.cce_files)} 个")
        return self.cce_files
    
    def find_cce_by_name_pattern(self, pattern: str) -> List[Path]:
        """根据名称模式查找CCE文件"""
        cce_files = self.find_cce_files()
        matched = []
        for cce in cce_files:
            if pattern.lower() in cce.name.lower():
                matched.append(cce)
        return matched
    
    def parse_cce_structure(self, cce_file: Path) -> Dict:
        """解析CCE文件结构"""
        content = cce_file.read_text()
        
        gm_tensors = re.findall(r'\bgmTensor_\w+', content)
        ub_tensors = re.findall(r'\bubTensor_\w+', content)
        
        shape_vars = self.parse_shape_variables(content)
        
        kernels = self.find_kernel_functions(content)
        
        return {
            'gm_tensors': gm_tensors,
            'ub_tensors': ub_tensors,
            'shape_vars': shape_vars,
            'kernels': kernels,
            'content': content
        }
    
    def add_shape_print(self, cce_file: Path, shape_vars: List[str], use_single_value: bool = False):
        """添加 shape 打印语句到 CCE 文件
        
        Args:
            cce_file: CCE 文件路径
            shape_vars: 要打印的 shape 变量列表
            use_single_value: 是否使用单值打印（AicoreLogF）而非批量打印（AiCorePrintShape）
        """
        content = cce_file.read_text()
        cce_info = self.parse_cce_structure(cce_file)
        
        if '#include "tilefwk/aicore_print.h"' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('#include') and 'aicore_print' not in line:
                    lines.insert(i + 1, '#include "tilefwk/aicore_print.h"')
                    break
            content = '\n'.join(lines)
        
        if not shape_vars:
            shape_vars = cce_info['shape_vars']
        
        if not shape_vars:
            logger.warning("  警告: 未找到 shape 变量")
            return
        
        print_stmts = []
        
        if use_single_value:
            for var in shape_vars:
                print_stmts.append(f'AicoreLogF(param->ctx, "{var}=%llu\\n", {var});')
        else:
            dim_groups = {}
            for var in shape_vars:
                match = re.match(r'(\w+)_dim_(\d+)', var)
                if match:
                    base_name = match.group(1)
                    dim_num = match.group(2)
                    if base_name not in dim_groups:
                        dim_groups[base_name] = []
                    dim_groups[base_name].append((int(dim_num), var))
            
            for _, dims in dim_groups.items():
                dims.sort()
                if len(dims) == 1:
                    var_name = dims[0][1]
                    print_stmts.append(f'AiCorePrintShape(param->ctx, Coord1Dim({var_name}));')
                elif len(dims) == 2:
                    var1 = dims[0][1]
                    var2 = dims[1][1]
                    print_stmts.append(f'AiCorePrintShape(param->ctx, Shape2Dim({var1}, {var2}));')
                elif len(dims) == 3:
                    var1 = dims[0][1]
                    var2 = dims[1][1]
                    var3 = dims[2][1]
                    print_stmts.append(f'AiCorePrintShape(param->ctx, Shape3Dim({var1}, {var2}, {var3}));')
                elif len(dims) == 4:
                    var1 = dims[0][1]
                    var2 = dims[1][1]
                    var3 = dims[2][1]
                    var4 = dims[3][1]
                    print_stmts.append(f'AiCorePrintShape(param->ctx, Shape4Dim({var1}, {var2}, {var3}, {var4}));')
        
        if not print_stmts:
            logger.warning("  警告: 无法生成 shape 打印语句")
            return
        
        print_code = "\n".join([f"    {s} // DEBUG SHAPE" for s in print_stmts])
        
        if cce_info['kernels']:
            kernel = cce_info['kernels'][0]
            first_brace = content.find('{', kernel['start'])
            if first_brace != -1:
                pos = first_brace + 1
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
                content = content[:pos] + "\n" + print_code + "\n" + content[pos:]
                
                cce_file.write_text(content)
                print_method = "AicoreLogF" if use_single_value else "AiCorePrintShape"
                logger.info(f"  已添加 {len(print_stmts)} 条 shape 打印语句 ({print_method})")
        else:
            logger.warning("  警告: 未找到 kernel 函数")
    
    def add_print_to_cce(
        self, cce_file: Path, tensor_names: List[str], config: Optional[PrintConfig] = None
    ):
        """添加打印语句到CCE
        
        Args:
            cce_file: CCE文件路径
            tensor_names: tensor名称列表
            config: 打印配置（print_type/dtype/end_offset/start_offset/insert_pos）
        """
        if config is None:
            config = PrintConfig()
        
        element_count = config.end_offset - config.start_offset + 1
        if element_count > 80:
            logger.warning(f"  警告: 元素数量 {element_count} > 80，调整偏移量范围")
            config.end_offset = config.start_offset + 79
            
        content = cce_file.read_text()
        
        if '#include "tilefwk/aicore_print.h"' not in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('#include') and 'aicore_print' not in line:
                    lines.insert(i + 1, '#include "tilefwk/aicore_print.h"')
                    break
            content = '\n'.join(lines)
        
        kernels = self.find_kernel_functions(content)
        gm_tensors = re.findall(r'\bgmTensor_\w+', content)
        ub_tensors = re.findall(r'\bubTensor_\w+', content)
        cce_info = {
            'gm_tensors': gm_tensors,
            'ub_tensors': ub_tensors,
            'kernels': kernels,
            'content': content
        }
        
        print_func = "AiCorePrintGmTensor" if config.print_type == "GM" else "AiCorePrintUbTensor"
        tensor_type = "__gm__" if config.print_type == "GM" else "__ub__"
        
        if tensor_names:
            tensor_list = tensor_names
        else:
            tensor_list = cce_info['gm_tensors'] if config.print_type == "GM" else cce_info['ub_tensors']
        
        print_stmts = []
        for tensor_name in tensor_list:
            if tensor_name in content:
                stmt = f'{print_func}(param->ctx, ({tensor_type}{config.dtype}*){tensor_name}.Getaddr(), '
                stmt += f'{config.end_offset}, {config.start_offset});'
                print_stmts.append(stmt)
        
        if not print_stmts:
            logger.warning(f"  警告: 未找到tensor {tensor_list}")
            return
        
        print_code = "\n".join([f"    {s} // DEBUG" for s in print_stmts])
        
        if config.insert_pos == "kernel_start" and cce_info['kernels']:
            kernel = cce_info['kernels'][0]
            first_brace = content.find('{', kernel['start'])
            if first_brace != -1:
                pos = first_brace + 1
                while pos < len(content) and content[pos] in ' \t\n\r':
                    pos += 1
                content = content[:pos] + "\n" + print_code + "\n" + content[pos:]
                
        elif config.insert_pos == "kernel_end" and cce_info['kernels']:
            kernel = cce_info['kernels'][0]
            line_start = content.rfind('\n', 0, kernel['end'])
            if line_start == -1:
                line_start = 0
            line_content = content[line_start:kernel['end'] + 1]
            stripped = line_content.strip()
            if stripped == '}':
                content = content[:line_start] + "\n" + print_code + content[line_start:]
            else:
                content = content[:kernel['end']] + "\n" + print_code + "\n}" + content[kernel['end'] + 1:]
            
        elif config.insert_pos == "tensor_after" and tensor_list:
            first_tensor = tensor_list[0]
            tensor_pos = content.find(f"= {first_tensor}")
            if tensor_pos == -1:
                tensor_pos = content.find(first_tensor)
            if tensor_pos != -1:
                line_end = content.find('\n', tensor_pos)
                if line_end != -1:
                    content = content[:line_end + 1] + "    " + print_code + "\n" + content[line_end + 1:]
        
        cce_file.write_text(content)
        logger.info(f"  已添加 {len(print_stmts)} 条打印语句")
    
    def run_test(self, test_cmd: List[str]) -> Tuple[int, Optional[Path]]:
        """运行测试"""
        env = os.environ.copy()
        env['ASCEND_WORK_PATH'] = str(self.work_path)
        env['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
        
        logger.info(f"  运行: {' '.join(test_cmd)}")
        result = subprocess.run(test_cmd, capture_output=True, text=True, env=env)
        
        # 查找日志
        log_files = list(self.log_dir.rglob("DumpAicoreLog*"))
        log_file: Optional[Path] = log_files[0] if log_files else None
        
        return result.returncode, log_file
    
    def detect_validshape_issues(self, ir_file: Path) -> Dict:
        """检测 validshape 问题
        
        Args:
            ir_file: IR 文件路径
            
        Returns:
            检测结果字典
        """
        if not ir_file or not ir_file.exists():
            return {'error': 'IR 文件不存在'}
        
        # 使用 get_op_info.py 脚本解析 IR 文件
        script_path = self.pypto_root / '.agents/skills/pypto-pass-error-locator/scripts/get_op_info.py'
        
        if not script_path.exists():
            return {'error': 'get_op_info.py 脚本不存在'}
        
        try:
            # 获取所有操作列表
            result = subprocess.run(
                [sys.executable, str(script_path), '--ir-file', str(ir_file), '--list-ops'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'error': f'解析 IR 文件失败: {result.stderr}'}
            
            # 解析操作列表
            ops_info = []
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('OP Magic:'):
                    # 提取操作信息
                    match = re.search(r'OP Magic: (\d+), Opcode: (\w+), Line: (\d+)', line)
                    if match:
                        op_magic = int(match.group(1))
                        opcode = match.group(2)
                        line_num = int(match.group(3))
                        ops_info.append({
                            'op_magic': op_magic,
                            'opcode': opcode,
                            'line': line_num
                        })
            
            # 检查每个操作的 shape 和 validshape
            issues = []
            for op_info in ops_info:
                cmd = [
                    sys.executable, str(script_path), '--ir-file', str(ir_file),
                    '--op-magic', str(op_info['op_magic']), '--format', 'json'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    try:
                        op_data = json.loads(result.stdout)
                        
                        # 检查 output shape 和 validshape
                        output_shape = op_data.get('output', {}).get('shape', [])
                        output_valid_shape = op_data.get('output', {}).get('valid_shape', [])
                        
                        # 检查输入 tensors 的 shape 和 validshape
                        input_issues = []
                        for input_tensor in op_data.get('inputs', []):
                            input_shape = input_tensor.get('shape', [])
                            input_valid_shape = input_tensor.get('valid_shape', [])
                            
                            if input_shape != input_valid_shape and input_shape and input_valid_shape:
                                input_issues.append({
                                    'tensor_id': input_tensor.get('logic_id'),
                                    'shape': input_shape,
                                    'valid_shape': input_valid_shape,
                                    'issue': 'shape != valid_shape'
                                })
                        
                        # 检查 offset 信息
                        offset_info = op_data.get('offset_info', {})
                        offset = offset_info.get('offset', [])
                        dynoffset = offset_info.get('dynoffset', [])
                        dynvalidshape = offset_info.get('dynvalidshape', [])
                        
                        # 如果有 offset/dynoffset 但没有 dynvalidshape，可能有问题
                        offset_issues = []
                        if (offset or dynoffset) and not dynvalidshape:
                            offset_issues.append({
                                'issue': 'offset/dynoffset 存在但 dynvalidshape 缺失',
                                'offset': offset,
                                'dynoffset': dynoffset
                            })
                        
                        if input_issues or offset_issues:
                            issues.append({
                                'op_magic': op_info['op_magic'],
                                'opcode': op_info['opcode'],
                                'line': op_info['line'],
                                'input_issues': input_issues,
                                'offset_issues': offset_issues
                            })
                    
                    except json.JSONDecodeError:
                        pass
            
            return {
                'total_ops': len(ops_info),
                'issues_count': len(issues),
                'issues': issues
            }
        
        except Exception as e:
            return {'error': f'检测过程出错: {str(e)}'}
    
    def generate_validshape_report(self, ir_file: Path) -> str:
        """生成 validshape 问题检测报告"""
       
        
        result = self.detect_validshape_issues(ir_file)
        
        if 'error' in result:
            return f"错误: {result['error']}"
        
        report_lines = []
        report_lines.append("=== ValidShape 问题检测报告 ===")
        report_lines.append(f"文件: {ir_file.name}")
        report_lines.append(f"总操作数: {result['total_ops']}")
        report_lines.append(f"发现问题数: {result['issues_count']}")
        report_lines.append("")
        
        if result['issues_count'] == 0:
            report_lines.append("✅ 未发现 validshape 问题")
        else:
            for issue in result['issues']:
                report_lines.append(f"### 操作 {issue['op_magic']} ({issue['opcode']}) - 第 {issue['line']} 行")
                
                if issue['input_issues']:
                    report_lines.append("  输入 Tensor 问题:")
                    for input_issue in issue['input_issues']:
                        report_lines.append(f"    Tensor ID: {input_issue['tensor_id']}")
                        report_lines.append(f"    Shape: {input_issue['shape']}")
                        report_lines.append(f"    ValidShape: {input_issue['valid_shape']}")
                        report_lines.append(f"    问题: {input_issue['issue']}")
                        report_lines.append(f"    建议: 检查 offset/dynoffset 配置是否正确")
                
                if issue['offset_issues']:
                    report_lines.append("  Offset 问题:")
                    for offset_issue in issue['offset_issues']:
                        report_lines.append(f"    问题: {offset_issue['issue']}")
                        report_lines.append(f"    Offset: {offset_issue['offset']}")
                        report_lines.append(f"    Dynoffset: {offset_issue['dynoffset']}")
                        report_lines.append(f"    建议: 检查是否需要添加 dynvalidshape")
                
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Print NPU Data Tool - 打印上板数据（tensor/shape/offset）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 初始化配置（首次使用需执行）
  python3 print_npu_data.py --init --work-path /path/to/work
 
  # 列出CCE文件信息
  python3 print_npu_data.py --work-path /path/to/work --list-cce
 
  # GM数据打印（默认偏移量0~63，共64个元素）
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0
  
  # GM数据打印（指定偏移量范围）
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0 --end-offset 79 --start-offset 0
  
  # GM数据打印（指定dtype）
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0 --dtype bfloat16_t
  
  # UB数据打印
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0 --print-type UB
 
  # Shape批量打印（使用 AiCorePrintShape）
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0 --print-shape sym_15_dim_0,sym_15_dim_1
  
  # Shape单值打印（使用 AicoreLogF）
  python3 print_npu_data.py --work-path /path/to/work --print-idx 0 --print-shape sym_15_dim_0 --single-value

  # 检测 validshape 问题
  python3 print_npu_data.py --work-path /path/to/work --ir-file path/to/ir_file.tifwkgr

打印方法：
  GM数据打印：--print-type GM（打印DDR/GM上的tensor数据，最常用）
  UB数据打印：--print-type UB（打印UB上的tensor数据）
  Shape批量打印：--print-shape（批量打印多维度shape，使用 AiCorePrintShape）
  Offset批量打印：手动添加（批量打印多个offset，使用 AiCorePrintShape + Coord2Dim）
  单值Shape打印：手动添加（打印单个shape变量，使用 AicoreLogF）
  单值Offset打印：手动添加（打印单个offset，使用 AicoreLogF）

AicoreLogF 示例（手动添加到CCE文件）：
  // 打印单个shape变量
  AicoreLogF(param->ctx, "sym_329_dim_0=%llu\\n", sym_329_dim_0);
  
  // 打印单个offset值
  AicoreLogF(param->ctx, "RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0,0,2,10,0)=%llu\\n",
             RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0,0,2,10,0));

偏移量参数说明：
  --end-offset: 打印末尾偏移量（默认63）
  --start-offset: 打印起始偏移量（默认0）
  元素数量 = end-offset - start-offset + 1（必须≤80）
        """
    )
    
    parser.add_argument("--work-path", required=True, help="ASCEND_WORK_PATH 工作目录")
    parser.add_argument("--pypto-root", default=".", help="PyPTO源码根目录")
    parser.add_argument("--init", action="store_true", help="仅初始化配置开关")
    parser.add_argument("--list-cce", action="store_true", help="仅列出CCE文件信息")
    parser.add_argument("--print-idx", type=int, help="指定打印哪个CCE(0-based index)")
    parser.add_argument("--tensor", help="指定要打印的tensor名称（多个用逗号分隔）")
    parser.add_argument("--print-type", choices=["GM", "UB"], default="GM", help="打印类型(GM/UB)")
    parser.add_argument("--dtype", choices=["float", "bfloat16_t", "half", "int32_t"], 
                        default="float", help="打印的数据类型（默认float）")
    parser.add_argument("--end-offset", type=int, default=63, 
                        help="打印末尾偏移量（默认63）")
    parser.add_argument("--start-offset", type=int, default=0, 
                        help="打印起始偏移量（默认0，元素数量=末尾-起始+1）")
    parser.add_argument("--pos", choices=["kernel_start", "kernel_end", "tensor_after"], 
                       default="kernel_start", help="打印语句插入位置")
    parser.add_argument("--rebuild", action="store_true", help="是否重新编译")
    parser.add_argument("--print-shape", help="打印 shape 变量（多个用逗号分隔）")
    parser.add_argument("--single-value", action="store_true", 
                        help="使用单值打印（AicoreLogF）而非批量打印（AiCorePrintShape）")
    parser.add_argument("--check-validshape", help="检测 validshape 问题（指定 IR 文件路径）")
    parser.add_argument("--ir-file", help="指定 IR 文件路径（用于 validshape 检测）")
    
    args = parser.parse_args()
    
    debugger = PrintNPUDataTool(args.work_path, args.pypto_root)
    
    if not debugger.check_env():
        return 1
    
    # 初始化配置
    debugger.enable_print_switch()
    
    if args.init:
        if args.rebuild:
            debugger.rebuild_pypto()
        logger.info("配置初始化完成")
        return 0
    
    if args.list_cce:
        debugger.find_cce_files()
        for i, cce in enumerate(debugger.cce_files):
            info = debugger.parse_cce_structure(cce)
            logger.info(f"\n[{i}] {cce.name}")
            logger.info(f"  GM tensors: {info['gm_tensors']}")
            logger.info(f"  UB tensors: {info['ub_tensors']}")
            logger.info(f"  Shape variables: {info['shape_vars']}")
            logger.info(f"  Kernels: {[k['name'] for k in info['kernels']]}")
        return 0
    
    if args.print_shape is not None:
        cce_files = debugger.find_cce_files()
        if args.print_idx is not None:
            if args.print_idx >= len(cce_files):
                logger.error(f"错误: CCE索引 {args.print_idx} 超出范围(0-{len(cce_files)-1})")
                return 1
            cce_files = [cce_files[args.print_idx]]
        
        for cce_file in cce_files:
            logger.info(f"\n=== 添加 Shape 打印: {cce_file.name} ===")
            
            backup = cce_file.with_suffix('.cpp.bak')
            shutil.copy(cce_file, backup)
            
            shape_vars = [s.strip() for s in args.print_shape.split(',')] if args.print_shape else []
            
            debugger.add_shape_print(cce_file, shape_vars, use_single_value=args.single_value)
            
            logger.info(f"  打印方式: {'AicoreLogF（单值打印）' if args.single_value else 'AiCorePrintShape（批量打印）'}")
            logger.info(f"  请运行测试后查看日志:")
            logger.info(f"    {args.work_path}/log/debug/device-*/DumpAicoreLog*")
        
        return 0
    
    if args.print_idx is not None:
        cce_files = debugger.find_cce_files()
        if args.print_idx >= len(cce_files):
            logger.error(f"错误: CCE索引 {args.print_idx} 超出范围(0-{len(cce_files)-1})")
            return 1
        
        cce_file = cce_files[args.print_idx]
        logger.info(f"\n=== 打印 CCE[{args.print_idx}]: {cce_file.name} ===")
        
        info = debugger.parse_cce_structure(cce_file)
        logger.info(f"  GM tensors: {info['gm_tensors']}")
        logger.info(f"  UB tensors: {info['ub_tensors']}")
        
        backup = cce_file.with_suffix('.cce.bak')
        shutil.copy(cce_file, backup)
        
        tensors = []
        if args.tensor:
            tensors = [t.strip() for t in args.tensor.split(',')]
        
        config = PrintConfig(
            print_type=args.print_type,
            dtype=args.dtype,
            end_offset=args.end_offset,
            start_offset=args.start_offset,
            insert_pos=args.pos
        )
        debugger.add_print_to_cce(cce_file, tensors, config)
        
        element_count = args.end_offset - args.start_offset + 1
        logger.info(f"\n  打印类型: {args.print_type}")
        logger.info(f"  数据类型: {args.dtype}")
        logger.info(f"  偏移量范围: {args.start_offset} ~ {args.end_offset}（共{element_count}个元素）")
        logger.info(f"  插入位置: {args.pos}")
        logger.info(f"  请运行测试后查看日志:")
        logger.info(f"    {args.work_path}/log/debug/device-*/DumpAicoreLog*")
        
        return 0
    
    if args.check_validshape or args.ir_file:
        ir_file = Path(args.check_validshape or args.ir_file)
        if not ir_file.exists():
            logger.error(f"错误: IR 文件不存在: {ir_file}")
            return 1
        
        report = debugger.generate_validshape_report(ir_file)
        logger.info(report)
        return 0
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main() or 0)