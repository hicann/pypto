#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ============================================================================
# PyPTO 上库前本地预检 (15 分钟版)
#
# 目标: 15 分钟内跑完, 尽量提前拦截 CI 门禁(ut.sh)会拦的 UT 错误。
#
# 设计取舍 (对照 CI):
#   - 只跑 gnu(Release) 增量编译 + 变更模块 UT      (~3min)   对应 make_gnu_*
#   - Python 变更模块 UT                           (~4-6min)  对应 Py3_ninja
#   - costmodel 仿真 UT                            (~1min)    对应 Py3_ninja_simulation
#   - clang/ASan、gcov 覆盖率超出预算, 不跑, 留给 CI (对应 Cpp_make_clang / 覆盖率)
#   - 900s 总预算守卫: 剩余时间不足时跳过靠后阶段并明确告警
#
# 用法:
#   bash tools/scripts/pre_commit_check.sh          # 15 分钟针对性预检(默认)
#   bash tools/scripts/pre_commit_check.sh --full   # 全量: gnu+clang/ASan+Python+覆盖率, 不限时
#
# 环境变量(自动探测失败时手动指定):
#   ASCEND_HOME_PATH         CANN 安装路径
#   PTO_TILE_LIB_CODE_PATH   pto-isa 路径: 源码目录或 CANN 安装后的 ${ASCEND_HOME_PATH}/<arch>-linux
#   PYPTO_THIRD_PARTY_PATH   三方库路径 (json/libboundscheck/makeself/cann-cmake)
#   PYPTO_UPSTREAM_URL       canonical 仓库 URL (默认 https://gitcode.com/cann/pypto.git)
#   TIME_BUDGET              总预算秒数, 默认 900
# ============================================================================

set -u

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p /tmp/opencode
TMP_DIR="$(mktemp -d /tmp/opencode/pre_commit_check.XXXXXX)"
PR_FILELIST="${TMP_DIR}/pr_filelist.txt"
COV_LOG_CPP="${TMP_DIR}/cov_cpp.log"
COV_LOG_PY="${TMP_DIR}/cov_py.log"

PY3="${PY3:-python3}"
COV_THRESHOLD=80
JOB_NUM=16
CASE_TIMEOUT=300  # 单用例超时(秒)
PY_DEFAULT_MODULES="ds_v32:interface:ir:operation:operator:pypto_pro:simulator"

FULL=0
TIME_BUDGET="${TIME_BUDGET:-900}"
START_TS=$(date +%s)
# 各阶段预估耗时(秒), 用于预算守卫
EST_GNU=300
EST_PY=240
EST_SIM=60

CANN_HOME="${ASCEND_HOME_PATH:-}"
PTO_ISA_PATH="${PTO_TILE_LIB_CODE_PATH:-}"
THIRD_PARTY_PATH="${PYPTO_THIRD_PARTY_PATH:-}"
PYPTO_UPSTREAM_URL="${PYPTO_UPSTREAM_URL:-https://gitcode.com/cann/pypto.git}"

log_info() { printf '\033[32m[INFO]\033[0m %s\n' "$*"; }
log_warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }
log_error() { printf '\033[31m[ERROR]\033[0m %s\n' "$*"; }

show_help() {
    sed -n '23,32p' "${BASH_SOURCE[0]}"
    exit 0
}

elapsed() { echo $(( $(date +%s) - START_TS )); }

# 预算守卫: 剩余时间不足则跳过该阶段 (--full 不限时)
budget_allow() {
    [[ "${FULL}" == "1" ]] && return 0
    local remain=$(( TIME_BUDGET - $(elapsed) ))
    if (( remain < $1 )); then
        log_warn "已用 $(elapsed)s, 剩余 ${remain}s 不足(预估 $1s), 跳过该阶段, 完整验证请跑 CI 或 --full。"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# 环境探测与检查
# ---------------------------------------------------------------------------
auto_detect_env() {
    [[ -z "${CANN_HOME}" ]] && for d in /usr/local/Ascend/cann-*; do
        [[ -d "$d/bin" ]] && CANN_HOME="$d" && break
    done
    # 自动探测 CANN 内置的 pto-isa（与 CI 安装方式一致）
    [[ -z "${PTO_ISA_PATH}" ]] && [[ -n "${CANN_HOME}" ]] && for d in "${CANN_HOME}/aarch64-linux" "${CANN_HOME}/x86_64-linux"; do
        [[ -d "${d}/include/pto" ]] && PTO_ISA_PATH="${d}" && break
    done
    [[ -z "${THIRD_PARTY_PATH}" ]] && for d in "${WORKSPACE}"/../pypto_3rd_lib_path "${WORKSPACE}"/../pypto_download/third_party_packages; do
        [[ -d "$d/json" || -d "$d/lib_cache" ]] && THIRD_PARTY_PATH="$d" && break
    done
}

check_env() {
    local ok=1
    [[ -n "${CANN_HOME}" && -f "${CANN_HOME}/bin/setenv.bash" ]] \
        || { log_error "CANN 未找到, 请设置 ASCEND_HOME_PATH。"; ok=0; }
    [[ -n "${PTO_ISA_PATH}" && -d "${PTO_ISA_PATH}/include/pto" ]] \
        || { log_error "pto-isa 未找到 (PTO_TILE_LIB_CODE_PATH), kernel 编译依赖。"; ok=0; }
    [[ -n "${THIRD_PARTY_PATH}" ]] || log_warn "未探测到三方库 (PYPTO_THIRD_PARTY_PATH), 离线环境需指定。"
    command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1 \
        || { log_error "gcc/g++ 未安装。"; ok=0; }
    command -v ninja >/dev/null 2>&1 || log_warn "ninja 未安装 (Python UT 需要)。"
    command -v "${PY3}" >/dev/null 2>&1 || { log_error "${PY3} 未找到。"; ok=0; }
    local entry
    for entry in torch torch_npu pytest xdist pytest_forked; do
        "${PY3}" -c "import ${entry}" >/dev/null 2>&1 \
            || { log_error "Python 依赖缺失: ${entry}"; ok=0; }
    done
    if [[ "${FULL}" == "1" ]]; then
        command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1 \
            || { log_error "clang 未安装 (--full 需要)。"; ok=0; }
        if command -v lcov >/dev/null 2>&1; then
            lcov --help 2>/dev/null | grep -q "build-directory" \
                || log_warn "lcov 版本过旧 (不支持 --build-directory), 覆盖率会失败, 需 lcov 2.x。"
        fi
    fi
    return $((1 - ok))
}

# ---------------------------------------------------------------------------
# 变更识别与模块映射
# ---------------------------------------------------------------------------
# 尽量同步远程基线到最新, 避免本地基线过期导致 diff 过大
fetch_base() {
    local base="$1" remote="" branch=""
    # 通过 canonical URL 拉取的临时 ref 已在 setup_canonical_base 中维护
    [[ "${base}" == refs/* ]] && return 0
    [[ "${base}" =~ ^([^/]+)/(.+)$ ]] || return 0
    remote="${BASH_REMATCH[1]}"
    branch="${BASH_REMATCH[2]}"
    git -C "${WORKSPACE}" rev-parse --verify "${remote}/${branch}" >/dev/null 2>&1 || return 0
    if git -C "${WORKSPACE}" fetch --no-tags "${remote}" "${branch}" >/dev/null 2>&1; then
        log_info "已同步基线 ${base} 到最新"
    else
        log_warn "无法拉取 ${base}, 使用本地缓存, 可能过期。"
    fi
}

# 没有 upstream remote 时, 从 canonical URL 拉取最新 master 到临时 ref,
# 避免 fork 的 origin/master 过期, 同时不污染用户的 remote 配置
setup_canonical_base() {
    git -C "${WORKSPACE}" rev-parse --verify "upstream/master" >/dev/null 2>&1 && return 0
    if git -C "${WORKSPACE}" fetch --no-tags "${PYPTO_UPSTREAM_URL}" "master:refs/precommit/pypto-upstream-master" >/dev/null 2>&1; then
        return 0
    fi
    return 0
}

detect_base() {
    local default_branch ref
    default_branch="$(git -C "${WORKSPACE}" rev-parse --abbrev-ref 'origin/HEAD' 2>/dev/null | sed 's|^origin/||')"
    # 优先使用上游仓库 (upstream/master) 作为基线, 避免 fork 的 origin/master 过期导致误判
    # 若无 upstream remote, 则使用从 canonical URL 拉取的临时 ref
    for ref in "upstream/master" "refs/precommit/pypto-upstream-master" "origin/${default_branch}" "origin/master" "origin/main" "${default_branch}" "master" "main" "HEAD~1"; do
        [[ -n "${ref}" ]] || continue
        git -C "${WORKSPACE}" rev-parse --verify "${ref}" >/dev/null 2>&1 && { echo "${ref}"; return; }
    done
    echo "HEAD~1"
}

# 输出: HAS_CPP HAS_PY UT_MODULES PY_MODULES
HAS_CPP=0; HAS_PY=0; UT_MODULES=""; PY_MODULES=""

add_mod() {  # $1=当前列表 $2=新模块
    case ":$1:" in *":$2:"*) echo "$1" ;; *) echo "${1}${1:+:}$2" ;; esac
}

analyze_changes() {
    git -C "${WORKSPACE}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { log_error "非 git 仓库"; return 1; }
    local base raw_cnt unique_cnt uncommitted unique_commits f mod

    # 先确保有一个最新的上游基线可用 (upstream/master 优先, 否则从 canonical URL 拉取临时 ref)
    setup_canonical_base
    base="$(detect_base)"
    fetch_base "${base}"

    # 原始提交数 (含 cherry-pick/merge 的他人提交)
    raw_cnt="$(git -C "${WORKSPACE}" rev-list --count --right-only --no-merges "${base}...HEAD" 2>/dev/null || echo 0)"
    # 排除与基线 patch 相同的提交后, 真正属于本 PR 的提交数
    unique_cnt="$(git -C "${WORKSPACE}" rev-list --count --cherry-pick --right-only --no-merges "${base}...HEAD" 2>/dev/null || echo 0)"
    uncommitted="$(git -C "${WORKSPACE}" status --porcelain 2>/dev/null)"

    # 只取本 PR 真正引入的 commit 所改动的文件, 避免过期基线或 cherry-pick 导致误判
    unique_commits="$(git -C "${WORKSPACE}" log --cherry-pick --right-only --no-merges --pretty=format:%H "${base}...HEAD" 2>/dev/null)"
    {
        if [[ -n "${unique_commits}" ]]; then
            echo "${unique_commits}" | git -C "${WORKSPACE}" diff-tree --stdin --no-commit-id --name-only -r 2>/dev/null
        fi
        # 未提交改动 (已暂存 + 未暂存)
        git -C "${WORKSPACE}" diff --name-only HEAD 2>/dev/null
        # 未跟踪文件
        git -C "${WORKSPACE}" ls-files --others --exclude-standard 2>/dev/null
    } | sort -u > "${PR_FILELIST}"

    log_info "变更基线: ${base} | 本 PR commit 数: ${unique_cnt} (原始 ${raw_cnt}) | 未提交: $([[ -n "${uncommitted}" ]] && echo 是 || echo 否)"
    if (( raw_cnt > unique_cnt + 5 )); then
        log_warn "分支历史包含大量非本 PR 的提交 (cherry-pick/merge 他人代码), 建议先 rebase 到 ${base}。"
    fi

    while read -r f; do
        [[ -z "${f}" ]] && continue
        case "${f}" in
            framework/*)
                HAS_CPP=1
                case "${f}" in
                    framework/src/machine/*|framework/src/adapter/*|framework/src/cann_host_runtime/*|framework/src/platform/*) mod="machine" ;;
                    framework/src/cost_model/*)  mod="simulation" ;;
                    framework/src/passes/*)      mod="passes" ;;
                    framework/src/interface/*|framework/include/*|framework/src/utils/*) mod="interface" ;;
                    framework/src/codegen/*)     mod="codegen" ;;
                    framework/tests/ut/*)        mod="${f#framework/tests/ut/}"; mod="${mod%%/*}"; [[ "${mod}" == "utils" ]] && mod="interface" ;;
                    *)                           mod="" ;;
                esac
                [[ -n "${mod}" ]] && UT_MODULES="$(add_mod "${UT_MODULES}" "${mod}")"
                ;;
            python/*)
                HAS_PY=1
                case "${f}" in
                    python/pypto_pro/*|python/src/bindings/pypto_pro/*)             mod="pypto_pro" ;;
                    python/src/bindings/ir/*|python/pypto/pil/*|python/pypto/ir.py) mod="ir" ;;
                    python/pypto/frontend/*)                                        mod="interpreter" ;;
                    python/pypto/op/*|python/pypto/operation.py)                    mod="operation" ;;
                    python/pypto/operator.py)                                       mod="operator" ;;
                    python/tests/ut/*)
                        mod="${f#python/tests/ut/}"; mod="${mod%%/*}"
                        [[ "${mod}" == "simulator" ]] && mod="interface"
                        [[ "${mod}" == "kirin" ]] && mod=""  # kirin 被 --ignore 跳过
                        ;;
                    python/pypto/*) mod="interface" ;;
                    *)              mod="" ;;
                esac
                [[ -n "${mod}" ]] && PY_MODULES="$(add_mod "${PY_MODULES}" "${mod}")"
                ;;
        esac
    done < "${PR_FILELIST}"
    [[ "${HAS_CPP}" == "1" && -z "${UT_MODULES}" ]] && UT_MODULES="interface"
    return 0
}

# ---------------------------------------------------------------------------
# UT 任务
# ---------------------------------------------------------------------------
need_cpp_clean() {
    # 返回 0 表示需要 --clean 重配置
    # 情况: 缓存是 Ninja(与 C++ Makefiles 冲突)、遗留 Python 前端开关、gcc<->clang 切换
    local cache="${WORKSPACE}/build/CMakeCache.txt" cxx=""
    [[ -f "${cache}" ]] || return 1
    grep -qE '^(CMAKE_GENERATOR:INTERNAL=Ninja|ENABLE_FEATURE_PYTHON_FRONT_END:BOOL=ON)' "${cache}" && return 0
    cxx="$(grep -aE '^CMAKE_CXX_COMPILER:[A-Z]*=' "${cache}" | head -1 | cut -d= -f2)"
    [[ -n "${cxx}" && "${cxx}" == *clang* ]] && return 0
    return 1
}

# 执行命令, 有日志文件时同时 tee 到日志
run_with_log() {
    local logfile="$1"; shift
    if [[ -n "${logfile}" ]]; then
        "$@" 2>&1 | tee "${logfile}"
        return "${PIPESTATUS[0]}"
    fi
    "$@"
}

run_gnu() {
    local clean_flag="" cov_flag="" logfile=""
    [[ "${FULL}" == "1" ]] && { cov_flag="--gcov --cov_increment"; logfile="${COV_LOG_CPP}"; }
    if [[ "${FULL}" == "1" ]] || need_cpp_clean; then
        clean_flag="--clean"
        [[ "${FULL}" == "0" ]] && log_warn "构建缓存不兼容需重编译 (有 ccache 较快), 首次全量编译可能超出 ${TIME_BUDGET}s 预算。"
    fi
    log_info ">>> gnu UT (Release/gcc, 模块: ${UT_MODULES})"
    run_with_log "${logfile}" "${PY3}" build_ci.py ${clean_flag} --frontend=cpp --build_type=Release --utest \
        --case_execute_timeout=${CASE_TIMEOUT} --utest_module="${UT_MODULES}" ${cov_flag} \
        --changed_files="${PR_FILELIST}" --cann_3rd_lib_path="${THIRD_PARTY_PATH}" \
        --job_num="${JOB_NUM}" --target=tile_fwk_utest
}

run_clang() {
    # 仅 --full; 对应 CI Cpp_make_clang, 本地 10 分钟预算跑不了
    log_info ">>> clang UT (Debug/clang+ASan, 模块: ${UT_MODULES})"
    "${PY3}" build_ci.py --clean --frontend=cpp --build_type=Debug --utest \
        --case_execute_timeout=${CASE_TIMEOUT} --utest_module="${UT_MODULES}" --clang --asan \
        --cann_3rd_lib_path="${THIRD_PARTY_PATH}" \
        --job_num="${JOB_NUM}" --target=tile_fwk_utest
}

run_py() {
    # py3 用 Ninja 生成器, 与 gnu/clang 的 Makefiles 共用 build/ 会冲突, 必须 --clean 重配置
    local cov_flag="" logfile=""
    [[ "${FULL}" == "1" ]] && { cov_flag="--py_cov --cov_increment"; logfile="${COV_LOG_PY}"; }
    log_info ">>> Python UT (模块: ${PY_MODULES:-${PY_DEFAULT_MODULES}})"
    run_with_log "${logfile}" "${PY3}" build_ci.py --clean --generator=Ninja \
        '--utest=python/tests/ut --ignore=python/tests/ut/kirin' \
        --py_abi=37 --case_execute_timeout=${CASE_TIMEOUT} \
        --changed_files="${PR_FILELIST}" --cann_3rd_lib_path="${THIRD_PARTY_PATH}" \
        --job_num="${JOB_NUM}" --verbose --editable --no_isolation \
        --utest_module="${PY_MODULES:-${PY_DEFAULT_MODULES}}" ${cov_flag}
}

run_sim() {
    log_info ">>> costmodel 仿真 UT"
    rm -rf "${WORKSPACE}/output"
    "${PY3}" python/tests/ut/simulator/costmodel_cpu_swimlane.py
}

# ---------------------------------------------------------------------------
# 覆盖率门禁 (仅 --full)
# ---------------------------------------------------------------------------
check_cov() {
    [[ "${FULL}" == "1" ]] || return 0
    local ok=1 overall pair key name logfile
    for pair in "cpp:C++" "py:Python"; do
        key="${pair%%:*}"; name="${pair##*:}"
        [[ "${key}" == "cpp" ]] && logfile="${COV_LOG_CPP}" || logfile="${COV_LOG_PY}"
        [[ -f "${logfile}" ]] || { log_warn "未生成 ${name} 覆盖率日志。"; continue; }
        overall="$(grep -oP 'Overall Coverage:\s*\K[0-9.]+' "${logfile}" | head -1)"
        if [[ -z "${overall}" ]]; then
            log_warn "${name} 无增量覆盖率数据 (可能无相关源码变更)。"
        elif awk "BEGIN{exit !(${overall} < ${COV_THRESHOLD})}"; then
            log_error "${name} 增量覆盖率 ${overall}% < ${COV_THRESHOLD}%, 未达标。"; ok=0
        else
            log_info "${name} 增量覆盖率 ${overall}% 达标。"
        fi
    done
    return $((1 - ok))
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
main() {
    local dry_run=0
    case "${1:-}" in
        --full) FULL=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) show_help ;;
        "") : ;;
        *) log_error "未知参数: $1"; show_help ;;
    esac

    cd "${WORKSPACE}" || exit 1
    log_info "工作目录: ${WORKSPACE} | 模式: $([[ "${FULL}" == "1" ]] && echo full || echo "quick(预算 ${TIME_BUDGET}s)")"

    if [[ "${dry_run}" == "1" ]]; then
        analyze_changes || exit 1
        log_info "HAS_CPP=${HAS_CPP} HAS_PY=${HAS_PY}"
        log_info "UT_MODULES=${UT_MODULES:-<无>}"
        log_info "PY_MODULES=${PY_MODULES:-<无>}"
        exit 0
    fi

    auto_detect_env
    check_env || { log_error "环境自检未通过, 请先修复。"; exit 1; }
    source "${CANN_HOME}/bin/setenv.bash"

    analyze_changes || exit 1
    if [[ "${HAS_CPP}" == "0" && "${HAS_PY}" == "0" ]]; then
        log_info "仅文档/配置变更, 跳过全部 UT。耗时 $(elapsed)s。"
        exit 0
    fi

    # --full 模式下跑全量模块, 与 CI 的覆盖/ASan 范围对齐; quick 模式仍只跑变更模块
    if [[ "${FULL}" == "1" ]]; then
        [[ "${HAS_CPP}" == "1" ]] && UT_MODULES="machine:simulation:passes:interface:codegen:operator"
        [[ "${HAS_PY}" == "1" ]] && PY_MODULES="ds_v32:interface:ir:kirin:operation:operator:pypto_pro:simulator"
        JOB_NUM=32
    fi

    local rc
    if [[ "${HAS_CPP}" == "1" ]] && budget_allow "${EST_GNU}"; then
        run_gnu; rc=$?
        [[ "${rc}" != "0" ]] && { log_error "gnu UT 失败。"; exit 1; }
    fi
    if [[ "${FULL}" == "1" && "${HAS_CPP}" == "1" ]]; then
        run_clang; rc=$?
        [[ "${rc}" != "0" ]] && { log_error "clang UT 失败。"; exit 1; }
    fi
    if [[ "${HAS_PY}" == "1" ]] && budget_allow "${EST_PY}"; then
        run_py; rc=$?
        [[ "${rc}" != "0" ]] && { log_error "Python UT 失败。"; exit 1; }
    fi
    if [[ "${HAS_PY}" == "1" ]] && budget_allow "${EST_SIM}"; then
        run_sim; rc=$?
        [[ "${rc}" != "0" ]] && { log_error "仿真 UT 失败。"; exit 1; }
    fi

    check_cov || { log_error "覆盖率门禁未通过。"; exit 1; }

    echo ""
    log_info "=============================================="
    log_info "全部检查通过 (耗时 $(elapsed)s), 可以提交 PR。"
    [[ "${FULL}" == "0" ]] && log_info "注: clang/ASan 与覆盖率由 CI 兜底, 合入前可手动 --full。"
    log_info "=============================================="
    exit 0
}

main "$@"
