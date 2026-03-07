#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""从 openlibing.com CodeCheck 页面提取违规项。"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


WaitStrategy = Literal["domcontentloaded", "load", "networkidle", "commit"]


logging.basicConfig(level=logging.INFO, format="%(message)s")

VIOLATION_RE = re.compile(r"文件路径:([^\n:]+):(\d+)\s*问题描述[：:]([^\n]+)\s*规则[：:]([^\n]+)")


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    description: str
    rule_id: str
    rule_description: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "file": self.file,
            "line": self.line,
            "description": self.description,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
        }


@dataclass(frozen=True)
class FetcherConfig:
    """Playwright fetcher configuration."""
    wait_strategies: list[WaitStrategy]
    nav_timeout_ms: int
    selector_timeout_ms: int
    post_wait_ms: int
    debug_dir: str


class Args(argparse.Namespace):
    url: str = ""
    output: str = "json"
    group: bool = False
    retries: int = 3
    wait_strategies: str = "domcontentloaded,load,networkidle"
    nav_timeout_ms: int = 90000
    selector_timeout_ms: int = 15000
    post_wait_ms: int = 5000
    jitter_ms: int = 500
    debug_dir: str = ""


def parse_violations_from_text(text: str) -> list[Violation]:
    violations: list[Violation] = []
    for raw_match in VIOLATION_RE.findall(text):
        file_path, line_text, description, rule = cast(tuple[str, str, str, str], raw_match)
        parts = rule.strip().split(" ", 1)
        rule_id = parts[0] if parts else ""
        rule_description = parts[1] if len(parts) > 1 else ""
        violations.append(
            Violation(
                file=file_path.strip(),
                line=int(line_text),
                description=description.strip(),
                rule_id=rule_id,
                rule_description=rule_description,
            )
        )
    return violations


def _dedup_violations(items: list[Violation]) -> list[Violation]:
    seen: set[tuple[str, int, str, str]] = set()
    result: list[Violation] = []
    for item in items:
        key = (item.file, item.line, item.description, item.rule_id)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _parse_wait_strategies(raw: str) -> list[WaitStrategy]:
    allowed: set[WaitStrategy] = {"domcontentloaded", "load", "networkidle", "commit"}
    parsed = [s.strip() for s in raw.split(",") if s.strip()]
    strategies: list[WaitStrategy] = []
    for s in parsed:
        if s in allowed:
            strategies.append(cast(WaitStrategy, s))
    if strategies:
        return strategies
    return cast(list[WaitStrategy], ["domcontentloaded", "load", "networkidle"])


def _save_debug_artifacts(page: object, debug_dir: str, attempt: int, stage: str) -> None:
    if not debug_dir:
        return

    from playwright.sync_api import Page

    debug_page = cast(Page, page)
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    prefix = debug_path / f"attempt{attempt}_{stage}_{ts}"

    try:
        debug_page.screenshot(path=str(prefix.with_suffix(".png")), full_page=True)
    except Exception as exc:
        logging.debug("save screenshot failed: %s", exc)

    try:
        html = debug_page.content()
        prefix.with_suffix(".html").write_text(html, encoding="utf-8")
    except Exception as exc:
        logging.debug("save html failed: %s", exc)


def _collect_candidate_texts(page: object) -> list[str]:
    from playwright.sync_api import Page

    candidate_page = cast(Page, page)
    texts: list[str] = []
    try:
        texts.append(candidate_page.inner_text("body"))
    except Exception as exc:
        logging.debug("collect body text failed: %s", exc)

    try:
        row_texts = candidate_page.locator("tr").all_inner_texts()
        if row_texts:
            texts.append("\n".join(row_texts))
    except Exception as exc:
        logging.debug("collect tr texts failed: %s", exc)

    try:
        texts.append(candidate_page.content())
    except Exception as exc:
        logging.debug("collect page content failed: %s", exc)

    return texts


def extract_violations_with_playwright(
    url: str,
    config: FetcherConfig,
    attempt: int,
) -> list[Violation]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 2200})
            last_error: Exception | None = None

            for strategy in config.wait_strategies:
                try:
                    response = page.goto(url, wait_until=strategy, timeout=config.nav_timeout_ms)
                    status = response.status if response else None
                    logging.info("  strategy=%s status=%s", strategy, status)
                    try:
                        page.wait_for_selector(
                            ".el-table, .el-table__body-wrapper, body",
                            timeout=config.selector_timeout_ms,
                        )
                    except Exception as exc:
                        logging.debug("selector wait skipped: %s", exc)

                    page.wait_for_timeout(config.post_wait_ms)
                    try:
                        page.locator(".el-pagination__sizes").click(timeout=5000)
                        page.wait_for_timeout(500)

                        options = page.locator(".el-select-dropdown__item").all()
                        largest_opt = None
                        largest_num = 0
                        for opt in options:
                            text = opt.inner_text()
                            digits = "".join(ch for ch in text if ch.isdigit())
                            num = int(digits) if digits else 0
                            if num > largest_num:
                                largest_num = num
                                largest_opt = opt

                        if largest_opt is not None:
                            largest_opt.click()
                            page.wait_for_timeout(2000)
                    except Exception as exc:
                        logging.debug("pagination size expand skipped: %s", exc)

                    collected: list[Violation] = []
                    for text in _collect_candidate_texts(page):
                        collected.extend(parse_violations_from_text(text))
                    violations = _dedup_violations(collected)

                    if violations:
                        return violations

                    _save_debug_artifacts(page, config.debug_dir, attempt, f"empty_{strategy}")
                    logging.warning("  no violations parsed with strategy=%s", strategy)
                except Exception as exc:
                    last_error = exc
                    _save_debug_artifacts(page, config.debug_dir, attempt, f"fail_{strategy}")
                    logging.warning("  strategy=%s failed: %s", strategy, exc)
            raise RuntimeError(f"all wait strategies failed or empty, last_error={last_error}")
        finally:
            browser.close()


def extract_with_retry(
    url: str,
    max_retries: int,
    config: FetcherConfig,
    jitter_ms: int,
) -> list[Violation]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries}...")
            return extract_violations_with_playwright(
                url=url,
                config=config,
                attempt=attempt,
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if attempt < max_retries:
                wait_s = (2 ** (attempt - 1)) + random.uniform(0, max(0, jitter_ms) / 1000.0)
                logging.warning(f"  Failed: {exc}")
                logging.info(f"  Waiting {wait_s:.2f}s before retry...")
                time.sleep(wait_s)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def group_by_rule(violations: list[Violation]) -> dict[str, list[Violation]]:
    by_rule: dict[str, list[Violation]] = {}
    for violation in violations:
        by_rule.setdefault(violation.rule_id, []).append(violation)
    return by_rule


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="从 openlibing.com 提取 CodeCheck 违规列表")
    _ = parser.add_argument("url", help="CodeCheck 报告 URL")
    _ = parser.add_argument("--output", "-o", choices=["json", "text"], default="json", help="输出格式")
    _ = parser.add_argument("--group", "-g", action="store_true", help="按规则分组输出")
    _ = parser.add_argument("--retries", type=int, default=3, help="提取重试次数（默认 3）")
    _ = parser.add_argument(
        "--wait-strategies",
        default="domcontentloaded,load,networkidle",
        help="导航等待策略，逗号分隔（domcontentloaded,load,networkidle,commit）",
    )
    _ = parser.add_argument("--nav-timeout-ms", type=int, default=90000, help="导航超时毫秒")
    _ = parser.add_argument("--selector-timeout-ms", type=int, default=15000, help="关键选择器等待超时毫秒")
    _ = parser.add_argument("--post-wait-ms", type=int, default=5000, help="页面加载后额外等待毫秒")
    _ = parser.add_argument("--jitter-ms", type=int, default=500, help="重试抖动毫秒上限")
    _ = parser.add_argument("--debug-dir", default="", help="失败时保存截图/HTML的目录")
    return parser.parse_args(namespace=Args())


def main() -> int:
    args = parse_args()

    try:
        retries = max(1, int(args.retries))
        wait_strategies = _parse_wait_strategies(args.wait_strategies)
        config = FetcherConfig(
            wait_strategies=wait_strategies,
            nav_timeout_ms=max(1000, int(args.nav_timeout_ms)),
            selector_timeout_ms=max(0, int(args.selector_timeout_ms)),
            post_wait_ms=max(0, int(args.post_wait_ms)),
            debug_dir=args.debug_dir.strip(),
        )
        violations = extract_with_retry(
            url=args.url,
            max_retries=retries,
            config=config,
            jitter_ms=max(0, int(args.jitter_ms)),
        )

        if args.output == "json":
            grouped = group_by_rule(violations)
            result: dict[str, object] = {
                "total": len(violations),
                "by_rule": {rule: len(items) for rule, items in grouped.items()},
                "violations": [v.to_dict() for v in violations],
            }
            if args.group:
                result["grouped"] = {rule: [item.to_dict() for item in items] for rule, items in grouped.items()}
            logging.info(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logging.info(f"Total violations: {len(violations)}\n")
            if args.group:
                grouped = group_by_rule(violations)
                for rule, items in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    desc = items[0].rule_description if items else ""
                    logging.info(f"## {rule}: {len(items)} violations")
                    logging.info(f"   {desc}\n")
                    for item in items:
                        logging.info(f"   - {item.file}:{item.line}")
                        logging.info(f"     {item.description}\n")
            else:
                for item in violations:
                    logging.info(f"{item.rule_id} | {item.file}:{item.line}")
                    logging.info(f"  {item.description}\n")

        return 0
    except Exception as exc:
        logging.error(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
