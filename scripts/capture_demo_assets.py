#!/usr/bin/env python3
"""Capture portfolio demo screenshots and a short walkthrough video."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo"
SCREENSHOTS = OUT / "screenshots"
VIDEO = OUT / "demo"
PUBLIC_MEDIA = ROOT / "portfolio-web" / "media"


def _chromium_path() -> str | None:
    for candidate in (
        os.environ.get("CHROMIUM_PATH"),
        "/snap/bin/chromium",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def main() -> int:
    url = os.environ.get("DEMO_URL", "http://127.0.0.1:4175")
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    VIDEO.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        executable = _chromium_path()
        if executable:
            launch_kwargs["executable_path"] = executable
        browser = p.chromium.launch(**launch_kwargs)

        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.screenshot(path=SCREENSHOTS / "01-workbench-desktop.png", full_page=False)
        page.locator("#demo-form button[type=submit]").click()
        page.wait_for_timeout(1100)
        page.screenshot(path=SCREENSHOTS / "02-generated-result-desktop.png", full_page=False)

        mobile = browser.new_page(viewport={"width": 390, "height": 900}, is_mobile=True)
        mobile.goto(url, wait_until="networkidle")
        mobile.screenshot(path=SCREENSHOTS / "03-workbench-mobile.png", full_page=False)

        context = browser.new_context(
            viewport={"width": 1440, "height": 1100},
            record_video_dir=str(VIDEO),
            record_video_size={"width": 1440, "height": 1100},
        )
        tour = context.new_page()
        tour.goto(url, wait_until="networkidle")
        tour.wait_for_timeout(500)
        tour.locator("#prompt").fill(
            "interview-ready AI generation platform, task queue, asset manager, generated result preview"
        )
        tour.locator("#seed").fill("20260628")
        tour.locator("#demo-form button[type=submit]").click()
        tour.wait_for_timeout(1600)
        tour.evaluate("window.scrollTo({ top: document.querySelector('#architecture').offsetTop - 20, behavior: 'instant' })")
        tour.wait_for_timeout(900)
        tour.evaluate("window.scrollTo({ top: document.querySelector('#runbook').offsetTop - 20, behavior: 'instant' })")
        tour.wait_for_timeout(900)
        video_obj = tour.video
        context.close()
        browser.close()

        video_path = video_obj.path() if video_obj else None
        if video_path:
            target = VIDEO / "demo-walkthrough.webm"
            if target.exists():
                target.unlink()
            shutil.move(video_path, target)

    cover_src = SCREENSHOTS / "02-generated-result-desktop.png"
    cover_dst = OUT / "cover.webp"
    with Image.open(cover_src) as img:
        img.crop((0, 0, min(img.width, 1440), min(img.height, 900))).save(cover_dst, "WEBP", quality=88)

    public_screenshots = PUBLIC_MEDIA / "screenshots"
    public_demo = PUBLIC_MEDIA / "demo"
    public_screenshots.mkdir(parents=True, exist_ok=True)
    public_demo.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cover_dst, PUBLIC_MEDIA / "cover.webp")
    shutil.copy2(VIDEO / "demo-walkthrough.webm", public_demo / "demo-walkthrough.webm")
    for screenshot in sorted(SCREENSHOTS.glob("*.png")):
        shutil.copy2(screenshot, public_screenshots / screenshot.name)

    print(f"Captured assets under {OUT}")
    print(f"Synced public demo media under {PUBLIC_MEDIA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
