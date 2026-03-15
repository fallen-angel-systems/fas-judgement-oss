"""
Website Transport (Playwright)
-------------------------------
WHY: Many AI chatbots are embedded as website widgets (Intercom, Crisp,
     custom chat UIs). Playwright lets us interact with these via headless
     browser automation — filling the input, clicking send, waiting for
     a new message to appear.

     The selectors are configurable so users can point at any chat widget.

LAYER: Transport (infrastructure) — imports playwright (optional dep).
SOURCE: Extracted from multi-turn-engine/transport.py WebsiteTransport class.
"""

import asyncio
import time

from .base import BaseTransport


class WebsiteTransport(BaseTransport):
    """Send attacks to website chat widgets via headless browser (Playwright)."""

    transport_type = "website"

    def __init__(
        self,
        url: str,
        input_selector: str = 'input[type="text"]',
        send_selector: str = 'button[type="submit"]',
        response_selector: str = ".chat-message:last-child",
        timeout: float = 30.0,
    ):
        self.url = url
        self.input_selector = input_selector
        self.send_selector = send_selector
        self.response_selector = response_selector
        self.timeout = timeout
        self._browser = None
        self._page = None

    async def _ensure_browser(self):
        """Lazy init Playwright browser (first send only)."""
        if self._page is not None:
            return
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=True)
            self._page = await self._browser.new_page()
            await self._page.goto(self.url, wait_until="networkidle", timeout=15000)
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

    async def send(self, message: str) -> str:
        """Type message into chat widget and capture new response."""
        await self._ensure_browser()
        page = self._page

        before_count = await page.locator(self.response_selector).count()
        await page.fill(self.input_selector, message)
        await page.click(self.send_selector)

        start = time.time()
        while time.time() - start < self.timeout:
            await asyncio.sleep(1.0)
            after_count = await page.locator(self.response_selector).count()
            if after_count > before_count:
                latest = page.locator(self.response_selector).last
                text = await latest.text_content()
                return text or "[empty response]"

        return f"[TIMEOUT: No new response within {int(self.timeout)}s]"

    async def check_connection(self) -> dict:
        """Verify page loads and selectors exist."""
        try:
            await self._ensure_browser()
            input_exists = await self._page.locator(self.input_selector).count() > 0
            send_exists = await self._page.locator(self.send_selector).count() > 0
            return {
                "connected": True,
                "url": self.url,
                "input_found": input_exists,
                "send_button_found": send_exists,
            }
        except Exception as e:
            return {"connected": False, "url": self.url, "error": str(e)}

    async def close(self):
        """Cleanup browser resources."""
        if self._browser:
            await self._browser.close()
        if hasattr(self, "_pw") and self._pw:
            await self._pw.stop()
        self._page = None
        self._browser = None

    def to_dict(self) -> dict:
        return {
            "type": "website",
            "url": self.url,
            "input_selector": self.input_selector,
            "send_selector": self.send_selector,
            "response_selector": self.response_selector,
        }
