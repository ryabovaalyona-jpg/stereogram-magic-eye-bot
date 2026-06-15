import io
import logging
import os
import unittest
from unittest.mock import patch

from PIL import Image

import main


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = main.SessionStore(
            max_users=2,
            ttl_seconds=10,
            clock=self.clock,
        )
        self.tile = Image.new(
            "RGB",
            (main.PATTERN_WIDTH, main.PATTERN_WIDTH),
            "purple",
        )

    def test_expired_session_is_removed(self):
        self.store.set(1, self.tile)
        self.clock.advance(11)

        self.assertIsNone(self.store.get(1))
        self.assertEqual(len(self.store), 0)

    def test_store_evicts_least_recently_used_session(self):
        self.store.set(1, self.tile)
        self.store.set(2, self.tile)
        self.assertIsNotNone(self.store.get(1))

        self.store.set(3, self.tile)

        self.assertIsNotNone(self.store.get(1))
        self.assertIsNone(self.store.get(2))
        self.assertIsNotNone(self.store.get(3))


class ImageTests(unittest.TestCase):
    def test_uploaded_photo_is_reduced_to_pattern_tile(self):
        source = Image.new("RGB", (1200, 900), "navy")
        source_buffer = io.BytesIO()
        source.save(source_buffer, format="JPEG")

        tile = main.decode_pattern_tile(source_buffer.getvalue())

        self.assertEqual(
            tile.size,
            (main.PATTERN_WIDTH, main.PATTERN_WIDTH),
        )
        self.assertEqual(tile.mode, "RGB")

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(main.InvalidImageError):
            main.decode_pattern_tile(b"not an image")

    def test_stereogram_and_hint_are_generated(self):
        tile = Image.new(
            "RGB",
            (main.PATTERN_WIDTH, main.PATTERN_WIDTH),
            (30, 120, 210),
        )

        result_bytes, hint_bytes = main.render_shape_images(tile, "heart")

        with Image.open(io.BytesIO(result_bytes)) as result:
            self.assertEqual(
                result.size,
                (main.OUTPUT_WIDTH, main.OUTPUT_HEIGHT),
            )
            self.assertEqual(result.format, "PNG")

        with Image.open(io.BytesIO(hint_bytes)) as hint:
            self.assertEqual(hint.size[0], 400)
            self.assertEqual(hint.format, "PNG")


class ConfigurationTests(unittest.TestCase):
    def test_http_client_logs_do_not_expose_api_urls(self):
        self.assertGreaterEqual(
            logging.getLogger("httpx").getEffectiveLevel(),
            logging.WARNING,
        )
        self.assertGreaterEqual(
            logging.getLogger("httpcore").getEffectiveLevel(),
            logging.WARNING,
        )

    def test_missing_token_exits_with_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as raised:
                main.main()

        self.assertEqual(raised.exception.code, 1)

    def test_bot_token_is_supported_as_fallback(self):
        with patch.dict(
            os.environ,
            {"BOT_TOKEN": "fallback-token"},
            clear=True,
        ):
            self.assertEqual(main.get_bot_token(), "fallback-token")

    def test_application_builds_without_network_call(self):
        application = main.build_application(
            "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        )

        self.assertGreaterEqual(len(application.handlers), 1)
        self.assertEqual(len(application.error_handlers), 1)


class TextValidationTests(unittest.TestCase):
    def test_empty_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            main.normalize_custom_text("   ")

    def test_long_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too_long"):
            main.normalize_custom_text("123456789")

    def test_valid_text_is_trimmed(self):
        self.assertEqual(main.normalize_custom_text("  Привет  "), "Привет")


if __name__ == "__main__":
    unittest.main()
