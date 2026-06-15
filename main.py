"""
🔮 Magic Eye Bot v3
"""

import asyncio
import io
import logging
import math
import os
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
    UnidentifiedImageError,
)

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application, CommandHandler, MessageHandler,
        CallbackQueryHandler, ConversationHandler, filters
    )
except ImportError as exc:
    print("Install dependencies with: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WAITING_PHOTO = 0
WAITING_SHAPE = 1
WAITING_TEXT = 2

PATTERN_WIDTH = 130
OUTPUT_WIDTH = 1690
OUTPUT_HEIGHT = 960
MAX_SHIFT = 20
MAX_TEXT_LENGTH = 8
SESSION_TTL_SECONDS = 60 * 60
MAX_USER_SESSIONS = 100


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as a usable image."""


@dataclass
class UserSession:
    pattern_tile: Image.Image
    last_shape: str | None = None
    expires_at: float = 0.0


class SessionStore:
    """Small in-memory LRU store with sliding expiration."""

    def __init__(
        self,
        max_users: int = MAX_USER_SESSIONS,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_users < 1:
            raise ValueError("max_users must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_users = max_users
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._sessions: OrderedDict[int, UserSession] = OrderedDict()

    def set(self, user_id: int, pattern_tile: Image.Image) -> UserSession:
        now = self.clock()
        self.cleanup(now)
        self._sessions.pop(user_id, None)
        session = UserSession(
            pattern_tile=pattern_tile,
            expires_at=now + self.ttl_seconds,
        )
        self._sessions[user_id] = session
        while len(self._sessions) > self.max_users:
            self._sessions.popitem(last=False)
        return session

    def get(self, user_id: int) -> UserSession | None:
        now = self.clock()
        self.cleanup(now)
        session = self._sessions.get(user_id)
        if session is None:
            return None
        session.expires_at = now + self.ttl_seconds
        self._sessions.move_to_end(user_id)
        return session

    def remove(self, user_id: int) -> None:
        self._sessions.pop(user_id, None)

    def cleanup(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        expired = [
            user_id
            for user_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for user_id in expired:
            self._sessions.pop(user_id, None)
        return len(expired)

    def __len__(self) -> int:
        self.cleanup()
        return len(self._sessions)


sessions = SessionStore()


def prepare_pattern_tile(photo: Image.Image) -> Image.Image:
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    width, height = photo.size
    crop_size = min(width, height)
    crop_left = (width - crop_size) // 2
    face_crop = photo.crop((crop_left, 0, crop_left + crop_size, crop_size))
    return face_crop.resize((PATTERN_WIDTH, PATTERN_WIDTH), Image.Resampling.LANCZOS)


def decode_pattern_tile(photo_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(photo_bytes)) as photo:
            photo.load()
            return prepare_pattern_tile(photo)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError("Unable to decode uploaded image") from exc


def make_pattern_strip(pattern_tile: Image.Image) -> Image.Image:
    tile = pattern_tile
    if tile.size != (PATTERN_WIDTH, PATTERN_WIDTH):
        tile = tile.resize(
            (PATTERN_WIDTH, PATTERN_WIDTH),
            Image.Resampling.LANCZOS,
        )
    strip = Image.new("RGB", (PATTERN_WIDTH, OUTPUT_HEIGHT))
    y = 0
    while y < OUTPUT_HEIGHT:
        h = min(PATTERN_WIDTH, OUTPUT_HEIGHT - y)
        strip.paste(tile.crop((0, 0, PATTERN_WIDTH, h)), (0, y))
        y += PATTERN_WIDTH
    return strip


# ============================================================
#  КАРТЫ ГЛУБИНЫ
# ============================================================

def make_depth_heart():
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)
    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2
    r = 150
    dd.ellipse([cx - r*2, cy - r - 60, cx, cy - 60 + r], fill=255)
    dd.ellipse([cx, cy - r - 60, cx + r*2, cy - 60 + r], fill=255)
    dd.polygon([(cx - r*2, cy - 10), (cx + r*2, cy - 10), (cx, cy + r*2 + 80)], fill=255)
    dd.rectangle([cx - r*2 + 20, cy - 60, cx + r*2 - 20, cy], fill=255)
    return depth.filter(ImageFilter.GaussianBlur(radius=5))


def make_depth_broken_heart():
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)
    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2
    r = 140
    d = 255
    # Сначала рисуем цельное сердце
    dd.ellipse([cx - r*2, cy - r - 60, cx, cy - 60 + r], fill=d)
    dd.ellipse([cx, cy - r - 60, cx + r*2, cy - 60 + r], fill=d)
    dd.polygon([(cx - r*2, cy - 10), (cx + r*2, cy - 10), (cx, cy + r*2 + 60)], fill=d)
    dd.rectangle([cx - r*2 + 20, cy - 60, cx + r*2 - 20, cy], fill=d)
    # Трещина — зигзаг через центр (чёрная толстая линия)
    crack = [
        (cx + 5, cy - r - 50),
        (cx - 35, cy - 80),
        (cx + 30, cy - 20),
        (cx - 25, cy + 50),
        (cx + 20, cy + 120),
        (cx - 10, cy + r*2 + 40),
    ]
    dd.line(crack, fill=0, width=20)
    return depth.filter(ImageFilter.GaussianBlur(radius=5))


def make_depth_star():
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)
    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2
    outer_r, inner_r = 300, 130  # большая звезда
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = outer_r if i % 2 == 0 else inner_r
        points.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
    dd.polygon(points, fill=255)
    return depth.filter(ImageFilter.GaussianBlur(radius=5))


def make_depth_cat():
    """Кот идущий — силуэт в профиль"""
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)
    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2 + 20
    d = 255

    # Тело — горизонтальный овал, увеличен
    dd.ellipse([cx - 280, cy - 100, cx + 230, cy + 130], fill=d)

    # Голова — круг, чуть впереди и выше
    head_cx = cx + 270
    head_cy = cy - 70
    head_r = 120
    dd.ellipse([head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r], fill=d)

    # Шея — соединение
    dd.polygon([(cx + 170, cy - 90), (head_cx - 75, head_cy + 40),
                (head_cx - 75, head_cy + 90), (cx + 170, cy + 20)], fill=d)

    # Уши
    dd.polygon([(head_cx - 55, head_cy - 100), (head_cx - 30, head_cy - 210),
                (head_cx + 20, head_cy - 95)], fill=d)
    dd.polygon([(head_cx + 35, head_cy - 105), (head_cx + 65, head_cy - 215),
                (head_cx + 105, head_cy - 90)], fill=d)

    # Передние лапы — широко расставлены вправо
    dd.rectangle([cx + 140, cy + 100, cx + 190, cy + 290], fill=d)
    dd.rectangle([cx + 60, cy + 100, cx + 110, cy + 280], fill=d)
    # Лапки
    dd.ellipse([cx + 125, cy + 272, cx + 205, cy + 308], fill=d)
    dd.ellipse([cx + 45, cy + 262, cx + 125, cy + 298], fill=d)

    # Задние лапы — широко расставлены влево
    dd.rectangle([cx - 230, cy + 80, cx - 180, cy + 270], fill=d)
    dd.rectangle([cx - 150, cy + 90, cx - 100, cy + 280], fill=d)
    # Лапки
    dd.ellipse([cx - 245, cy + 255, cx - 165, cy + 291], fill=d)
    dd.ellipse([cx - 165, cy + 265, cx - 85, cy + 301], fill=d)

    # Хвост — поднят вверх, изогнут
    for i in range(45):
        t = i / 45
        tx = cx - 280 - int(40 * t)
        ty = cy - int(300 * t) + int(80 * t * t)
        dd.ellipse([tx - 20, ty - 20, tx + 20, ty + 20], fill=d)

    return depth.filter(ImageFilter.GaussianBlur(radius=5))


def make_depth_skull():
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)
    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2 - 40
    d = 255
    # Череп — большой овал (увеличенный)
    dd.ellipse([cx - 220, cy - 250, cx + 220, cy + 140], fill=d)
    # Челюсть
    dd.ellipse([cx - 150, cy + 50, cx + 150, cy + 250], fill=d)
    # Глазницы
    dd.ellipse([cx - 150, cy - 160, cx - 30, cy - 40], fill=0)
    dd.ellipse([cx + 30, cy - 160, cx + 150, cy - 40], fill=0)
    # Нос
    dd.polygon([(cx - 30, cy + 10), (cx + 30, cy + 10), (cx, cy + 70)], fill=0)
    # Зубы
    for tx in range(-100, 101, 40):
        dd.rectangle([cx + tx - 4, cy + 120, cx + tx + 4, cy + 200], fill=0)
    return depth.filter(ImageFilter.GaussianBlur(radius=5))


def make_depth_text(text):
    text = text.upper().strip()[:MAX_TEXT_LENGTH]
    depth = Image.new('L', (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    dd = ImageDraw.Draw(depth)

    # Ищем шрифт — с поддержкой кириллицы
    font_paths = [
        "MullerBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, 300)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    cx, cy = OUTPUT_WIDTH // 2, OUTPUT_HEIGHT // 2

    if len(text) <= 4:
        _draw_bold_centered(dd, text, cx, cy, font, spacing=100)
    else:
        if ' ' in text:
            parts = text.split(' ', 1)
        else:
            mid = len(text) // 2
            parts = [text[:mid], text[mid:]]
        bb = dd.textbbox((0, 0), "A", font=font)
        line_h = bb[3] - bb[1]
        gap = 60
        _draw_bold_centered(dd, parts[0], cx, cy - line_h//2 - gap//2, font, spacing=90)
        _draw_bold_centered(dd, parts[1], cx, cy + line_h//2 + gap//2, font, spacing=90)

    return depth.filter(ImageFilter.GaussianBlur(radius=3))


def _draw_bold_centered(dd, text, cx, cy, font, spacing):
    chars = list(text)
    char_widths = []
    for ch in chars:
        bb = dd.textbbox((0, 0), ch, font=font)
        char_widths.append(bb[2] - bb[0])
    total_w = sum(char_widths) + spacing * (len(chars) - 1)
    x = cx - total_w // 2
    bb_h = dd.textbbox((0, 0), chars[0], font=font)
    h = bb_h[3] - bb_h[1]
    y = cy - h // 2
    for i, ch in enumerate(chars):
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                dd.text((x + dx, y + dy), ch, fill=255, font=font)
        x += char_widths[i] + spacing


# ============================================================
#  ГЕНЕРАТОР
# ============================================================

DEPTH_MAKERS = {
    "heart": make_depth_heart,
    "broken_heart": make_depth_broken_heart,
    "star": make_depth_star,
    "cat": make_depth_cat,
    "skull": make_depth_skull,
}

SHAPE_NAMES = {
    "heart": "❤️ Сердце",
    "broken_heart": "💔 Разбитое сердце",
    "star": "⭐ Звезда",
    "cat": "🐈 Кот",
    "skull": "💀 Череп",
}


def generate_stereogram(pattern_tile, depth_map):
    strip = make_pattern_strip(pattern_tile)
    pp = strip.load()
    dpx = depth_map.load()

    stereogram = Image.new('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    out_px = stereogram.load()

    for y in range(OUTPUT_HEIGHT):
        row = [pp[x % PATTERN_WIDTH, y] for x in range(OUTPUT_WIDTH)]
        for x in range(PATTERN_WIDTH, OUTPUT_WIDTH):
            dv = dpx[x, y] / 255.0
            shift = int(dv * MAX_SHIFT)
            src = x - PATTERN_WIDTH + shift
            if 0 <= src < OUTPUT_WIDTH:
                row[x] = row[src]
        for x in range(OUTPUT_WIDTH):
            out_px[x, y] = row[x]

    draw = ImageDraw.Draw(stereogram)
    for dx in [OUTPUT_WIDTH//2 - PATTERN_WIDTH//2, OUTPUT_WIDTH//2 + PATTERN_WIDTH//2]:
        draw.ellipse([dx-8, 14, dx+8, 30], fill=(0, 0, 0))
        draw.ellipse([dx-6, 16, dx+6, 28], fill=(255, 255, 255))

    return stereogram


def make_hint_image(depth_map, pattern_tile):
    """Подсказка — силуэт поверх фото"""
    tile = pattern_tile
    hint_w = 400
    hint_h = int(hint_w * OUTPUT_HEIGHT / OUTPUT_WIDTH)
    bg = Image.new('RGB', (hint_w, hint_h))
    tile_small = tile.resize(
        (hint_w // 4, hint_h // 4),
        Image.Resampling.LANCZOS,
    )
    for y in range(0, hint_h, tile_small.size[1]):
        for x in range(0, hint_w, tile_small.size[0]):
            bg.paste(tile_small, (x, y))
    depth_small = depth_map.resize(
        (hint_w, hint_h),
        Image.Resampling.LANCZOS,
    )
    overlay = Image.new('RGBA', (hint_w, hint_h), (0, 0, 0, 0))
    dpx = depth_small.load()
    opx = overlay.load()
    for y in range(hint_h):
        for x in range(hint_w):
            v = dpx[x, y]
            if v > 30:
                opx[x, y] = (255, 50, 50, min(180, v))
    bg = bg.convert('RGBA')
    bg = Image.alpha_composite(bg, overlay)
    return bg.convert('RGB')


def render_images(pattern_tile: Image.Image, depth_map: Image.Image) -> tuple[bytes, bytes]:
    result = generate_stereogram(pattern_tile, depth_map)
    hint = make_hint_image(depth_map, pattern_tile)

    result_buffer = io.BytesIO()
    result.save(result_buffer, format="PNG", optimize=True)

    hint_buffer = io.BytesIO()
    hint.save(hint_buffer, format="PNG", optimize=True)

    return result_buffer.getvalue(), hint_buffer.getvalue()


def render_shape_images(pattern_tile: Image.Image, choice: str) -> tuple[bytes, bytes]:
    depth = DEPTH_MAKERS.get(choice, make_depth_heart)()
    return render_images(pattern_tile, depth)


def render_text_images(pattern_tile: Image.Image, text: str) -> tuple[bytes, bytes]:
    return render_images(pattern_tile, make_depth_text(text))


def normalize_custom_text(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError("empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("too_long")
    return text


async def send_generated_images(message, result_bytes: bytes, hint_bytes: bytes, caption: str):
    with io.BytesIO(result_bytes) as result_buffer:
        await message.reply_photo(photo=result_buffer, caption=caption)

    with io.BytesIO(hint_bytes) as hint_buffer:
        await message.reply_document(
            document=hint_buffer,
            filename="hint.png",
            caption="🔍 Подсказка (нажми чтобы открыть)",
        )


# ============================================================
#  TELEGRAM
# ============================================================

async def start(update, context):
    sessions.remove(update.effective_user.id)
    await update.message.reply_text(
        "🔮 *Magic Eye Generator*\n\n"
        "Отправь мне фото — я превращу его в 3D-стереограмму!\n\n"
        "Расфокусируй взгляд, чтобы увидеть скрытое изображение 👀\n\n"
        "by @alyonafrost",
        parse_mode='Markdown'
    )
    return WAITING_PHOTO


async def help_command(update, context):
    await update.message.reply_text(
        "👁 *Как увидеть скрытое изображение?*\n\n"
        "1. Держи телефон прямо перед собой на расстоянии вытянутой руки\n"
        "2. Расслабь взгляд, как будто смотришь сквозь экран вдаль\n"
        "3. Не моргай и не фокусируйся на картинке — просто смотри «в никуда»\n"
        "4. Через несколько секунд из узора начнёт проявляться 3D-форма\n"
        "5. Когда увидишь — постарайся удержать взгляд, не фокусируясь\n\n"
        "💡 *Советы:*\n"
        "• Приблизь экран к носу, затем медленно отодвигай\n"
        "• Попробуй смотреть на своё отражение в экране\n"
        "• Хорошее освещение помогает\n\n"
        "📎 Используй /start чтобы создать новую стереограмму\n"
        "🎲 /random — случайная форма\n"
        "🔄 /again — другая случайная форма",
        parse_mode='Markdown'
    )


async def receive_photo(update, context):
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    try:
        pattern_tile = await asyncio.to_thread(
            decode_pattern_tile,
            bytes(photo_bytes),
        )
    except InvalidImageError:
        logger.warning(
            "Rejected invalid image from user_id=%s",
            update.effective_user.id,
        )
        await update.message.reply_text(
            "Не удалось прочитать это фото. Отправь другое изображение."
        )
        return WAITING_PHOTO

    sessions.set(update.effective_user.id, pattern_tile)

    keyboard = [
        [
            InlineKeyboardButton("❤️", callback_data="heart"),
            InlineKeyboardButton("💔", callback_data="broken_heart"),
            InlineKeyboardButton("⭐", callback_data="star"),
        ],
        [
            InlineKeyboardButton("🐈", callback_data="cat"),
            InlineKeyboardButton("💀", callback_data="skull"),
            InlineKeyboardButton("✍️ Текст", callback_data="text"),
        ]
    ]

    await update.message.reply_text(
        "📸 Фото получено!\n\n"
        "Выбери скрытую 3D-форму или напиши свой текст:\n\n"
        "👀 Как увидеть 3D: расфокусируй взгляд и совмести две точки наверху в три\n\n"
        "✍️ Текст — до 8 символов (латиница / кириллица)",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return WAITING_SHAPE


async def shape_chosen(update, context):
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    choice = query.data

    session = sessions.get(user_id)
    if session is None:
        await query.edit_message_text(
            "Фото больше не хранится. Отправь его ещё раз!"
        )
        return WAITING_PHOTO

    if choice == "text":
        await query.edit_message_text(
            f"✍️ Напиши текст (макс {MAX_TEXT_LENGTH} символов):"
        )
        return WAITING_TEXT

    await query.edit_message_text("🔮 Генерирую...")

    result_bytes, hint_bytes = await asyncio.to_thread(
        render_shape_images,
        session.pattern_tile.copy(),
        choice,
    )
    await send_generated_images(
        query.message,
        result_bytes,
        hint_bytes,
        caption=(
            "🔮 Готово! Расфокусируй взгляд 👀\n\n"
            "by @alyonafrost"
        ),
    )

    return WAITING_PHOTO


async def receive_text(update, context):
    try:
        text = normalize_custom_text(update.message.text)
    except ValueError as exc:
        if str(exc) == "empty":
            await update.message.reply_text(
                "Текст не должен быть пустым. Попробуй ещё:"
            )
            return WAITING_TEXT
        await update.message.reply_text(
            f"Максимум {MAX_TEXT_LENGTH} символов! Попробуй ещё:"
        )
        return WAITING_TEXT

    user_id = update.effective_user.id
    session = sessions.get(user_id)
    if session is None:
        await update.message.reply_text(
            "Фото больше не хранится. Отправь его ещё раз!"
        )
        return WAITING_PHOTO

    await update.message.reply_text("🔮 Генерирую...")

    result_bytes, hint_bytes = await asyncio.to_thread(
        render_text_images,
        session.pattern_tile.copy(),
        text,
    )
    await send_generated_images(
        update.message,
        result_bytes,
        hint_bytes,
        caption=(
            f"🔮 «{text}» — расфокусируй взгляд!\n\n"
            "by @alyonafrost"
        ),
    )

    return WAITING_PHOTO


async def random_shape(update, context):
    user_id = update.effective_user.id
    session = sessions.get(user_id)
    if session is None:
        await update.message.reply_text("Сначала отправь фото, потом /random!")
        return WAITING_PHOTO

    choice = random.choice(list(DEPTH_MAKERS))
    session.last_shape = choice
    await update.message.reply_text(
        f"🎲 Случайная форма: {SHAPE_NAMES[choice]}\n🔮 Генерирую..."
    )

    result_bytes, hint_bytes = await asyncio.to_thread(
        render_shape_images,
        session.pattern_tile.copy(),
        choice,
    )
    await send_generated_images(
        update.message,
        result_bytes,
        hint_bytes,
        caption=(
            f"🔮 Готово! Спрятано: {SHAPE_NAMES[choice]}\n"
            "Расфокусируй взгляд 👀\n\n"
            "by @alyonafrost"
        ),
    )
    return WAITING_PHOTO


async def again(update, context):
    user_id = update.effective_user.id
    session = sessions.get(user_id)
    if session is None:
        await update.message.reply_text("Сначала отправь фото!")
        return WAITING_PHOTO

    available = [key for key in DEPTH_MAKERS if key != session.last_shape]
    choice = random.choice(available)
    session.last_shape = choice

    await update.message.reply_text(
        f"🔄 Новая форма: {SHAPE_NAMES[choice]}\n🔮 Генерирую..."
    )

    result_bytes, hint_bytes = await asyncio.to_thread(
        render_shape_images,
        session.pattern_tile.copy(),
        choice,
    )
    await send_generated_images(
        update.message,
        result_bytes,
        hint_bytes,
        caption=(
            f"🔮 Готово! Спрятано: {SHAPE_NAMES[choice]}\n"
            "Расфокусируй взгляд 👀\n\n"
            "by @alyonafrost"
        ),
    )
    return WAITING_PHOTO


async def cancel(update, context):
    sessions.remove(update.effective_user.id)
    await update.message.reply_text("Пока! 🔮")
    return ConversationHandler.END


async def error_handler(update, context):
    logger.error(
        "Unhandled error while processing update",
        exc_info=context.error,
    )
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await message.reply_text(
            "Произошла ошибка. Попробуй ещё раз или отправь /start."
        )
    except Exception:
        logger.exception("Failed to notify user about handler error")


def get_bot_token() -> str:
    token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("BOT_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is required. Add it to Railway Variables."
        )
    return token


def build_application(token: str) -> Application:
    if not token.strip():
        raise ValueError("Bot token must not be empty")

    app = Application.builder().token(token).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("random", random_shape),
            CommandHandler("again", again),
            MessageHandler(filters.PHOTO, receive_photo),
        ],
        states={
            WAITING_PHOTO: [MessageHandler(filters.PHOTO, receive_photo)],
            WAITING_SHAPE: [
                CallbackQueryHandler(shape_chosen),
                MessageHandler(filters.PHOTO, receive_photo),
            ],
            WAITING_TEXT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, receive_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        allow_reentry=True,
        per_message=False,
    )

    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command))
    app.add_error_handler(error_handler)
    return app


def main():
    try:
        token = get_bot_token()
    except RuntimeError as exc:
        logger.critical("%s", exc)
        raise SystemExit(1) from exc

    app = build_application(token)
    logger.info("Magic Eye Bot v3 started")
    app.run_polling()


if __name__ == "__main__":
    main()
