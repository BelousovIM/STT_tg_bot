from pathlib import Path

import aiohttp
from aiogram import Bot, Dispatcher, executor, types

from stt import BOT_TOKEN, URL

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot=bot)


async def handle_file(file: types.File, file_path: Path):
    async with aiohttp.ClientSession(URL) as session:
        await bot.download_file(file_path=file.file_path, destination=file_path)
        with open(file_path, "rb") as voice_file:
            files = [("files", voice_file)]
            async with session.post("/voices", data=files) as resp:
                return await resp.text()


@dp.message_handler(content_types=[types.ContentType.VOICE])
async def voice_message_handler(message: types.Message):
    """
    This handler will be called when user sends voice message
    """
    voice = await message.voice.get_file()
    path = Path("data/audio")
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / voice.file_id

    response_text = await handle_file(file=voice, file_path=file_path)
    await message.reply(response_text)


@dp.message_handler(commands=["start", "restart"])
async def voice_message_handler(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/restart` command
    """
    await message.reply(
        """
        :)
        Hi! I'm SttBot and I can help you to transcribe your audio! 
        How? Just send me a voice message in English and look...
        """
    )


@dp.message_handler(commands=["help"])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/help` command
    """
    await message.reply(
        """
        You can control me by sending these commands:
        /start 
        /restart
        """
    )


def main():
    executor.start_polling(dp)


if __name__ == "__main__":
    main()
