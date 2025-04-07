import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Load environment variables from .env file
load_dotenv()


TOKEN = os.getenv("TELEGRAM_TOKEN")

# Define the fixed custom keyboard buttons
keyboard_buttons = [['Hello', 'World'], ['Telegram', 'Bot']]
keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

# Start command handler – sends the keyboard to the user
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose one of the buttons below:", reply_markup=keyboard)

# Message handler – replies with the exact same text the user sends
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text(user_text)

# Main function – sets up and runs the bot
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == '__main__':
    main()
