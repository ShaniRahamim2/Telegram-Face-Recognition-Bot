import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

# States for the bot
STATE_IDLE = 0
STATE_AWAITING_IMAGE = 1
STATE_AWAITING_NAME = 2
STATE_RECOGNIZE_IMAGE = 3

user_states = {}         # maps user_id to state
temp_faces = {}          # temporary image storage per user (for naming)

# Define custom keyboard buttons
keyboard_buttons = [['Add face'], ['Recognize faces'], ['Reset faces']]
keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

# Start command handler – sends the keyboard to the user
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_states[user_id] = STATE_IDLE
    await update.message.reply_text("Choose an action:", reply_markup=keyboard)

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
