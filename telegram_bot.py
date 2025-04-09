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
    
# Message handler – replies to user messages based on their state
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    # User selected 'Add face' - set state to await image upload
    if text == "Add face":
        user_states[user_id] = STATE_AWAITING_IMAGE
        await update.message.reply_text("Upload an image with a single face.")

    # User selected 'Recognize faces' - set state to await image for recognition
    elif text == "Recognize faces":
        user_states[user_id] = STATE_RECOGNIZE_IMAGE
        await update.message.reply_text("Upload an image and I will try to recognize the faces.")

    # User selected 'Reset faces' - delete all stored known faces
    elif text == "Reset faces":
        for f in os.listdir("known_faces"):
            os.remove(os.path.join("known_faces", f))
        await update.message.reply_text("All faces have been deleted.", reply_markup=keyboard)

    # If user is expected to enter a name for a previously uploaded face
    elif user_states.get(user_id) == STATE_AWAITING_NAME:
        image_path = temp_faces.pop(user_id, None)

        if image_path:
            person_name = text.strip()
            os.rename(image_path, f"known_faces/{person_name}.jpg")
            await update.message.reply_text(f"Face saved as {person_name}.", reply_markup=keyboard)

            # Reset state to idle after saving the face
            user_states[user_id] = STATE_IDLE
        else:
            # If something went wrong (e.g. image not found)
            await update.message.reply_text("Something went wrong. Please try again.")

# Main function – sets up and runs the bot
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running... Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == '__main__':
    main()
