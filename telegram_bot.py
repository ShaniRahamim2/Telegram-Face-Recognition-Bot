import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import face_recognition
import cv2

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

# States for the bot
STATE_IDLE = 0
STATE_AWAITING_IMAGE = 1
STATE_AWAITING_NAME = 2
STATE_RECOGNIZE_IMAGE = 3
STATE_SIMILAR_CELEB = 4

user_states = {}         # maps user_id to state
temp_faces = {}          # temporary image storage per user (for naming)

# Define custom keyboard buttons
keyboard_buttons = [['Add face'], ['Recognize faces'], ['Reset faces'], ['Similar celebs']]
keyboard = ReplyKeyboardMarkup(keyboard_buttons, resize_keyboard=True)

# Load face encodings and names from known_faces folder
def get_known_faces():
    encodings = []
    names = []

    for file in os.listdir("known_faces"):
        image = face_recognition.load_image_file(f"known_faces/{file}")
        enc = face_recognition.face_encodings(image)

        if enc:
            encodings.append(enc[0])
            names.append(file.split(".")[0])

    return encodings, names

# Loads all celebrity face encodings, names, and image paths from the celebs folder
def load_celeb_encodings(celeb_dir="celebs"):
    encodings = [] 
    names = []    
    paths = []     

    # Loop through each celeb directory inside the main folder
    for celeb_name in os.listdir(celeb_dir):
        celeb_path = os.path.join(celeb_dir, celeb_name)

        # Ensure it's a directory (not a file)
        if os.path.isdir(celeb_path):
            for file in os.listdir(celeb_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(celeb_path, file)

                    # Load image and extract face encodings
                    image = face_recognition.load_image_file(full_path)
                    face_enc = face_recognition.face_encodings(image)

                    if face_enc:
                        encodings.append(face_enc[0])
                        names.append(celeb_name)
                        paths.append(full_path)

    return encodings, names, paths

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
        await update.message.reply_text("All faces have been deleted.", reply_markup=keyboard)ֿ

    # Handles the 'Similar celebs' button – sets state and prompts user to upload a face image
    elif text == "Similar celebs":
        user_states[user_id] = STATE_SIMILAR_CELEB
        await update.message.reply_text("Upload me a picture of a single person and I will find the most similar celeb.")

    # If user is expected to enter a name for a previously uploaded face
    elif user_states.get(user_id) == STATE_AWAITING_NAME:
        image_path = temp_faces.pop(user_id, None)

        if image_path:
            person_name = text.strip()
            os.rename(image_path, f"known_faces/{person_name}.jpg")
            await update.message.reply_text(f"Great. I will now remember this face", reply_markup=keyboard)

            # Reset state to idle after saving the face
            user_states[user_id] = STATE_IDLE
        else:
            # If something went wrong (e.g. image not found)
            await update.message.reply_text("Something went wrong. Please try again.")

# processes user-uploaded photos for adding or recognizing faces 
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_states.get(user_id, STATE_IDLE)

    # Get the highest resolution photo sent by the user
    photo = await update.message.photo[-1].get_file()
    temp_path = f"temp_{user_id}.jpg"
    await photo.download_to_drive(temp_path)

    # Load the image and extract face encodings
    image = face_recognition.load_image_file(temp_path)
    encodings = face_recognition.face_encodings(image)

    # Handle 'Add face' mode
    if state == STATE_AWAITING_IMAGE:
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly ONE face.")
            os.remove(temp_path)
        else:
            temp_faces[user_id] = temp_path
            user_states[user_id] = STATE_AWAITING_NAME
            await update.message.reply_text("Great! What's the name of this person?")

    # Handle 'Recognize faces' mode
    elif state == STATE_RECOGNIZE_IMAGE:
        known_encodings, known_names = get_known_faces()
        face_locations = face_recognition.face_locations(image)
        found_names = []

        image_cv = cv2.imread(temp_path)

        for location, face_enc in zip(face_locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, face_enc)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]

            found_names.append(name)

            # Draw box and name
            top, right, bottom, left = location
            cv2.rectangle(image_cv, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image_cv, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        result_path = f"recognized_{user_id}.jpg"
        cv2.imwrite(result_path, image_cv)

        caption = "Faces found: " + ", ".join(found_names) if found_names else "I don’t recognize anyone."
        await update.message.reply_photo(photo=open(result_path, 'rb'), caption=caption, reply_markup=keyboard)

        # Cleanup and reset state
        os.remove(temp_path)
        os.remove(result_path)
        user_states[user_id] = STATE_IDLE

        # Handle 'Similar celebs' mode
    elif state == STATE_SIMILAR_CELEB:
        # Make sure exactly one face is detected
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly ONE face.")
            os.remove(temp_path)
            return

        # Get the encoding of the uploaded face
        user_encoding = encodings[0]

        # Load all known celeb encodings from the celebs folder
        celeb_encodings, celeb_names, celeb_image_paths = load_celeb_encodings()

        if not celeb_encodings:
            await update.message.reply_text("No celebrity data found.")
            os.remove(temp_path)
            return

        # Compare user's face to all celebrity faces
        distances = face_recognition.face_distance(celeb_encodings, user_encoding)
        best_match_index = distances.argmin()
        best_name = celeb_names[best_match_index]
        best_image_path = celeb_image_paths[best_match_index]

        # Send back the most similar celebrity photo
        await update.message.reply_photo(
            photo=open(best_image_path, 'rb'),
            caption=f"The celeb that the person is most similar to is {best_name}.",
            reply_markup=keyboard
        )

        # Clean up and reset state
        os.remove(temp_path)
        user_states[user_id] = STATE_IDLE


# Main function – sets up and runs the bot
def main():
    os.makedirs("known_faces", exist_ok=True)

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
