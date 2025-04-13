import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import face_recognition
import cv2

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tempfile

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
keyboard_buttons = [['Add face'], ['Recognize faces'], ['Reset faces'], ['Similar celebs'], ['Map'], ['Instructions']]
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

# Loads celebrity face encodings, names, and image paths from the celebs folder (only one image per celeb)
def load_celeb_encodings(celeb_dir="celebs"):
    encodings = [] 
    names = []    
    paths = []     

    # Loop through each celeb directory inside the main folder
    for celeb_name in os.listdir(celeb_dir):
        celeb_path = os.path.join(celeb_dir, celeb_name)

        # Ensure it's a directory (not a file)
        if os.path.isdir(celeb_path):
            # Get list of all image files in the celeb directory
            image_files = [f for f in os.listdir(celeb_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Skip if no images found
            if not image_files:
                continue
                
            # Take only the first image file for each celebrity
            first_image = image_files[0]
            full_path = os.path.join(celeb_path, first_image)

            # Load image and extract face encodings
            try:
                image = face_recognition.load_image_file(full_path)
                face_enc = face_recognition.face_encodings(image)

                if face_enc:
                    encodings.append(face_enc[0])
                    names.append(celeb_name)
                    paths.append(full_path)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")

    return encodings, names, paths

def create_face_similarity_map():
    # Load all known faces (uploaded by users)
    known_encodings, known_names = get_known_faces()
    
    # Prepare structure for t-SNE mapping
    all_encodings = []
    all_names = []
    all_images = []
    
    # Add known faces
    from PIL import Image

    for encoding, name in zip(known_encodings, known_names):
        all_encodings.append(encoding)
        all_names.append(name)
        # Load user image
        try:
            user_img = face_recognition.load_image_file(f"known_faces/{name}.jpg")
            all_images.append(Image.fromarray(user_img))
        except Exception:
            # Create a placeholder if image can't be loaded
            all_images.append(Image.new("RGB", (45, 45), color=(150, 150, 150)))

    # Load celebrity faces
    celeb_dir = "celebs"
    for celeb_name in os.listdir(celeb_dir):
        celeb_path = os.path.join(celeb_dir, celeb_name)
        
        # Ensure it's a directory (not a file)
        if os.path.isdir(celeb_path):
            for file in os.listdir(celeb_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(celeb_path, file)
                    
                    try:
                        # Load image and extract face encodings
                        image = face_recognition.load_image_file(full_path)
                        face_enc = face_recognition.face_encodings(image)
                        
                        if face_enc:
                            all_encodings.append(face_enc[0])
                            all_names.append(celeb_name)
                            all_images.append(Image.fromarray(image))
                    except Exception:
                        continue
    
    # If there are no faces to map, return an error
    if len(all_encodings) < 2:
        return None, "Not enough faces in the system. Please add at least two faces first."
    
    # Convert to numpy array for t-SNE
    face_encodings_array = np.array(all_encodings)
    
    # Dimensionality reduction using t-SNE
    perplexity = min(30, len(face_encodings_array) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(face_encodings_array)
    
    # Normalize coordinates to 0-1 range for easier plotting
    min_x, min_y = np.min(reduced, axis=0)
    max_x, max_y = np.max(reduced, axis=0)
    norm_x = (reduced[:, 0] - min_x) / (max_x - min_x)
    norm_y = (reduced[:, 1] - min_y) / (max_y - min_y)
    norm_y *= 0.9  # Shift everything down to make space for title
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_title("Face Similarity Map (similar faces are positioned closer)", fontsize=14, pad=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add face thumbnails and labels to the plot
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    for x, y, img, label in zip(norm_x, norm_y, all_images, all_names):
        # Create thumbnail image
        try:
            # Extract face from the image for cleaner display
            face_locations = face_recognition.face_locations(np.array(img))
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_img = img.crop((left, top, right, bottom))
                thumb = face_img.resize((45, 45))
            else:
                thumb = img.resize((45, 45))
                
            # Add image at the computed location
            im = OffsetImage(thumb, zoom=1)
            ab = AnnotationBbox(im, (x, y), frameon=True, pad=0.3)
            ax.add_artist(ab)
            
            # Add name label
            ax.text(x, y - 0.035, label, fontsize=6, ha='center', va='top')
        except Exception as e:
            # If image processing fails, add text label only
            ax.text(x, y, f"{label}", ha='center', fontsize=9)
    
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        plt.savefig(temp_file.name, bbox_inches='tight', dpi=300)
        plt.close()
        return temp_file.name, None

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

    # Handles the 'Similar celebs' button – sets state and prompts user to upload a face image
    elif text == "Similar celebs":
        user_states[user_id] = STATE_SIMILAR_CELEB
        await update.message.reply_text("Upload me a picture of a single person and I will find the most similar celeb.")

    elif text == "Map":
        await handle_map_button(update, context)

    # User selected 'Instructions' - send instructions 
    # feacher that added by me
    elif text == "Instructions":
        instructions = (
            "*Instructions*\n\n"
            "*Add face* – Upload a face and give it a name so I can recognize it later.\n"
            "*Recognize faces* – Upload a photo, and I’ll label known faces in it.\n"
            "*Reset faces* – Delete all the user faces you've added.\n"
            "*Similar celebs* – Send a face, and I’ll find the most similar celebrity.\n"
            "*Map* – I’ll generate a face similarity map for all users and celebs.\n"
            "*Instructions* – Show this help message.\n"
        )
        await update.message.reply_markdown(instructions, reply_markup=keyboard)

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

# handle Map button press
async def handle_map_button(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text("Generating face similarity map... This may take a few seconds.")
    
    map_path, error_message = create_face_similarity_map()
    
    if error_message:
        await update.message.reply_text(error_message, reply_markup=keyboard)
        return
    
    await update.message.reply_photo(
        photo=open(map_path, 'rb'),
        caption="Face similarity map - similar faces are positioned closer to each other",
        reply_markup=keyboard
    )
    
    # Clean up temporary file
    os.remove(map_path)

# Main function – sets up and runs the bot
def main():
    os.makedirs("known_faces", exist_ok=True)
    os.makedirs("celebs", exist_ok=True)
    
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
