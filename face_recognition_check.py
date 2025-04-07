import face_recognition

# Load the images from the 'data' folder
image1 = face_recognition.load_image_file("data/face1.png")
image2 = face_recognition.load_image_file("data/face2.png")
image3 = face_recognition.load_image_file("data/face3.png")

# Get the face encodings (vector representations)
enc1 = face_recognition.face_encodings(image1)[0]
enc2 = face_recognition.face_encodings(image2)[0]
enc3 = face_recognition.face_encodings(image3)[0]

# Compare image1 to image2 and image3
distance_1_2 = face_recognition.face_distance([enc2], enc1)[0]
distance_1_3 = face_recognition.face_distance([enc3], enc1)[0]

# Print distances
print(f"Distance from face1 to face2: {distance_1_2}")
print(f"Distance from face1 to face3: {distance_1_3}")

# Decide which is more similar
if distance_1_2 < distance_1_3:
    print("Face1 is more similar to Face2.")
else:
    print("Face1 is more similar to Face3.")
