import os
from database import register_user

def upload_image(username):
    # Ensure the directory exists
    directory = 'utils/images'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Input for image upload
    image_path = input("Enter the path to the image you want to upload: ")
    
    if os.path.exists(image_path):
        # Save the image with the username as the filename
        new_image_path = os.path.join(directory, f"{username}.jpg")
        os.rename(image_path, new_image_path)
        print(f"Image uploaded successfully to {new_image_path}")
    else:
        print("Image file does not exist. Please try again.")

# Input for the new user registration
username = input("Enter a new username: ")
password = input("Enter a new password: ")

# Register the user
register_user(username, password)

# Upload the image for the registered user
upload_image(username)