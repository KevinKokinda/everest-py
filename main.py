import hashlib
import random
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import os

def generate_salt(length=16):
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(length))

def hash_key(key, salt):
    return hashlib.sha512((key + salt).encode()).hexdigest()

def rotate_key(key, rotation_amount):
    return key[-rotation_amount:] + key[:-rotation_amount]

def xor_encrypt_decrypt(text, key):
    return ''.join(chr(ord(c) ^ ord(k)) for c, k in zip(text, key))

def shift_characters(text, shift_amount):
    return ''.join(chr((ord(c) + shift_amount) % 256) for c in text)

def reverse_text(text):
    return text[::-1]

def base64_encode(text):
    return base64.b64encode(text.encode()).decode()

def base64_decode(text):
    return base64.b64decode(text.encode()).decode()

def generate_random_key(length):
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(length))

def add_noise(text, noise_factor=2):
    noisy_text = ""
    for char in text:
        noisy_text += char + ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(noise_factor))
    return noisy_text

def remove_noise(noisy_text, noise_factor=2):
    return noisy_text[::noise_factor + 1]

def complex_encrypt(password, key):
    salt = generate_salt()
    hashed_key = hash_key(key, salt)
    
    step1 = xor_encrypt_decrypt(password, hashed_key[:len(password)])
    hashed_key = rotate_key(hashed_key, 3)
    
    step2 = shift_characters(step1, 7)
    hashed_key = rotate_key(hashed_key, 5)
    
    step3 = reverse_text(step2)
    hashed_key = rotate_key(hashed_key, 7)
    
    step4 = base64_encode(step3)
    hashed_key = rotate_key(hashed_key, 11)
    
    step5 = xor_encrypt_decrypt(step4, hashed_key[:len(step4)])
    hashed_key = rotate_key(hashed_key, 13)
    
    step6 = shift_characters(step5, 13)
    hashed_key = rotate_key(hashed_key, 17)
    
    step7 = add_noise(step6)
    step8 = base64_encode(step7)
    step9 = xor_encrypt_decrypt(step8, generate_random_key(len(step8)))
    
    encrypted_password = shift_characters(step9, 19) + salt
    return encrypted_password

def complex_decrypt(encrypted_password, key):
    salt = encrypted_password[-16:]
    encrypted_core = encrypted_password[:-16]
    
    hashed_key = hash_key(key, salt)
    
    step9 = shift_characters(encrypted_core, -19)
    step8 = xor_encrypt_decrypt(step9, generate_random_key(len(step9)))
    
    step7 = base64_decode(step8)
    hashed_key = rotate_key(hashed_key, 17)
    
    step6 = remove_noise(step7)
    hashed_key = rotate_key(hashed_key, 13)
    
    step5 = shift_characters(step6, -13)
    hashed_key = rotate_key(hashed_key, 11)
    
    step4 = xor_encrypt_decrypt(step5, hashed_key[:len(step5)])
    hashed_key = rotate_key(hashed_key, 7)
    
    step3 = base64_decode(step4)
    hashed_key = rotate_key(hashed_key, 5)
    
    step2 = reverse_text(step3)
    hashed_key = rotate_key(hashed_key, 3)
    
    step1 = shift_characters(step2, -7)
    decrypted_password = xor_encrypt_decrypt(step1, hashed_key[:len(step1)])
    
    return decrypted_password

def text_to_bin(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary):
    binary_values = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(b, 2)) for b in binary_values)

def encode_image(image_path, text, output_path):
    image = Image.open(image_path)
    binary_text = text_to_bin(text) + '1111111111111110'
    binary_index = 0
    
    pixels = list(image.getdata())
    new_pixels = []

    for pixel in pixels:
        new_pixel = []
        for value in pixel:
            if binary_index < len(binary_text):
                new_pixel.append(value & ~1 | int(binary_text[binary_index]))
                binary_index += 1
            else:
                new_pixel.append(value)
        new_pixels.append(tuple(new_pixel))

    image.putdata(new_pixels)
    image.save(output_path)

def decode_image(image_path):
    image = Image.open(image_path)
    pixels = list(image.getdata())
    binary_text = ''

    for pixel in pixels:
        for value in pixel:
            binary_text += str(value & 1)

    delimiter_index = binary_text.find('1111111111111110')
    binary_text = binary_text[:delimiter_index]

    return bin_to_text(binary_text)

def complex_encrypt_with_steg(password, key, image_path, output_image):
    encrypted_password = complex_encrypt(password, key)
    encode_image(image_path, encrypted_password, output_image)
    return output_image

def complex_decrypt_with_steg(image_path, key):
    encrypted_password = decode_image(image_path)
    decrypted_password = complex_decrypt(encrypted_password, key)
    return decrypted_password

def create_password_strength_model():
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(5,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def password_to_features(password):
    length = len(password)
    digits = sum(c.isdigit() for c in password)
    uppercase = sum(c.isupper() for c in password)
    lowercase = sum(c.islower() for c in password)
    special = sum(not c.isalnum() for c in password)
    return np.array([length, digits, uppercase, lowercase, special])

def train_password_strength_model():
    passwords = ["password123!", "SecurePa$$", "123456", "admin", "StrongPassword!"]
    labels = [0, 1, 0, 0, 1]
    features = np.array([password_to_features(pw) for pw in passwords])
    labels = np.array(labels)
    model = create_password_strength_model()
    model.fit(features, labels, epochs=10)
    return model

password_strength_model = train_password_strength_model()

def predict_password_strength(password):
    features = password_to_features(password).reshape(1, -1)
    strength = password_strength_model.predict(features)[0][0]
    return "Strong" if strength > 0.5 else "Weak"

def save_key(key, filename):
    with open(filename, 'w') as f:
        f.write(base64_encode(key))
    print(f"Key saved to {filename}")

def load_key(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            key = base64_decode(f.read())
        print(f"Key loaded from {filename}")
        return key
    else:
        print(f"Key file {filename} does not exist.")
        return None

def main():
    action = input("Would you like to encrypt a password or decrypt a password? (Enter 'encrypt' or 'decrypt'): ").strip().lower()
    if action == 'encrypt':
        password = input("Enter a password to encrypt: ")
        key = input("Enter a key for encryption: ")
        encryption_method = input("Choose encryption method ('standard' or 'steganography'): ").strip().lower()
        save_key_option = input("Would you like to save the key? (yes/no): ").strip().lower()
        if save_key_option == 'yes':
            key_filename = input("Enter the filename to save the key: ").strip()
            save_key(key, key_filename)
        if encryption_method == 'standard':
            strength = predict_password_strength(password)
            print(f"Password strength: {strength}")
            encrypted = complex_encrypt(password, key)
            print("Encrypted password:", encrypted)
        elif encryption_method == 'steganography':
            image_path = input("Enter the path of the image to hide the password in: ")
            output_image = input("Enter the path to save the output image: ")
            strength = predict_password_strength(password)
            print(f"Password strength: {strength}")
            output = complex_encrypt_with_steg(password, key, image_path, output_image)
            print(f"Password encrypted and hidden in image: {output}")
        else:
            print("Invalid encryption method")
    elif action == 'decrypt':
        load_key_option = input("Would you like to load a saved key? (yes/no): ").strip().lower()
        if load_key_option == 'yes':
            key_filename = input("Enter the filename to load the key from: ").strip()
            key = load_key(key_filename)
            if key is None:
                print("Decryption aborted.")
                return
        else:
            key
