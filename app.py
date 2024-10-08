import hashlib
import random
import base64

def generate_salt(length=16):
    return ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(length))

def hash_key(key, salt):
    return hashlib.sha512((key + salt).encode()).hexdigest()

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
    step2 = shift_characters(step1, 7)
    step3 = reverse_text(step2)
    step4 = base64_encode(step3)
    step5 = xor_encrypt_decrypt(step4, hashed_key[:len(step4)])
    step6 = shift_characters(step5, 13)
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
    step6 = remove_noise(step7)
    step5 = shift_characters(step6, -13)
    step4 = xor_encrypt_decrypt(step5, hashed_key[:len(step5)])
    step3 = base64_decode(step4)
    step2 = reverse_text(step3)
    step1 = shift_characters(step2, -7)
    decrypted_password = xor_encrypt_decrypt(step1, hashed_key[:len(step1)])

    return decrypted_password

def main():
    action = input("Would you like to encrypt a password or decrypt a password? (Enter 'encrypt' or 'decrypt'): ").strip().lower()

    if action == 'encrypt':
        password = input("Enter a password to encrypt: ")
        key = input("Enter a key for encryption: ")
        encrypted = complex_encrypt(password, key)
        print("Encrypted password:", encrypted)

    elif action == 'decrypt':
        encrypted_password = input("Enter the encrypted password: ")
        key = input("Enter the key for decryption: ")
        decrypted = complex_decrypt(encrypted_password, key)
        print("Decrypted password:", decrypted)

    else:
        print("Invalid")

main()
