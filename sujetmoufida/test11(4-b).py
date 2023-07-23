import random

def generate_key(parts):
    """
    Cette fonction prend une liste de parts et génère une clé aléatoire
    en combinant les parts de manière sécurisée.
    """
    key = ""
    for i in range(len(parts[0])):
        char_set = set(part[i] for part in parts)
        key += random.choice(list(char_set))
    return key
 
 
 
def publish_key(parts, new_parts, chosen_key):
    """
    Cette fonction publie une nouvelle clé en combinant une partie des parts
    originales avec de nouvelles parts. La nouvelle clé est stockée dans un
    fichier ou envoyée à un serveur distant, par exemple.
    """
    combined_parts = parts[:len(parts)//2] + new_parts[len(new_parts)//2:]
    key = generate_key(combined_parts)
    with open("new_key.txt", "w") as f:
        f.write(key)
    print(f"La nouvelle clé ({key}) a été publiée avec succès !")

# Exemple d'utilisation
parts = ["ak", "bk", "a'k", "b'k","a''k ", "b''k "] # Exemple de parts
chosen_key = "s'k" # Exemple de clé choisie par le participant
new_parts = ["uvwxy", "12345", "67890", "abcdef"] # Exemple de nouvelles parts
publish_key(parts, new_parts, chosen_key) # Publier une nouvelle clé en combinant une partie des parts originales avec les nouvelles parts