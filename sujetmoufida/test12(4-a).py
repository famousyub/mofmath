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

# Exemple d'utilisation
parts = ["abcde", "fghij", "klmno", "pqrst"] # Exemple de parts
random_key = generate_key(parts) # Générer une clé aléatoire à partir des parts
print("Clé choisie au hasard   :", random_key)
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

def publish_key(parts, key):
    """
    Cette fonction publie une clé générée à partir des parts.
    La clé est stockée dans un fichier ou envoyée à un serveur distant, par exemple.
    """
    with open("key.txt", "w") as f:
        f.write(key)
    print(f"La clé ({key}) a été publiée avec succès !")

# Exemple d'utilisation
parts = ["abcde", "fghij", "klmno", "pqrst"] # Exemple de parts
key = generate_key(parts) # Générer une clé aléatoire à partir des parts
publish_key(parts, key) # Publier la clé générée