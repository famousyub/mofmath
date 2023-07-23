def store_parts(parts, file_name):
    """
    Cette fonction stocke l'ensemble des parts dans un fichier texte.
    """
    with open(file_name, "w") as f:
        for part in parts:
            f.write(part + "\n")
    print(f"L'ensemble des parts a été stocké dans le fichier {file_name} !")

# Exemple d'utilisation
parts = ["abcde", "fghij", "klmno", "pqrst"] # Exemple de parts
store_parts(parts, "parts.txt") # Stocker l'ensemble des parts dans un fichier texte



# autre methode pour publier l'esnsemble des parts  cest que Envoyer l'ensemble des parts à un serveur distant 
import requests

def send_parts(parts, url):
    """
    Cette fonction envoie l'ensemble des parts à un serveur distant via une requête HTTP POST.
    """
    data = {"parts": parts}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("L'ensemble des parts a été envoyé avec succès !")
    else:
        print("Erreur lors de l'envoi de l'ensemble des parts...")

# Exemple d'utilisation
parts = ["abcde", "fghij", "klmno", "pqrst"] # Exemple de parts
url = "https://example.com/api/parts" # Exemple d'URL de serveur distant
send_parts(parts, url) # Envoyer l'ensemble des parts à un serveur distant