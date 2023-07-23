def find_max_votes(votes):
    """
    Cette fonction prend un dictionnaire de votes et retourne le participant
    avec le maximum de votes. Si plusieurs participants ont le même nombre
    de votes maximum, ils sont renvoyés sous forme de liste.
    """
    max_votes = max(votes.values())
    max_participants = [participant for participant, num_votes in votes.items() if num_votes == max_votes]
    if len(max_participants) == 1:
        return max_participants[0]
    else:
        return max_participants

def disqualify_participants(votes, threshold):
    """
    Cette fonction prend un dictionnaire de votes et un seuil de disqualification,
    et retourne une liste de participants malhonnêtes (ceux qui ont plus de votes que le seuil).
    """
    disqualified = [participant for participant, num_votes in votes.items() if num_votes > threshold]
    return disqualified

# Exemple d'utilisation
votes = {"Participant 1": 10, "Participant 2": 15, "Participant 3": 7, "Participant 4": 15, "Participant 5": 8} # Exemple de votes pour chaque participant
max_votes = find_max_votes(votes) # Trouver les participants avec le maximum de votes
print("Participant(s) avec le maximum de votes :", max_votes)
malicious_participants = disqualify_participants(votes, 12) # Disqualifier les participants qui ont plus de 12 votes
if len(malicious_participants) > 0:
    print("Les participants malhonnêtes sont :", malicious_participants)
else:
    print("Aucun participant malhonnête n'a été identifié.")