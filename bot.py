import discord
import os
import asyncio
from Utils.Notifier import notification_queue, notification_sender_task

DISCORD_TOKEN = ""

# Définir les intents nécessaires pour le bot
# Vous devrez peut-être ajuster cela en fonction des besoins de votre bot
intents = discord.Intents.default()
intents.message_content = True # Nécessaire pour lire le contenu des messages
# Intents supplémentaires pourraient être nécessaires pour fetch_user
intents.members = True 

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    """Affiche un message lorsque le bot est connecté et prêt."""
    print(f'Connecté en tant que {client.user}')
    print('------')
    # Démarrer la tâche d'envoi de notifications en arrière-plan
    print("Lancement de la tâche d'envoi de notifications...")
    asyncio.create_task(notification_sender_task(client))

@client.event
async def on_message(message):
    """Gère les messages reçus."""
    # Empêche le bot de répondre à ses propres messages
    if message.author == client.user:
        return

def run_bot():
    """Fonction pour démarrer le bot."""
    if not DISCORD_TOKEN:
        print("Erreur : Le token Discord n'est pas défini.")
        # Modifier pour ne pas utiliser .env ici car le token est hardcodé
        # print("Veuillez créer un fichier .env et y ajouter DISCORD_TOKEN=VOTRE_TOKEN") 
        return

    try:
        print("Démarrage du bot Discord...")
        # Utilise run() qui gère la boucle d'événements asyncio
        client.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        print("Erreur de connexion : Token Discord invalide.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue lors du lancement du bot : {e}")

# La section if __name__ == '__main__': peut rester pour des tests directs du bot
if __name__ == '__main__':
    # Permet de lancer le bot directement depuis ce fichier pour des tests
    run_bot() 