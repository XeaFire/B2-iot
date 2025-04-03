import asyncio
import discord
import queue
import os # Ajouté

# File d'attente thread-safe pour communiquer entre le thread de détection (Flask/OpenCV) 
# et le thread du bot Discord (asyncio).
notification_queue = queue.Queue()

TARGET_USER_ID = 272795289218318336 # L'ID de l'utilisateur à notifier

async def notification_sender_task(client: discord.Client):
    """Tâche asynchrone qui écoute la file d'attente et envoie les DMs avec image."""
    print("Tâche d'envoi de notifications (avec images) démarrée.")
    loop = asyncio.get_running_loop()
    while True:
        image_path_to_delete = None # Garder une trace de l'image à supprimer
        try:
            # Attendre des données de la file d'attente (tuple: message, image_path)
            notification_data = await loop.run_in_executor(None, notification_queue.get)
            message_content, image_path = notification_data # Déballer le tuple
            
            print(f"Notification reçue: '{message_content}', Image: {os.path.basename(image_path) if image_path else 'Aucune'}")
            
            # Marquer l'image pour suppression potentielle après l'envoi
            if image_path and os.path.exists(image_path):
                image_path_to_delete = image_path 
            
            # Récupérer l'objet utilisateur Discord
            user = await client.fetch_user(TARGET_USER_ID)
            
            if user:
                discord_file = None
                if image_path and os.path.exists(image_path):
                    try:
                        # Créer l'objet discord.File
                        discord_file = discord.File(image_path)
                        print(f"Objet discord.File créé pour {os.path.basename(image_path)}")
                    except Exception as file_error:
                        print(f"Erreur lors de la création de discord.File pour {image_path}: {file_error}")
                        discord_file = None # Ne pas essayer d'envoyer un fichier invalide
                        # L'image sera quand même supprimée dans finally

                # Envoyer le message privé (avec ou sans fichier)
                await user.send(content=message_content, file=discord_file)
                
                if discord_file:
                    print(f"Notification avec image envoyée à {user.name}")
                    # Fermer explicitement le fichier après l'envoi peut aider avec les handleurs
                    # discord_file.close()
                else:
                    print(f"Notification texte envoyée à {user.name} (pas d'image ou erreur fichier)")
            else:
                print(f"Erreur: Impossible de trouver l'utilisateur avec l'ID {TARGET_USER_ID}")
            
        except discord.NotFound:
            print(f"Erreur: Utilisateur {TARGET_USER_ID} non trouvé.")
        except discord.Forbidden:
            print(f"Erreur: Le bot n'a pas la permission d'envoyer un message/fichier à l'utilisateur {TARGET_USER_ID}.")
            print("Vérifiez que l'utilisateur partage un serveur avec le bot ou autorise les DMs.")
        except discord.HTTPException as http_error:
             print(f"Erreur HTTP Discord lors de l'envoi de la notification: {http_error}")
             # Si l'erreur est liée à la taille du fichier, on pourrait le notifier
             if 'request entity too large' in str(http_error).lower():
                 print("L'image était peut-être trop volumineuse pour Discord.")
        except Exception as e:
            print(f"Erreur inattendue dans notification_sender_task: {e}")
            await asyncio.sleep(5)
        finally:
            # Supprimer l'image temporaire après l'envoi (ou tentative d'envoi)
            if image_path_to_delete:
                try:
                    os.remove(image_path_to_delete)
                    # print(f"Image temporaire supprimée: {os.path.basename(image_path_to_delete)}") # Log optionnel
                except OSError as remove_error:
                    print(f"Erreur lors de la suppression de l'image temporaire {image_path_to_delete}: {remove_error}") 