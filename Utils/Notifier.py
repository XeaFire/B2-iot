import asyncio
import discord
import queue
import os # Ajouté

# File d'attente thread-safe pour communiquer entre le thread de détection (Flask/OpenCV) 
# et le thread du bot Discord (asyncio).
notification_queue = queue.Queue()

TARGET_USER_ID = 272795289218318336 # L'ID de l'utilisateur à notifier

async def notification_sender_task(client: discord.Client):
    print("Tâche d'envoi de notifications (avec images) démarrée.")
    loop = asyncio.get_running_loop()
    while True:
        image_path_to_delete = None
        try:
            notification_data = await loop.run_in_executor(None, notification_queue.get)
            message_content, image_path = notification_data
            
            print(f"Notification reçue: '{message_content}', Image: {os.path.basename(image_path) if image_path else 'Aucune'}")
            
            if image_path and os.path.exists(image_path):
                image_path_to_delete = image_path
            
            user = await client.fetch_user(TARGET_USER_ID)
            
            if user:
                discord_file = None
                embed = discord.Embed(
                    title="🚨 Mouvement détecté",
                    description=message_content,
                    color=discord.Color.red()
                )
                embed.set_footer(text="Système de détection - Sécurité")
                
                if image_path and os.path.exists(image_path):
                    try:
                        discord_file = discord.File(image_path, filename="capture.jpg")
                        embed.set_image(url="attachment://capture.jpg")
                        print(f"Objet discord.File créé pour {os.path.basename(image_path)}")
                    except Exception as file_error:
                        print(f"Erreur lors de la création de discord.File pour {image_path}: {file_error}")
                        discord_file = None

                await user.send(embed=embed, file=discord_file)
                
                if discord_file:
                    print(f"Notification avec image envoyée à {user.name}")
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
            if 'request entity too large' in str(http_error).lower():
                print("L'image était peut-être trop volumineuse pour Discord.")
        except Exception as e:
            print(f"Erreur inattendue dans notification_sender_task: {e}")
            await asyncio.sleep(5)
        finally:
            if image_path_to_delete:
                try:
                    os.remove(image_path_to_delete)
                except OSError as remove_error:
                    print(f"Erreur lors de la suppression de l'image temporaire {image_path_to_delete}: {remove_error}")
