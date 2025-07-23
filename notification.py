from pushbullet import Pushbullet
from pushbullet import Pushbullet
from pushbullet import InvalidKeyError

def send_push_notification(title, message):
    # Pushbullet API key
    api_key = "o.UeYzG4hVGPQOm8eICahMerGEc5FSjHXL"

    try:
        # Create a Pushbullet object
        pb = Pushbullet(api_key)

        # Send push notification
        push = pb.push_note(title, message)

        print(f"Push notification sent. Push ID: {push['iden']}")

    except Exception as e:
        print(f"Error sending push notification: {e}")

# Example usage:
# send_push_notification("Livestock Health Alert", "Recommended action: Isolate the animal and consult a veterinarian.")
