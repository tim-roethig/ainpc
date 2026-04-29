import os
from dotenv import load_dotenv

from npc import NPC

if __name__ == "__main__":
    load_dotenv("../secrets.env")

    npc = NPC(model_api_key=os.environ["API_KEY"], language="german")

    conversation = []
    while True:
        player_input = input()
        conversation = npc.respond_to_text(player_input=player_input, conversation=conversation)
        print(conversation)