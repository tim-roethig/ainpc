import os
from dotenv import load_dotenv

from npc import NPC

if __name__ == "__main__":
    load_dotenv("../secrets.env")

    npc = NPC(model_api_key=os.environ["API_KEY"], language="german")

    npc.respond_to_text()

