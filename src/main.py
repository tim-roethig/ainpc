import os
import asyncio
from dotenv import load_dotenv

from npc import NPC


async def main():
    load_dotenv("../secrets.env")

    npc = NPC(api_key=os.environ["API_KEY"])

    async with npc:
        while True:
            # `asyncio.to_thread` keeps the event loop responsive while we
            # wait for keyboard input, so audio playback and websocket
            # frames don't stall behind a blocking `input()` call.
            raw_player_input = await asyncio.to_thread(input, ">> ")
            player_input = raw_player_input.strip()
            if player_input in {"/quit", "/exit", ""}:
                break
            await npc.respond_to_text(player_input)

    # The conversation is over and `npc.transcript` is fully populated.
    npc._update_memory(npc.transcript)


if __name__ == "__main__":
    asyncio.run(main())
