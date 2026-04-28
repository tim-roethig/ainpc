class NPC:
    """
    A NPC powered by an omni modal LLM.

    Context:
    The NPC has access to:
    1. A SYSTEM.md file that contains the agents system prompt containing information about the
    world, the location, the character, the voice and the goal of the NPC.
    2. A EVENTS.md file that describes important events that took place in his world.
    3. A mem0 memories database.

    Input:
    A player can talk to the NPC either via text or audio.
    Optionally the NPC consumes an image of the player to make responses more immersive.

    Output:
    The NPC responds with both text and audio. The text should just be the transcript of the audio.
    """
    def __init__(
            self,
            model_api_baseurl: str,
            model_api_key: str,
            npc_directory: str = ".",
    ):
        self.language = language
        self.npc_directory = npc_directory
        self.model_api_baseurl = model_api_baseurl
        self.model_api_key = model_api_key

    def _update_memory(self, conversation: list[dict]) -> None:
        """
        Updates the memory of the NPC based on the conversation and the existing MEMORIES.md.

        :param conversation: The messages of the last completed conversation between the player
        and the NPC. The conversation parameter is a OpenAI compliant messages list.
        :return:
        """
        pass

    def _update_events(self, event: dict[str, str]) -> None:
        """
        Update EVENTS.md with a new event.

        :param event: A dictionary with the keys: time, location, event_description.
        :return:
        """
        pass

    def respond_to_text(
            self,
            player_input: str,
            conversation: list[dict],
            player_image: str = None
    ) -> list[dict]:
        """
        Responds to a text input sent to the NPC.

        :param player_input: The player conversation turn typed in by the player.
        :param conversation: The past conversation between the player and the NPC.
        :param player_image: An optional image of the player.
        :return: The updated conversation, including the NPC's answer.
        """
        pass