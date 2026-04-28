from google import genai

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
            model_api_key: str,
            language: str,
            model_name: str = "models/gemini-3.1-flash-live-preview",
            npc_directory: str = "./test_npc",
    ):
        self.language = language
        self.npc_directory = npc_directory
        self.model_name = model_name
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=model_api_key,
        )

    def _update_memory(self, conversation: list[dict]) -> None:
        """
        Updates the mem0 memories database.

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