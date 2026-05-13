"""NPC backed by the Gemini Live API for real-time voice conversations."""

import asyncio
import json
import yaml
import sounddevice
from google import genai


GIVE_ITEM_TOOL = {
    "function_declarations": [
        {
            "name": "give_item",
            "description": "Give a specific item to the player. Call this when the NPC decides to hand something over.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "item_name": {
                        "type": "STRING",
                        "description": "Short name of the item being given, e.g. 'iron key'.",
                    },
                },
                "required": ["item_name"],
            },
        }
    ]
}

MEMORY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "memories": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
}


class NPC:
    """
    A non-player character powered by the Gemini Live API.

    The NPC is used as an async context manager. Entering the context opens a
    long-lived websocket session to the model and starts an audio playback
    stream; exiting closes both. While inside the context, the caller invokes
    `respond_to_text` repeatedly, once per player utterance. The full ordered
    transcript of the conversation is available on `self.transcript` after the
    context exits, so it can be fed into the long-term-memory updater.

    Example:
        npc = NPC(api_key=...)
        async with npc:
            await npc.respond_to_text("Hello, who are you?")
            await npc.respond_to_text("Tell me about this place.")
        npc._update_memory(npc.transcript)
    """

    # Sentinel pushed onto the audio queue to signal the playback task that no
    # more chunks will arrive and it should exit cleanly.
    _AUDIO_STREAM_FINISHED = object()

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/gemini-3.1-flash-live-preview",
        agent_yaml_path: str = "./test_npc/AGENT.yaml",
    ):
        self.agent_yaml_path = agent_yaml_path
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        # Per-conversation state, populated in `__aenter__`. `None` outside an
        # `async with` block so misuse fails fast with a clear AttributeError.
        self.live_session = None
        self._live_session_context_manager = None
        self.audio_output_stream: sounddevice.RawOutputStream | None = None
        self.audio_playback_queue: asyncio.Queue | None = None
        self.audio_playback_task: asyncio.Task | None = None

        self.transcript: str = ""

        with open(self.agent_yaml_path, "r", encoding="utf-8") as yaml_file:
            self.agent_data = yaml.safe_load(yaml_file) or {}

    def _create_system_prompt(self) -> str:
        """Render the YAML persona into a Markdown system prompt."""
        rendered_sections: list[str] = []
        for section_key, section_value in self.agent_data.items():
            section_lines = [f"### {section_key}"]
            if isinstance(section_value, dict):
                section_description = section_value.get("description")
                if section_description:
                    section_lines.append(str(section_description).strip())
                section_content = section_value.get("content")
                if section_content is not None:
                    section_lines.append(str(section_content).strip())
            rendered_sections.append("\n".join(section_lines))
        return "\n\n".join(rendered_sections)

    def _t2t_generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "thinking_config": {"thinking_level": "MINIMAL"},
                "response_mime_type": "application/json",
                "response_schema": MEMORY_RESPONSE_SCHEMA,
            },
        )
        return response.text

    def _update_memory(self, transcript: str) -> None:
        existing_memories = self.agent_data["memory"]["content"] or []
        old_memory = (
            "\n".join(f"- {item}" for item in existing_memories)
            if existing_memories
            else "No memories yet. You meet the person for the first time."
        )

        update_prompt = f"""Your task is to update stored memories of interactions between you (a NPC) and a player.

Background information about you:
Name: {self.agent_data["name"]}
Your character: {self.agent_data["character"]["content"]}
The World you live in: {self.agent_data["world"]["content"]}
The location you are at the moment: {self.agent_data["location"]["content"]}
Your main goat at the moment: {self.agent_data["goal"]["content"]}

These are your existing memories:
{old_memory}

This is the last conversation you had with the Player:
{transcript.strip()}

Guidelines:
- Return a list of memories.
- You can add new items and/or update existing items.
- Only store memories that could become important in later conversations.
- Do not maintain more than 20 memories, delete insignificant or group memories together."""

        new_memory = json.loads(self._t2t_generate(prompt=update_prompt))["memories"]

        self.agent_data["memory"]["content"] = new_memory
        with open(self.agent_yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(self.agent_data, yaml_file, default_flow_style=False, sort_keys=False)

    def give_item(self, item_name: str) -> dict:
        """Hand an item to the player. Stub for now."""
        print(f"gave {item_name=}")
        return {"status": "ok", "item_name": item_name}

    async def __aenter__(self) -> "NPC":
        self.transcript = ""

        live_session_config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": "Iapetus"},
                },
            },
            # Ask the server to send a text transcript of its own audio output
            # so we can populate `self.transcript` and echo it to stdout.
            "output_audio_transcription": {},
            "thinking_config": {"thinking_level": "minimal"},
            "system_instruction": self._create_system_prompt(),
            "tools": [GIVE_ITEM_TOOL],
        }

        # `client.aio.live.connect(...)` returns an async context manager. We
        # don't use `async with` here because we need to hold the session open
        # across multiple method calls; instead we drive its lifecycle manually
        # via `__aenter__` / `__aexit__`.
        self._live_session_context_manager = self.client.aio.live.connect(
            model=self.model_name,
            config=live_session_config,
        )
        self.live_session = await self._live_session_context_manager.__aenter__()

        # Gemini Live emits 24 kHz mono 16-bit PCM.
        self.audio_output_stream = sounddevice.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
        )
        self.audio_output_stream.start()

        # A queue decouples receive() (which must not block) from the speaker
        # write (which is blocking I/O).
        self.audio_playback_queue = asyncio.Queue()

        async def play_audio_chunks_until_finished() -> None:
            while True:
                next_audio_chunk = await self.audio_playback_queue.get()
                if next_audio_chunk is self._AUDIO_STREAM_FINISHED:
                    return
                # Run the blocking PortAudio write on a worker thread so the
                # event loop stays free to handle incoming websocket frames.
                await asyncio.to_thread(
                    self.audio_output_stream.write,
                    next_audio_chunk,
                )

        self.audio_playback_task = asyncio.create_task(play_audio_chunks_until_finished())
        return self

    async def __aexit__(self, exception_type, exception_value, exception_traceback) -> None:
        # Order matters: drain queued audio first (so the user hears the tail
        # of the last reply), then close the websocket, then the speaker.
        try:
            self.audio_playback_queue.put_nowait(self._AUDIO_STREAM_FINISHED)
            await self.audio_playback_task
        finally:
            try:
                await self._live_session_context_manager.__aexit__(
                    exception_type,
                    exception_value,
                    exception_traceback,
                )
            finally:
                # `stop()` blocks until the speaker buffer empties.
                await asyncio.to_thread(self.audio_output_stream.stop)
                self.audio_output_stream.close()

    async def respond_to_text(self, user_text: str) -> str:
        """Send one player utterance and play the NPC's spoken reply."""
        await self.live_session.send_realtime_input(text=user_text)

        print(f"user: {user_text}")
        print("npc: ", end="", flush=True)

        response_text_chunks: list[str] = []
        async for live_server_message in self.live_session.receive():
            if live_server_message.tool_call:
                function_responses = []
                for function_call in live_server_message.tool_call.function_calls:
                    if function_call.name == "give_item":
                        result = self.give_item(**(function_call.args or {}))
                        function_responses.append({
                            "id": function_call.id,
                            "name": function_call.name,
                            "response": result,
                        })
                if function_responses:
                    await self.live_session.send_tool_response(
                        function_responses=function_responses,
                    )
                continue

            server_content = live_server_message.server_content
            if not server_content:
                continue

            if server_content.output_transcription and server_content.output_transcription.text:
                transcript_text_chunk = server_content.output_transcription.text
                response_text_chunks.append(transcript_text_chunk)
                print(transcript_text_chunk, end="", flush=True)

            if server_content.model_turn and server_content.model_turn.parts:
                for model_turn_part in server_content.model_turn.parts:
                    if model_turn_part.inline_data and model_turn_part.inline_data.data:
                        # Non-blocking enqueue keeps this receive loop fast;
                        # falling behind here causes websocket backpressure.
                        self.audio_playback_queue.put_nowait(model_turn_part.inline_data.data)

            if server_content.turn_complete:
                print()
                break

        full_response_text = "".join(response_text_chunks)
        self.transcript += f"Player\n{user_text}\n\n"
        self.transcript += f"{self.agent_data['name']}\n{full_response_text}\n\n"
        return full_response_text
