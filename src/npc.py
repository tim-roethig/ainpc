import asyncio
import json
import yaml
import sounddevice
from google import genai
from google.genai import types


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

    # Sentinel object pushed onto the audio queue to signal the playback task
    # that no more audio chunks will arrive and it should exit cleanly.
    _AUDIO_STREAM_FINISHED = object()

    def __init__(
            self,
            api_key: str,
            model_name: str = "models/gemini-3.1-flash-live-preview",
            agent_yaml_path: str = "./test_npc/AGENT.yaml",
    ):
        """
        Configure the NPC. No network or audio resources are acquired here;
        those are opened lazily inside `__aenter__` so that the lifetime of
        the Live session matches one full conversation.

        :param api_key: Google Gemini API key.
        :param model_name: Live-capable Gemini model identifier.
        :param agent_yaml_path: Path to the YAML file describing the NPC's
            persona, world, voice, and goal. Used to build the system prompt.
        """
        self.agent_yaml_path = agent_yaml_path
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        # Per-conversation state. Populated in `__aenter__`, cleaned up in
        # `__aexit__`. They are `None` outside an `async with` block so that
        # accidentally calling `respond_to_text` without entering the context
        # fails fast with a clear AttributeError.
        self.live_session = None
        self._live_session_context_manager = None
        self.audio_output_stream: sounddevice.RawOutputStream | None = None
        self.audio_playback_queue: asyncio.Queue | None = None
        self.audio_playback_task: asyncio.Task | None = None

        # transcript of the most recent conversation
        self.transcript: str = ""

        with open(self.agent_yaml_path, "r") as yaml_file:
            self.agent_data = yaml.safe_load(yaml_file) or {}

    def _create_system_prompt(self) -> str:
        """
        Render the YAML persona file into a Markdown system prompt.

        Each top-level YAML key becomes a `### key` section. If the value is a
        mapping, its `description` and non-null `content` fields are appended
        as paragraphs in that order.

        :return: Markdown-formatted system prompt string.
        """
        with open(self.agent_yaml_path, "r") as yaml_file:
            agent_data = yaml.safe_load(yaml_file) or {}

        rendered_sections: list[str] = []
        for section_key, section_value in agent_data.items():
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
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "memories": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                },
            ),
        )
        return response.text

    def _update_memory(self, transcript: str) -> None:
        old_memory = self.agent_data["memory"]["content"]
        if old_memory:
            old_memory = "\n".join(f"- {item}" for item in old_memory)
        else:
            old_memory = "No memories yet. You meet the person for the first time."

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

        with open(self.agent_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        data["memory"]["content"] = new_memory
        with open(self.agent_yaml_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def _update_events(self, event: dict[str, str]) -> None:
        """
        Append a new event to EVENTS.md. Stub for now.

        :param event: Dict with keys `time`, `location`, `event_description`.
        """
        pass

    def give_item(self, item_name: str) -> dict:
        """Hand an item to the player. Stub for now."""
        print(f"gave {item_name=}")
        return {"status": "ok", "item_name": item_name}

    async def __aenter__(self) -> "NPC":
        """
        Open the Live session and start the audio playback machinery.

        This runs once at the start of a conversation. The websocket and the
        audio device are kept open for the entire `async with` block, so each
        call to `respond_to_text` only pays for sending text and draining the
        reply, not for connection setup.
        """
        # reset transcript for a new conversation
        self.transcript = ""

        give_item_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="give_item",
                    description="Give a specific item to the player. Call this when the NPC decides to hand something over.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "item_name": types.Schema(
                                type=types.Type.STRING,
                                description="Short name of the item being given, e.g. 'iron key'.",
                            ),
                        },
                        required=["item_name"],
                    ),
                )
            ]
        )

        # Build the Live session config.
        live_session_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Iapetus"
                    )
                )
            ),
            # Ask the server to send a text transcript of its own audio output.
            # We use this to populate `self.transcript` and to print the
            # NPC's reply to stdout in real time.
            output_audio_transcription=types.AudioTranscriptionConfig(),
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            system_instruction=self._create_system_prompt(),
            tools=[give_item_tool]
        )

        # `client.aio.live.connect(...)` returns an async context manager.
        # We don't use `async with` here because we need to hold the session
        # open across multiple method calls; instead we drive its lifecycle
        # manually via `__aenter__` / `__aexit__`.
        self._live_session_context_manager = self.client.aio.live.connect(
            model=self.model_name,
            config=live_session_config,
        )
        self.live_session = await self._live_session_context_manager.__aenter__()

        # Open the speaker stream. Gemini Live emits 24 kHz mono 16-bit PCM,
        # which is what these parameters configure.
        self.audio_output_stream = sounddevice.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype="int16",
        )
        self.audio_output_stream.start()

        # A queue decouples receive() (which must not block) from the speaker
        # write (which is blocking I/O). Audio chunks land here as they
        # arrive; the playback task pulls them off and writes them out.
        self.audio_playback_queue = asyncio.Queue()

        async def play_audio_chunks_until_finished() -> None:
            """
            Pull audio chunks off the queue and write them to the speaker.

            Runs for the lifetime of the conversation. Exits when it sees the
            `_AUDIO_STREAM_FINISHED` sentinel, which `__aexit__` enqueues.
            """
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

        self.audio_playback_task = asyncio.create_task(
            play_audio_chunks_until_finished()
        )
        return self

    async def __aexit__(self, exception_type, exception_value, exception_traceback) -> None:
        """
        Tear down the Live session and audio device.

        Order matters: we first let the playback task finish draining any
        audio still in the queue (so the user hears the tail of the last
        reply), then close the websocket, then close the speaker. Each step
        is in its own try/finally so a failure in one cleanup step doesn't
        skip the others.
        """
        try:
            # Tell the playback task that no more chunks will arrive and wait
            # for it to drain whatever is already queued.
            self.audio_playback_queue.put_nowait(self._AUDIO_STREAM_FINISHED)
            await self.audio_playback_task
        finally:
            try:
                # Close the websocket via the same context manager we opened.
                await self._live_session_context_manager.__aexit__(
                    exception_type, exception_value, exception_traceback,
                )
            finally:
                # `stop()` blocks until the speaker buffer empties; only then
                # is it safe to close the device.
                await asyncio.to_thread(self.audio_output_stream.stop)
                self.audio_output_stream.close()

    async def respond_to_text(self, user_text: str) -> str:
        """
        Send one player utterance and play the NPC's spoken reply.

        Must be called inside an `async with npc:` block. The call returns
        once the model signals `turn_complete`; the audio tail of the reply
        may continue to play in the background while the next turn begins,
        which is fine because the playback task and the speaker stream are
        shared across turns.

        :param user_text: What the player said (or typed) this turn.
        :return: The full text of the NPC's reply, also appended to
            `self.transcript` along with the user's input.
        """
        await self.live_session.send_realtime_input(text=user_text)

        print(f"user: {user_text}")
        print("npc: ", end="", flush=True)

        # We collect transcript fragments as they stream in and join them at
        # the end. A list + final join is cheaper than repeated string
        # concatenation, especially for longer replies.
        response_text_chunks: list[str] = []
        async for live_server_message in self.live_session.receive():
            # Tool calls arrive on their own messages, before server_content.
            if live_server_message.tool_call:
                function_responses = []
                for function_call in live_server_message.tool_call.function_calls:
                    if function_call.name == "give_item":
                        result = self.give_item(**(function_call.args or {}))
                        function_responses.append(
                            types.FunctionResponse(
                                id=function_call.id,
                                name=function_call.name,
                                response=result,
                            )
                        )
                if function_responses:
                    await self.live_session.send_tool_response(
                        function_responses=function_responses
                    )
                continue

            server_content = live_server_message.server_content
            if not server_content:
                continue

            # Streamed text transcript of the NPC's spoken reply.
            if (
                server_content.output_transcription
                and server_content.output_transcription.text
            ):
                transcript_text_chunk = server_content.output_transcription.text
                response_text_chunks.append(transcript_text_chunk)
                # Echo to stdout immediately so the user sees the reply being
                # transcribed in real time.
                print(transcript_text_chunk, end="", flush=True)

            # Streamed PCM audio chunks from the model's voice. Each chunk
            # gets queued for the playback task to write to the speaker.
            if server_content.model_turn and server_content.model_turn.parts:
                for model_turn_part in server_content.model_turn.parts:
                    if (
                        model_turn_part.inline_data
                        and model_turn_part.inline_data.data
                    ):
                        # Non-blocking enqueue keeps this receive loop fast,
                        # which matters because falling behind here causes
                        # backpressure on the websocket.
                        self.audio_playback_queue.put_nowait(
                            model_turn_part.inline_data.data
                        )

            # Server signals that the model is done speaking for this turn.
            # Audio queued before this point may still be playing; that's
            # handled by the playback task and is not our concern here.
            if server_content.turn_complete:
                print()  # newline after the streamed transcript
                break

        # Persist the full turn to the transcript so it's available for the
        # post-conversation memory update.
        full_response_text = "".join(response_text_chunks)
        self.transcript += f"Player\n{user_text}\n\n"
        self.transcript += f"{self.agent_data["name"]}\n{full_response_text}\n\n"
        return full_response_text