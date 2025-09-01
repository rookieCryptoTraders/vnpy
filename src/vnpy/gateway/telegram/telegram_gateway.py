# This code was adjusted by Gemini
from __future__ import annotations

import asyncio
import threading

from telethon import TelegramClient, events

from vnpy.event import EventEngine
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    SubscribeRequest,
    TelegramData,
)
from vnpy.trader.event import EVENT_TELEGRAM


class TelegramGateway(BaseGateway):
    """
    Gateway for receiving messages from Telegram.
    This gateway connects to Telegram as a user account and listens for new
    messages in specified chat groups. It forwards the message content
    as TelegramData events.
    Requires `telethon` and `pysocks` to be installed.
    """

    default_name: str = "TELEGRAM"
    default_setting: dict[str, str | int | float | bool] = {
        "api_id": 0,
        "api_hash": "",
        "session_name": "telegram_session",
    }
    exchanges: list = []

    def __init__(self, event_engine: EventEngine, gateway_name: str) -> None:
        """Initializes the gateway."""
        super().__init__(event_engine, gateway_name)
        self.client: TelegramClient | None = None
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.active: bool = False

        self.api_id: int = 0
        self.api_hash: str = ""
        self.session_name: str = ""
        self.subscribed_groups: set[int] = set()

    def connect(self, setting: dict) -> None:
        """
        Start gateway connection.
        """
        self.api_id = int(setting["telegram.api_id"])
        self.api_hash = str(setting["telegram.api_hash"])
        self.session_name = str(setting["telegram_.ession_name"])

        if not self.api_id or not self.api_hash:
            self.write_log("Missing required settings: api_id or api_hash.")
            return

        self.active = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        self.write_log("Telegram gateway started.")

    def _run(self) -> None:
        """
        Run the asyncio event loop in a separate thread.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.client = TelegramClient(
            self.session_name, self.api_id, self.api_hash, loop=self.loop
        )

        @self.client.on(events.NewMessage())
        async def handle_new_message(event: events.NewMessage.Event) -> None:
            """Handle new messages from any chat."""
            peer_id = (
                event.message.peer_id.to_dict().get("channel_id")
                or event.message.peer_id.to_dict().get("chat_id")
                or event.message.peer_id.to_dict().get("user_id")
            )

            # telethon group id may be positive, but telegram bot api returns negative id
            # for consistency, we convert it to negative
            if peer_id and not str(peer_id).startswith("-100"):
                peer_id = int("-100" + str(peer_id))

            if peer_id in self.subscribed_groups:
                message_text: str = event.message.message
                if message_text:
                    telegram_data = TelegramData(
                        msg=message_text,
                        peer_id=peer_id,
                        gateway_name=self.gateway_name,
                    )
                    self.on_event(EVENT_TELEGRAM, telegram_data)

        async def start_client() -> None:
            """Connect and run the client."""
            await self.client.start()
            self.write_log("Telegram client connected.")
            await self.client.run_until_disconnected()

        self.loop.run_until_complete(start_client())

    def close(self) -> None:
        """
        Close gateway connection.
        """
        self.active = False
        if self.client and self.client.is_connected() and self.loop:
            future = asyncio.run_coroutine_threadsafe(
                self.client.disconnect(), self.loop
            )
            try:
                future.result(timeout=5)  # Wait for disconnect to complete
            except TimeoutError:
                self.write_log("Telegram client disconnect timed out.")

        if self.thread:
            self.thread.join()

        self.write_log("Telegram gateway stopped.")

    def subscribe(self, req: SubscribeRequest) -> None:
        """
        Subscribe to a chat group.
        The group ID should be passed in req.symbol.
        """
        try:
            group_id = int(req.symbol)
            self.subscribed_groups.add(group_id)
            self.write_log(f"Subscribed to Telegram group: {group_id}")
        except ValueError:
            self.write_log(
                f"Invalid group ID format for subscription: {req.symbol}. It must be an integer."
            )
