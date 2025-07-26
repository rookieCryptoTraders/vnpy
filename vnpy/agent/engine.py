# This code was written by Gemini
from __future__ import annotations
import re
import logging
from typing import TYPE_CHECKING

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.event import EVENT_TELEGRAM
from vnpy.trader.object import TelegramData
from vnpy.trader.setting import SETTINGS

if TYPE_CHECKING:
    import google.generativeai as genai


URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"


class GeminiAgent:
    """
    A wrapper for the Gemini AI model to provide analysis functions.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """Initializes the Gemini Agent."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.model: "genai.GenerativeModel" = genai.GenerativeModel(model_name)

    def analyse_text(self, text: str) -> str:
        """
        Analyses the given text using Gemini.
        """
        prompt = f"Please analyse the following text and provide a summary:\n\n{text}"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analysing text: {e}"

    def analyse_url(self, url: str) -> str:
        """
        Analyses the content of a given URL using Gemini.
        """
        # In a real implementation, you would fetch the content of the URL first.
        prompt = f"Please analyse the content of the website at the following URL and provide a summary:\n\n{url}"
        try:
            # Placeholder for fetching URL content. You can use libraries like requests or beautifulsoup.
            # import requests
            # from bs4 import BeautifulSoup
            # page = requests.get(url)
            # soup = BeautifulSoup(page.content, 'html.parser')
            # page_text = soup.get_text()
            # response = self.model.generate_content(f"Analyse this content: {page_text}")
            response = self.model.generate_content(prompt) # For now, just send the URL
            return response.text
        except Exception as e:
            return f"Error analysing URL: {e}"


class AgentEngine(BaseEngine):
    """
    The AgentEngine is responsible for processing events from various sources,
    such as Telegram, and using an AI agent to analyse the content.
    """

    engine_name: str = "Agent"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Initializes the AgentEngine."""
        super().__init__(main_engine, event_engine, self.engine_name)
        self.gemini_agent: GeminiAgent | None = None

        self.write_log("AgentEngine initialized.")
        self.init_agent()
        self.register_event()

    def init_agent(self) -> None:
        """
        Initializes the Gemini agent from settings.
        """
        api_key = SETTINGS.get("agent.gemini_api_key")
        model_name = SETTINGS.get("agent.gemini_model_name", "gemini-pro")

        if not api_key:
            self.write_log("Gemini API key not found in settings. Agent will not be active.", level=logging.WARNING)
            return

        try:
            self.gemini_agent = GeminiAgent(api_key=api_key, model_name=model_name)
            self.write_log("Gemini agent initialized successfully.")
        except ImportError:
            self.write_log("Failed to import google-generativeai. Please install it.", level=logging.ERROR)
        except Exception as e:
            self.write_log(f"Failed to initialize Gemini agent: {e}", level=logging.ERROR)

    def register_event(self) -> None:
        """Registers the event handler for Telegram events."""
        self.event_engine.register(EVENT_TELEGRAM, self.process_telegram_event)
        self.write_log("Registered event handler for EVENT_TELEGRAM.")

    def process_telegram_event(self, event: Event) -> None:
        """
        Processes incoming Telegram messages.
        """
        if not self.gemini_agent:
            self.write_log("Gemini agent not initialized. Cannot process message.", level=logging.WARNING)
            return

        telegram_data: TelegramData = event.data
        message: str = telegram_data.msg
        self.write_log(f"Processing Telegram message from peer {telegram_data.peer_id}: {message}")

        # Check if the message contains a URL
        urls = re.findall(URL_REGEX, message)

        analysis_result = ""
        try:
            if urls:
                for url in urls:
                    # The regex might return a tuple, get the first element
                    url_to_process = url[0] if isinstance(url, tuple) else url
                    self.write_log(f"Detected URL, analysing: {url_to_process}")
                    analysis_result += self.gemini_agent.analyse_url(url_to_process) + "\n"
            else:
                self.write_log("No URL detected, analysing as plain text.")
                analysis_result = self.gemini_agent.analyse_text(message)

            # The analysis result can be logged, sent back to Telegram,
            # or used to trigger other actions in the system.
            self.write_log(f"AI Analysis Result:\n{analysis_result}")

            # Example of sending feedback (format to be defined later)
            # feedback_event = Event(EVENT_AI_FEEDBACK, analysis_result)
            # self.event_engine.put(feedback_event)

        except Exception as e:
            self.write_log(f"Error during AI analysis: {e}", level=logging.ERROR)

    def close(self) -> None:
        """Stops the engine."""
        self.write_log("AgentEngine stopped.")
