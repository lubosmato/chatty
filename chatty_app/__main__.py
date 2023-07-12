from llama_cpp import Llama
from sys import stdout
import argparse
import signal
import pickle
from typing import Iterator
import re
from pathlib import Path
from termcolor import colored


def main():
    App().run()

class App:
    PROMPT = "Q:"

    @staticmethod
    def validated_path(arg: str) -> str:
        if not re.match(r"^[a-zA-Z0-9]+$", arg):
            raise ValueError
        return arg

    def __init__(self) -> None:
        parser = argparse.ArgumentParser("chatty")
        parser.add_argument("-s", "--session", help="session key (not implemented)", type=App.validated_path, nargs=argparse.OPTIONAL)
        parser.add_argument("-i", "--interactive", help="interactive mode", action="store_true")
        parser.add_argument("prompt", help="prompt to execute", nargs=argparse.REMAINDER)
        args = parser.parse_args()

        signal.signal(signal.SIGINT, self.handle_ctrl_c)

        session_key = args.session
        self.prompt = " ".join(args.prompt)
        self.is_interactive = args.interactive
        self.is_predicting = False
        self.should_abort_prediction = False

        if not self.prompt:
            self.is_interactive = True

        self.chatty = Chatty(session_key)

        if session_key:
            print(f"using session '{session_key}'")

        self.chatty.load_session()

    def handle_ctrl_c(self, signum, frame):
        if not self.is_predicting:
            signal.default_int_handler(signum, frame)
        self.should_abort_prediction = True

    def run(self) -> None:
        if self.prompt:
            print(colored(self.PROMPT, "green"), self.prompt)
            self.execute_prompt()
            print("\n")

        if self.is_interactive:
            while not self.should_abort_prediction:
                try:
                    print(colored(f"{self.PROMPT} ", "green"), end="")
                    self.prompt = input()
                    self.execute_prompt()
                    print("\n")
                except KeyboardInterrupt:
                    self.should_abort_prediction = True
        print()

    def execute_prompt(self) -> None:
        self.is_predicting = True
        try:
            for part in self.chatty.predict(self.prompt):
                print(part, end="")
                stdout.flush()
                if self.should_abort_prediction:
                    self.should_abort_prediction = False
                    break
        except KeyboardInterrupt:
            pass
        self.is_predicting = False
        self.chatty.save_session()

class Chatty:
    def __init__(self, session_key: str | None):
        self.root_path = Path(__file__).parent.parent.resolve()
        self.model_path = self.root_path / "models" / "Wizard-Vicuna-7B-Uncensored.ggmlv3.q8_0.bin"
        # self.model_path = self.root_path / "models" / "wizard-mega-13B.ggmlv3.q8_0.bin"
        session_key = session_key if session_key else "default"
        self.session_path = self.root_path / "sessions" / session_key

        self.llm = Llama(model_path=str(self.model_path), verbose=False)

    def predict(self, prompt: str) -> Iterator[str]:
        predictions = self.llm(prompt, echo=False, max_tokens=1024, stream=True, temperature=0.1)
        return (
            prediction["choices"][0]["text"]
            for prediction in predictions
        )

    def load_session(self) -> None:
        try:
            with open(self.session_path, "rb") as f:
                state = pickle.load(f)
                self.llm.load_state(state)
        except Exception:
            pass

    def save_session(self) -> None:
        with open(self.session_path, "wb") as f:
            pickle.dump(self.llm.save_state(), f)

if __name__ == "__main__":
    main()
