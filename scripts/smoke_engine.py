import asyncio
import argparse
from mrce.mrcore.engine import MRCEngine
from mrce.mrcore.coordinator import State

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="print FECR details")
    args = parser.parse_args()

    engine = MRCEngine()

    user_msg = input("\nUSER > ")
    while True:
        if user_msg.lower() in {"exit", "quit"}:
            break
        reply = await engine.run_turn(user_msg, debug=args.debug)
        print(f"\n[MRCE] {reply}")
        if engine.coordinator.state in {State.CONVERGE, State.CONTRADICTION}:
            break
        user_msg = input("\nUSER > ")

if __name__ == "__main__":
    asyncio.run(main())
