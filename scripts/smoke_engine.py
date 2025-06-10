import asyncio
from mrce.mrcore.engine import MRCEngine

async def main():
    engine = MRCEngine()

    while True:
        user_msg = input("\nUSER > ")
        if user_msg.lower() in {"exit", "quit"}:
            break
        reply = await engine.run_turn(user_msg)
        print(f"\n[MRCE] {reply}")

if __name__ == "__main__":
    asyncio.run(main())
