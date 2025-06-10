import asyncio, json
from pathlib import Path
from rich import print

from mrce.mrcore.loader import load_personae
from mrce.mrcore.coordinator import CoordinatorLogic

async def main():
    agents = load_personae(Path("configs/personae"))
    coordinator = CoordinatorLogic()

    user = "Give me a one‑sentence fact about black holes."
    context = {"fecr_vector": [1/13]*13}

    # Coordinator guidance first
    guide_msg = coordinator.guide(user, context["fecr_vector"])
    context["guide"] = guide_msg

    # parallel expert replies
    replies = await asyncio.gather(*(ag.step(user, context) for ag in agents))

    # print full replies with rich (no wrap truncation)
    for ag, rep in zip(agents, replies):
        print(f"[bold cyan][{ag.name}][/]: {rep}")

    # Coordinator summary
    summary = coordinator.summarise(replies, context)
    print(f"[bold magenta][Coordinator‑Summary][/]: {summary}")

if __name__ == "__main__":
    asyncio.run(main())
