from pathlib import Path
import yaml

from mrce.mrcore.agent import ExpertAgent


def load_personae(dir_path: Path | str = "configs/personae") -> list[ExpertAgent]:
    """
    Load every *.yaml in configs/personae into an ExpertAgent.
    """
    dir_path = Path(dir_path)
    agents: list[ExpertAgent] = []
    for yfile in sorted(dir_path.glob("*.yaml")):
        cfg = yaml.safe_load(yfile.read_text(encoding="utf-8"))
        agents.append(ExpertAgent(cfg))
    return agents
