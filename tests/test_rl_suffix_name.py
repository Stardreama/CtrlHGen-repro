import ast
from pathlib import Path
from types import SimpleNamespace


def load_rl_suffix_name():
    source_path = Path(__file__).resolve().parents[1] / "akgr" / "abduction_model" / "main.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "rl_suffix_name"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["rl_suffix_name"]


def test_grpo_suffix_uses_current_rl_args():
    rl_suffix_name = load_rl_suffix_name()
    args = SimpleNamespace(
        rl_type="GRPO",
        rl_lr=1e-6,
        rl_smatch_factor=0,
        rl_init_kl_coef=0.2,
        rl_cliprange=0.2,
        rl_minibatch=1,
        rl_horizon=10000,
        rl_epochs=1,
        rl_share_embed_layer=False,
        rl_lr_no_decay=False,
        rl_use_peft=False,
        rl_search_split="train",
        rl_proportion=0.01,
        rl_factor="[1.0, 1.0, 0.5, 0.0]",
    )

    assert rl_suffix_name(args, 1) == (
        "grpo_1e-06_0_0.2_0.2_1_10000_1x0.01_f[1.0,1.0,0.5,0.0]-1"
    )


if __name__ == "__main__":
    test_grpo_suffix_uses_current_rl_args()
