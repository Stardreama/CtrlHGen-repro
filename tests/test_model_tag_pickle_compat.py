import ast
import pickle
from pathlib import Path
from types import SimpleNamespace


class FakePreTrainedModel:
    pass


def dummy_add_model_tags(self, tags):
    pass


def load_install_model_tag_compat():
    source_path = Path(__file__).resolve().parents[1] / "akgr" / "utils" / "load_util.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "install_model_tag_pickle_compat"
    )
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "transformers": SimpleNamespace(PreTrainedModel=FakePreTrainedModel),
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["install_model_tag_pickle_compat"]


def test_install_model_tag_compat_allows_old_dynamic_method_pickles():
    model = FakePreTrainedModel()
    model.add_model_tags = dummy_add_model_tags.__get__(model)
    payload = pickle.dumps(model)

    try:
        pickle.loads(payload)
    except AttributeError as exc:
        assert "dummy_add_model_tags" in str(exc)
    else:
        raise AssertionError("pickle should fail before installing compatibility method")

    install_model_tag_pickle_compat = load_install_model_tag_compat()
    install_model_tag_pickle_compat()

    loaded = pickle.loads(payload)
    loaded.add_model_tags(["trl"])


if __name__ == "__main__":
    test_install_model_tag_compat_allows_old_dynamic_method_pickles()
