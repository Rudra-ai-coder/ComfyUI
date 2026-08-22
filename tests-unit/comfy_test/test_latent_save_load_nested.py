import os

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.nested_tensor  # noqa: E402
import folder_paths  # noqa: E402
import nodes  # noqa: E402


def test_save_load_nested_av_latent(tmp_path, monkeypatch):
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 32, 2, 40)
    samples = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    saver = nodes.SaveLatent()
    saver.output_dir = str(tmp_path)
    out = saver.save(samples, filename_prefix="h3")
    loc = out["ui"]["latents"][0]
    path = os.path.join(str(tmp_path), loc["subfolder"], loc["filename"]) if loc["subfolder"] else os.path.join(str(tmp_path), loc["filename"])

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name, default_dir=None: path)
    loaded, = nodes.LoadLatent().load(loc["filename"])
    loaded_video, loaded_audio = loaded["samples"].unbind()
    assert torch.equal(loaded_video, video)
    assert torch.equal(loaded_audio, audio)


def test_save_load_plain_latent_unchanged(tmp_path, monkeypatch):
    samples = {"samples": torch.randn(1, 4, 8, 8)}
    saver = nodes.SaveLatent()
    saver.output_dir = str(tmp_path)
    out = saver.save(samples, filename_prefix="sd")
    loc = out["ui"]["latents"][0]
    path = os.path.join(str(tmp_path), loc["filename"])

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name, default_dir=None: path)
    loaded, = nodes.LoadLatent().load(loc["filename"])
    assert torch.equal(loaded["samples"], samples["samples"])


def test_load_latent_path_from_output_subfolder(tmp_path, monkeypatch):
    samples = {"samples": torch.randn(1, 4, 8, 8)}
    saver = nodes.SaveLatent()
    saver.output_dir = str(tmp_path)
    out = saver.save(samples, filename_prefix="latents/ComfyUI")
    loc = out["ui"]["latents"][0]
    rel = "{}/{}".format(loc["subfolder"], loc["filename"]) if loc["subfolder"] else loc["filename"]

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    loaded, = nodes.LoadLatentPath().load("output", rel)
    assert torch.equal(loaded["samples"], samples["samples"])


def test_load_latent_path_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    assert nodes.LoadLatentPath.VALIDATE_INPUTS("output", "../secret.latent") is not True


def test_list_latent_combo_includes_output_subfolder(tmp_path, monkeypatch):
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    (latent_dir / "clip.latent").write_bytes(b"x")
    monkeypatch.setattr(folder_paths, "get_directory_by_type", lambda t: str(tmp_path) if t == "output" else None)
    files = nodes._list_latent_combo_files()
    assert "latents/clip.latent [output]" in files
