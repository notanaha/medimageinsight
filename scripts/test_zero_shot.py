"""Offline tests for every script in this folder. No Azure access, no upload, no charges.

    python scripts/test_zero_shot.py

Everything is synthetic: the label sheet, the tiles, the embeddings and the Foundry client are
all stand-ins, so this checks the plumbing - label handling, magnification arithmetic, the
cosine scoring and the ROC - not the biology. No clinical data is read.
"""

import contextlib
import csv
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_dataset
import cohort
import embed_tiles
import score_zero_shot
import tile_wsi
from cohort import CLASS_NAMES, labeled_only, read_cohort, read_cohort_csv, write_cohort
from mii_client import (DIMENSIONS, EMBED_PATH, TOKEN_SCOPE, PremiumClient, load_labels,
                        normalize_vectors, prepare_image, resource_endpoint, svs_overview)


def quiet():
    """The entry points print a progress report; keep it out of the test output."""
    return contextlib.redirect_stdout(io.StringIO())


def write_sheet(path, rows):
    """A sheet shaped like the real one: a Windows source path plus the two columns we read."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["H&E Images", "Subject ID", "Responder/Not Responder", "Response"])
        writer.writerows([*row, ""] for row in rows)
    return Path(path)


class ClientTests(unittest.TestCase):
    def test_endpoint_from_project_resource_or_full_url(self):
        base = "https://example.services.ai.azure.com"
        for value in (base, base + "/", base + "/api/projects/test", base + EMBED_PATH):
            with self.subTest(value=value):
                self.assertEqual(resource_endpoint(value), base + EMBED_PATH)

    def test_endpoint_rejects_unsafe_urls(self):
        for value in ("http://example.com", "https://user:secret@example.com",
                      "https://example.com?key=x", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resource_endpoint(value)

    def test_labels_validate_prompts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.json"
            labels = {"CT": "CT image", "H&E": "H&E stained histology"}
            path.write_text(json.dumps(labels), encoding="utf-8")
            self.assertEqual(load_labels(path), labels)
            for invalid in ({"CT": "CT"}, {"CT": "", "MRI": "MRI"}, {"a": "same", "b": "same"}, []):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(labels=invalid), self.assertRaises(ValueError):
                    load_labels(path)

    def test_cosine_normalization_and_ranking(self):
        vectors = np.zeros((2, DIMENSIONS))
        vectors[0, 0] = 3
        vectors[1, 1] = 4
        normalized = normalize_vectors(vectors, 2)
        np.testing.assert_allclose(normalized @ normalized.T, np.eye(2))
        self.assertEqual(int((normalized[1:2] @ normalized.T).argmax()), 1)

    def test_invalid_embedding_rejected(self):
        for values in ([np.zeros((1, DIMENSIONS)), [[1, 2]], np.full((1, DIMENSIONS), np.nan)]):
            with self.subTest(shape=np.shape(values)), self.assertRaises(ValueError):
                normalize_vectors(values, 1)

    def test_payload_auth_and_one_modality_per_request(self):
        credential = Mock()
        credential.get_token.return_value = SimpleNamespace(token="offline-test-token")
        client = PremiumClient("https://example.com" + EMBED_PATH, "MedImageInsight-Premium",
                               credential)
        client.session = Mock()
        response = Mock(status_code=200, headers={"apim-request-id": "test-request"})
        response.json.return_value = {"id": "test-id", "embeddings": {"float": [[1] * DIMENSIONS]}}
        client.session.post.return_value = response
        for modality, values in (("texts", ["CT image"]), ("images", ["data:image/png;base64,AA=="])):
            vectors = client.embed(modality, values)
            self.assertEqual(vectors.shape, (1, DIMENSIONS))
            kwargs = client.session.post.call_args.kwargs
            self.assertEqual(set(kwargs["json"]), {"model", modality, "embedding_types"})
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer offline-test-token")
            self.assertFalse(kwargs["allow_redirects"])
        credential.get_token.assert_called_with(TOKEN_SCOPE)
        # The call log is written to the job output, so it must never carry the token.
        self.assertNotIn("offline-test-token", json.dumps(client.calls))
        client.close()

    def test_api_errors_do_not_retry_or_echo_raw_body(self):
        client = PremiumClient("https://example.com" + EMBED_PATH, "test", Mock())
        client.session = Mock()
        client.session.post.return_value = Mock(status_code=403, headers={}, text="sensitive-body")
        with self.assertRaisesRegex(RuntimeError, "HTTP 403") as raised:
            client.embed("texts", ["CT image"])
        self.assertNotIn("sensitive-body", str(raised.exception))
        client.session.post.assert_called_once()
        client.close()

    def test_invalid_modality_rejected_before_auth(self):
        credential = Mock()
        client = PremiumClient("https://example.com" + EMBED_PATH, "test", credential)
        for modality, inputs in (("text", ["x"]), ("images", [])):
            with self.assertRaises(ValueError):
                client.embed(modality, inputs)
        credential.get_token.assert_not_called()
        client.close()

    def test_png_preparation_preserves_aspect_and_flattens_alpha(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.png", Path(folder) / "output.png"
            Image.new("RGBA", (200, 100), (0, 0, 0, 0)).save(source)
            metadata = prepare_image(source, output, 80)
            self.assertEqual(metadata["original_size"], [200, 100])
            self.assertEqual(metadata["prepared_size"], [80, 40])
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_svs_reads_only_a_bounded_pyramid_level(self):
        full = SimpleNamespace(shape=(20000, 20000, 3), axes="YXS",
                               dtype=np.dtype("uint8"), asarray=Mock())
        low = SimpleNamespace(shape=(20, 30, 3), axes="YXS", dtype=np.dtype("uint8"),
                              asarray=Mock(return_value=np.zeros((20, 30, 3), dtype=np.uint8)))
        with patch("tifffile.TiffFile") as reader:
            reader.return_value.__enter__.return_value.series = [
                SimpleNamespace(shape=full.shape, levels=[full, low])]
            image, metadata = svs_overview(Path("test.svs"))
        self.assertEqual(image.size, (30, 20))
        self.assertEqual(metadata["pyramid_level"], 1)
        full.asarray.assert_not_called()

    def test_svs_without_safe_level_rejected_before_decode(self):
        full = SimpleNamespace(shape=(20000, 20000, 3), axes="YXS",
                               dtype=np.dtype("uint8"), asarray=Mock())
        with patch("tifffile.TiffFile") as reader:
            reader.return_value.__enter__.return_value.series = [
                SimpleNamespace(shape=full.shape, levels=[full])]
            with self.assertRaisesRegex(ValueError, "no safe RGB pyramid"):
                svs_overview(Path("test.svs"))
        full.asarray.assert_not_called()


class ComponentSpecTests(unittest.TestCase):
    """Rules the AML service enforces on submit but `job._validate()` does not check locally."""

    COMPONENT_DIR = Path(__file__).resolve().parent.parent / "component"

    def specs(self):
        paths = sorted(self.COMPONENT_DIR.glob("*.yml"))
        self.assertTrue(paths, f"no component specs under {self.COMPONENT_DIR}")
        for path in paths:
            yield path, path.read_text(encoding="utf-8")

    def test_optional_inputs_are_bracketed_and_have_no_default(self):
        # The service rejects the job with "Optional input <name> must be placed in nested
        # argument: $[[]]" - and only at submit time, after the code has already uploaded.
        for path, text in self.specs():
            command, inputs = text.split("inputs:", 1)
            for name, body in re.findall(r"\n  (\w+):\n((?:    .*\n)*)", inputs):
                if "optional: true" not in body:
                    continue
                with self.subTest(spec=path.name, input=name):
                    self.assertNotIn("default:", body,
                                     "an input with a default always has a value, so it is not optional")
                    self.assertRegex(command, rf"\$\[\[[^\]]*\{{\{{inputs\.{name}\}}\}}[^\]]*\]\]")

    def test_every_declared_input_is_used_by_the_command(self):
        for path, text in self.specs():
            command, inputs = text.split("inputs:", 1)
            for name, _ in re.findall(r"\n  (\w+):\n((?:    .*\n)*)", inputs.split("outputs:")[0]):
                with self.subTest(spec=path.name, input=name):
                    self.assertIn(f"${{{{inputs.{name}}}}}", command)


class CohortTests(unittest.TestCase):
    """One row is one image, i.e. one lesion site. Subject ID must never change the row count."""

    def read(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            return read_cohort(write_sheet(Path(folder) / "sheet.csv", rows))

    def test_two_slides_from_one_subject_stay_two_rows(self):
        rows = self.read([[r"P:\images\site_a.svs", "test-subject", "Responder"],
                          [r"P:\images\site_b.svs", "test-subject", "Non-responder"]])
        self.assertEqual([row["file_name"] for row in rows], ["site_a.svs", "site_b.svs"])
        # Only the file name and the label may travel to Azure.
        self.assertNotIn("test-subject", json.dumps(rows))

    def test_missing_subject_id_does_not_drop_a_labeled_image(self):
        self.assertEqual(self.read([["site.svs", "", "Responder"]]),
                         [{"file_name": "site.svs", "label": "Responder"}])

    def test_spellings_normalised_and_blank_label_not_imputed(self):
        values = ["Non-Responder", "Non-responder", "Non responder", " Responder ", ""]
        rows = self.read([[f"image_{index}.svs", "", value]
                          for index, value in enumerate(values)])
        self.assertEqual([row["label"] for row in rows],
                         ["Non-Responder"] * 3 + ["Responder", None])
        self.assertEqual(len(labeled_only(rows)), 4)

    def test_duplicate_image_and_unknown_label_rejected(self):
        with self.assertRaises(ValueError):
            self.read([["same.svs", "a", "Responder"], ["SAME.svs", "b", "Responder"]])
        with self.assertRaises(ValueError):
            self.read([["site.svs", "a", "unknown"]])

    def test_cohort_csv_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cohort.csv"
            rows = [{"file_name": "a.svs", "label": "Responder"},
                    {"file_name": "b.svs", "label": "Non-Responder"}]
            write_cohort(path, rows)
            self.assertEqual(read_cohort_csv(path), rows)
            self.assertEqual(Path(path).read_text(encoding="utf-8").splitlines()[0],
                             "file_name,label")
            write_cohort(path, [{"file_name": "a.svs", "label": "Maybe"}])
            with self.assertRaises(ValueError):
                read_cohort_csv(path)

    def test_cli_creates_cohort_under_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sheet = write_sheet(root / "sheet.csv", [["sample.svs", "", "Responder"]])
            with contextlib.chdir(root), patch.object(sys, "argv", ["cohort.py", str(sheet)]), quiet():
                cohort.main()
            self.assertEqual(read_cohort_csv(root / "outputs" / "cohort.csv"),
                             [{"file_name": "sample.svs", "label": "Responder"}])
            self.assertFalse((root / "cohort.csv").exists())

    def test_preflight_uses_folder_local_config_and_data(self):
        for configured in ("", "data/custom.csv"):
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "data").mkdir()
                (root / ".env").write_text(f'LABELS_CSV="{configured}"\n', encoding="utf-8")
                write_sheet(root / (configured or "data/labels.csv"),
                            [["sample.svs", "", "Responder"]])
                with (patch.object(check_dataset, "HERE", root),
                      patch.dict("os.environ", {}, clear=True),
                      patch.object(sys, "argv", ["check_dataset.py", "--csv-only"]),
                      patch.object(check_dataset, "list_blobs") as blobs, quiet()):
                    check_dataset.main()
                blobs.assert_not_called()
                reports = list((root / "outputs").glob("dataset-check-*.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertEqual(report["labeled_rows"], 1)
                self.assertEqual(report["model_calls"], 0)


class TilingTests(unittest.TestCase):
    """`tile_slide` needs a real OpenSlide, so the parts that can be tested alone are."""

    def fake_slide(self, base_mpp=0.25, downsamples=(1.0, 4.0, 16.0)):
        return SimpleNamespace(
            properties={"openslide.mpp-x": str(base_mpp), "openslide.mpp-y": str(base_mpp)},
            level_count=len(downsamples), level_downsamples=list(downsamples))

    def test_base_mpp_is_never_guessed(self):
        self.assertEqual(tile_wsi.read_base_mpp(self.fake_slide(0.2465)), 0.2465)
        with self.assertRaisesRegex(ValueError, "no MPP metadata"):
            tile_wsi.read_base_mpp(SimpleNamespace(properties={}))

    def test_level_choice_only_ever_downsamples(self):
        slide = self.fake_slide(base_mpp=0.25)
        # 0.25 * 1 = 0.25, * 4 = 1.0, * 16 = 4.0 MPP per level.
        self.assertEqual(tile_wsi.pick_level(slide, 0.25, 0.5), 0)
        self.assertEqual(tile_wsi.pick_level(slide, 0.25, 1.0), 1)
        self.assertEqual(tile_wsi.pick_level(slide, 0.25, 8.0), 2)

    def test_effective_mpp_lands_inside_the_tolerance(self):
        for base_mpp, target_mpp, tile_size in ((0.25, 0.5, 512), (0.2465, 0.5, 512),
                                                (0.2523, 0.5, 512), (0.5, 0.5, 256),
                                                (0.25, 1.0, 512)):
            slide = self.fake_slide(base_mpp)
            level = tile_wsi.pick_level(slide, base_mpp, target_mpp)
            downsample = slide.level_downsamples[level]
            read_size = round(tile_size * target_mpp / base_mpp / downsample)
            achieved = tile_wsi.effective_mpp(base_mpp, downsample, read_size, tile_size)
            with self.subTest(base_mpp=base_mpp, target_mpp=target_mpp):
                self.assertLessEqual(abs(achieved - target_mpp),
                                     target_mpp * tile_wsi.MPP_TOLERANCE)

    def test_effective_mpp_is_not_expected_to_equal_the_target(self):
        # The cohort's scanner writes 0.2523 MPP, which needs 1014.665 px for a 512 px tile at
        # 0.5 MPP. Reading 1015 px lands 0.03% high. Asserting equality here would reject a
        # correct run, which is why MPP_TOLERANCE exists rather than an equality check.
        achieved = tile_wsi.effective_mpp(0.2523, 1.0, 1015, 512)
        self.assertNotEqual(achieved, 0.5)
        self.assertLess(abs(achieved - 0.5), 0.5 * tile_wsi.MPP_TOLERANCE)

    def test_otsu_splits_a_two_peak_histogram(self):
        values = np.concatenate([np.full(1000, 40), np.full(1000, 220)])
        threshold = tile_wsi.otsu_threshold(values)
        # tissue_occupancy keeps pixels <= threshold, so the dark peak must land on the
        # tissue side and the bright peak outside it.
        self.assertTrue(40 <= threshold < 220)

    def test_occupancy_is_measured_on_a_thumbnail_not_on_tiles(self):
        # Left half dark (tissue), right half white (glass).
        thumbnail = Image.new("L", (64, 32), 255)
        thumbnail.paste(Image.new("L", (32, 32), 20), (0, 0))
        slide = SimpleNamespace(level_dimensions=[(6400, 3200)],
                                get_thumbnail=Mock(return_value=thumbnail),
                                read_region=Mock())
        occupancy = tile_wsi.tissue_occupancy(slide, (4, 2), span=1600)
        self.assertEqual(occupancy.shape, (2, 4))
        self.assertTrue((occupancy[:, :2] > 0.9).all())
        self.assertTrue((occupancy[:, 2:] < 0.1).all())
        slide.read_region.assert_not_called()


class EmbedTilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.rows = [{"file_name": "negative.svs", "label": "Non-Responder"},
                     {"file_name": "positive.svs", "label": "Responder"}]
        for row in self.rows:
            (self.images / row["file_name"]).touch()

    def test_missing_or_ambiguous_slide_is_never_skipped(self):
        (self.images / "positive.svs").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "positive.svs"):
            embed_tiles.resolve_slides(self.rows, self.images)

        (self.images / "positive.svs").touch()
        nested = self.images / "rescans"
        nested.mkdir()
        (nested / "negative.svs").touch()
        with self.assertRaises(FileNotFoundError):
            embed_tiles.resolve_slides(self.rows, self.images)

    def test_tiles_are_sent_in_batches_as_png_data_urls(self):
        client = Mock()
        client.embed.side_effect = lambda modality, inputs: np.eye(len(inputs), DIMENSIONS)
        tiles = [Image.new("RGB", (8, 8), "white") for _ in range(5)]

        vectors = embed_tiles.embed_tiles(client, tiles, batch_size=2)

        self.assertEqual(vectors.shape, (5, DIMENSIONS))
        self.assertEqual([len(call.args[1]) for call in client.embed.call_args_list], [2, 2, 1])
        for call in client.embed.call_args_list:
            self.assertEqual(call.args[0], "images")
            self.assertTrue(all(item.startswith("data:image/png;base64,") for item in call.args[1]))

    def test_run_writes_one_npz_per_slide_and_never_reads_labels(self):
        cohort_csv = self.root / "cohort.csv"
        write_cohort(cohort_csv, self.rows)
        output_dir = self.root / "step1"
        tiles = [Image.new("RGB", (8, 8), "white") for _ in range(3)]
        metadata = {"candidate_tiles": 10, "effective_mpp": 0.5, "selected_tiles": 3}

        argv = ["embed_tiles.py", "--images_dir", str(self.images), "--cohort_csv",
                str(cohort_csv), "--output_dir", str(output_dir), "--endpoint",
                "https://example.services.ai.azure.com", "--batch_size", "2"]
        with patch.object(sys, "argv", argv), \
                patch.object(embed_tiles, "tile_slide", return_value=(tiles, dict(metadata))), \
                patch.object(embed_tiles, "log_metrics", Mock()), \
                patch.object(embed_tiles, "PremiumClient") as factory, \
                patch("azure.identity.ManagedIdentityCredential", Mock()):
            client = factory.return_value
            client.calls = []
            client.embed.side_effect = lambda modality, inputs: np.eye(len(inputs), DIMENSIONS)
            with quiet():
                embed_tiles.main()
            client.close.assert_called_once()
            # Step 1 is label-free on purpose, so the saved embeddings can be re-scored.
            self.assertTrue(all(call.args[0] == "images" for call in client.embed.call_args_list))

        images = json.loads((output_dir / "images.json").read_text(encoding="utf-8"))
        self.assertEqual([row["image_id"] for row in images["images"]],
                         ["image_000", "image_001"])
        self.assertNotIn("Responder", json.dumps(images))
        for image_id in ("image_000", "image_001"):
            payload = np.load(output_dir / "embeddings" / f"{image_id}.npz")
            self.assertEqual(payload["image_embeddings"].shape, (3, DIMENSIONS))


class ScoringTests(unittest.TestCase):
    def test_score_direction_and_ties(self):
        for scores, expected in (([-0.9, -0.1, 0.2, 0.8], 1.0),
                                 ([0.9, 0.1, -0.2, -0.8], 0.0),
                                 ([0, 0, 0, 0], 0.5)):
            with self.subTest(scores=scores):
                self.assertEqual(score_zero_shot.evaluate([0, 0, 1, 1], scores)[0]["roc_auc"],
                                 expected)

    def test_threshold_is_fixed_at_zero_and_baseline_is_reported(self):
        metrics, curve = score_zero_shot.evaluate([0, 1, 1, 1], [0.5, -0.5, 0.5, 0.5])
        self.assertEqual(metrics["threshold"], 0.0)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [1, 2]])
        # Answering "Responder" for everything already scores 0.75 on this split.
        self.assertEqual(metrics["majority_baseline_accuracy"], 0.75)
        self.assertFalse(metrics["training"])
        self.assertFalse(metrics["probability"])
        self.assertEqual(len(curve["fpr"]), len(curve["tpr"]))

    def test_invalid_evaluation_inputs_fail(self):
        for labels, scores in (([1, 1], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0, 1], [0.1])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                score_zero_shot.evaluate(labels, scores)

    def run_scoring(self, embeddings_dir, cohort_csv, output_dir, text_vectors):
        argv = ["score_zero_shot.py", "--embeddings_dir", str(embeddings_dir),
                "--cohort_csv", str(cohort_csv), "--output_dir", str(output_dir),
                "--endpoint", "https://example.services.ai.azure.com",
                "--prompt_negative", "no response", "--prompt_positive", "response"]
        with patch.object(sys, "argv", argv), \
                patch.object(score_zero_shot, "log_metrics", Mock()), \
                patch.object(score_zero_shot, "PremiumClient") as factory, \
                patch("azure.identity.ManagedIdentityCredential", Mock()):
            client = factory.return_value
            client.embed.return_value = text_vectors
            with quiet():
                score_zero_shot.main()
            self.assertEqual(client.embed.call_args.args[0], "texts")
            # One text request for both prompts; the tiles are never sent again.
            client.embed.assert_called_once()
            client.close.assert_called_once()
        return json.loads((Path(output_dir) / "metrics.json").read_text(encoding="utf-8"))

    def test_separable_embeddings_score_a_perfect_auc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            embeddings_dir = root / "step1"
            (embeddings_dir / "embeddings").mkdir(parents=True)

            # Two prompt vectors along axis 0 and 1; each slide's tiles lean to its own axis.
            text_vectors = np.eye(2, DIMENSIONS)
            rows = [{"file_name": "negative.svs", "label": "Non-Responder"},
                    {"file_name": "positive.svs", "label": "Responder"}]
            images = []
            for index, row in enumerate(rows):
                tile_vectors = np.tile(text_vectors[CLASS_NAMES.index(row["label"])], (4, 1))
                np.savez_compressed(embeddings_dir / "embeddings" / f"image_{index:03d}.npz",
                                    image_embeddings=tile_vectors)
                images.append({"image_id": f"image_{index:03d}",
                               "file_name": row["file_name"], "tile_count": 4})
            (embeddings_dir / "images.json").write_text(
                json.dumps({"model": "test", "images": images}), encoding="utf-8")
            cohort_csv = root / "cohort.csv"
            write_cohort(cohort_csv, rows)

            metrics = self.run_scoring(embeddings_dir, cohort_csv, root / "step2", text_vectors)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(metrics["n_tiles"], 8)
        self.assertEqual(metrics["class_order"], CLASS_NAMES)

    def test_cohort_and_embeddings_must_describe_the_same_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            embeddings_dir = root / "step1"
            (embeddings_dir / "embeddings").mkdir(parents=True)
            (embeddings_dir / "images.json").write_text(json.dumps(
                {"images": [{"image_id": "image_000", "file_name": "other.svs",
                             "tile_count": 1}]}), encoding="utf-8")
            cohort_csv = root / "cohort.csv"
            write_cohort(cohort_csv, [{"file_name": "negative.svs", "label": "Non-Responder"}])
            with self.assertRaisesRegex(ValueError, "different set of images"):
                self.run_scoring(embeddings_dir, cohort_csv, root / "step2", np.eye(2, DIMENSIONS))

    def test_identical_prompts_are_rejected_before_any_request(self):
        argv = ["score_zero_shot.py", "--embeddings_dir", ".", "--cohort_csv", ".",
                "--output_dir", ".", "--endpoint", "https://example.services.ai.azure.com",
                "--prompt_negative", " same text ", "--prompt_positive", "same text"]
        with patch.object(sys, "argv", argv), \
                patch.object(score_zero_shot, "PremiumClient") as factory:
            with self.assertRaisesRegex(ValueError, "must differ"), quiet():
                score_zero_shot.main()
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
