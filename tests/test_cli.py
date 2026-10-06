#!/usr/bin/env python3
import csv, hashlib, json, os, pathlib, subprocess, sys, tempfile, time, unittest

BINARY = pathlib.Path(sys.argv[1]).resolve()
sys.argv = sys.argv[:1]


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name, text):
        p = self.root / name
        p.write_text(text)
        return str(p)

    def execute(self, command, **kwargs):
        config = dict(
            responses=self.file(
                "counts.mtx",
                "%%MatrixMarket matrix coordinate integer general\n1 6 6\n1 1 1\n1 2 2\n1 3 3\n1 4 2\n1 5 1\n1 6 3\n",
            ),
            design=self.file("design.tsv", "1\n" * 6),
            output=str(self.root / "result"),
            family="poisson",
        )
        config.update(kwargs)
        p = self.root / "config.json"
        p.write_text(json.dumps(config))
        r = subprocess.run([BINARY, command, p], text=True, capture_output=True)
        return r, config

    def test_json_and_hashes(self):
        r, c = self.execute("fit", threads=2)
        self.assertEqual(r.returncode, 0, r.stderr)
        record = json.loads((self.root / "result.run.json").read_text())
        self.assertEqual(record["parameters"]["threads"], 2)
        self.assertEqual(
            record["input_sha1"][c["responses"]],
            hashlib.sha1(pathlib.Path(c["responses"]).read_bytes()).hexdigest(),
        )
        self.assertGreater(record["wall_seconds"], 0)

    def test_pattern_matrix(self):
        r, c = self.execute("fit")
        self.assertEqual(r.returncode, 0, r.stderr)
        path = pathlib.Path(c["responses"])
        path.write_text(
            "%%MatrixMarket matrix coordinate pattern general\n1 6 2\n1 2\n1 4\n"
        )
        p = self.root / "config.json"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_malformed_input(self):
        r, c = self.execute("fit")
        path = pathlib.Path(c["responses"])
        path.write_text("%%MatrixMarket matrix coordinate real general\n1 6 1\n1 7 2\n")
        p = self.root / "config.json"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid MatrixMarket entry", r.stderr)

    def test_formula_and_unknown_terms(self):
        r, c = self.execute("fit")
        c.pop("design")
        c["covariates"] = self.file(
            "covariates.tsv", "depth\tbatch\n1\ta\n2\ta\n3\ta\n1\tb\n2\tb\n3\tb\n"
        )
        c["formula"] = "~ log(depth) + factor(batch)"
        p = self.root / "config.json"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        c["formula"] = "~ missing"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown covariate", r.stderr)

    def test_nonconverged_rows_retained(self):
        r, c = self.execute("fit")
        path = pathlib.Path(c["responses"])
        path.write_text(
            "%%MatrixMarket matrix coordinate integer general\n2 6 1\n2 1 1\n"
        )
        p = self.root / "config.json"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = list(
            csv.DictReader(
                (self.root / "result.tsv").read_text().splitlines(), delimiter="\t"
            )
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "boundary_response")
        self.assertEqual(rows[0]["wald_p0"], "NA")
        self.assertEqual(
            json.loads((self.root / "result.run.json").read_text())[
                "invalid_tests_or_fits"
            ],
            1,
        )

    def test_boundary_coefficients_are_explicit_and_not_tested(self):
        r, c = self.execute("fit", threads=2)
        pathlib.Path(c["design"]).write_text("1\t0\n" * 3 + "1\t1\n" * 3)
        pathlib.Path(c["responses"]).write_text(
            "%%MatrixMarket matrix coordinate integer general\n1 6 3\n1 1 1\n1 2 2\n1 3 3\n"
        )
        p = self.root / "config.json"
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "fit", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        row = next(
            csv.DictReader(
                (self.root / "result.tsv").read_text().splitlines(), delimiter="\t"
            )
        )
        self.assertEqual(row["status"], "converged_boundary")
        self.assertEqual(row["boundary1"], "1")
        self.assertEqual(row["se1"], "NA")
        self.assertEqual(row["wald_p1"], "NA")
        self.assertEqual(row["boundary0"], "0")
        self.assertNotEqual(row["wald_p0"], "NA")
        c["coefficient"] = 1
        p.write_text(json.dumps(c))
        r = subprocess.run([BINARY, "pseudobulk-de", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        row = next(
            csv.DictReader(
                (self.root / "result.tsv").read_text().splitlines(), delimiter="\t"
            )
        )
        self.assertEqual(row["coefficient_boundary"], "1")
        self.assertEqual(row["p_value"], "NA")
        self.assertEqual(float(row["adjusted_p"]), 1)

    def test_single_response_draw_scheduling_preserves_output(self):
        r, config = self.execute(
            "sceptre",
            assignments=self.file(
                "guides.mtx",
                "%%MatrixMarket matrix coordinate integer general\n1 6 3\n1 1 1\n1 3 1\n1 5 1\n",
            ),
            guides=self.file("guides.tsv", "id\ttarget\tnon_targeting\ng1\tt1\t0\n"),
            pairs=self.file("pairs.tsv", "response\ttarget\tkind\n0\tt1\tdiscovery\n"),
            moi="high",
            control_group="complement",
            resampling_mechanism="crt",
            cell_qc=False,
            assignment_qc=False,
            min_nonzero=1,
            approximation="no_approximation",
            budgets=[99],
            threads=8,
            seed=719,
            null_model="direct",
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        baseline = (self.root / "result.tsv").read_bytes()
        row = next(csv.DictReader(baseline.decode().splitlines(), delimiter="\t"))
        self.assertEqual(row["status"], "ok")
        for threads, parallel in [(8, False), (1, True)]:
            config.update(threads=threads, parallel_draws=parallel)
            path = self.root / "config.json"
            path.write_text(json.dumps(config))
            result = subprocess.run(
                [BINARY, "sceptre", path], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.root / "result.tsv").read_bytes(), baseline)

    def test_null_model_flag_is_explicit_and_thread_stable(self):
        _, config = self.execute("fit")
        config.update(
            family="nb",
            assignments=self.file(
                "guides.mtx",
                "%%MatrixMarket matrix coordinate integer general\n1 6 3\n1 1 1\n1 3 1\n1 5 1\n",
            ),
            guides=self.file("guides.tsv", "id\ttarget\tnon_targeting\ng1\tt1\t0\n"),
            pairs=self.file("pairs.tsv", "response\ttarget\tkind\n0\tt1\tdiscovery\n"),
            moi="high",
            control_group="complement",
            cell_qc=False,
            assignment_qc=False,
            min_nonzero=1,
            approximation="no_approximation",
            budgets=[99],
            threads=1,
        )
        path = self.root / "config.json"
        outputs = {}
        for mode, threads in [
            (None, 1),
            ("direct", 1),
            ("sceptre", 1),
            ("sceptre", 16),
        ]:
            if mode is not None:
                config["null_model"] = mode
            config["threads"] = threads
            path.write_text(json.dumps(config))
            result = subprocess.run(
                [BINARY, "sceptre", path], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output = (self.root / "result.tsv").read_bytes()
            row = next(csv.DictReader(output.decode().splitlines(), delimiter="\t"))
            self.assertEqual(row["status"], "ok")
            label = mode or "sceptre"
            if label in outputs:
                self.assertEqual(output, outputs[label])
            outputs[label] = output
            metadata = json.loads((self.root / "result.run.json").read_text())
            self.assertEqual(metadata["null_model"], label)
            self.assertEqual(
                metadata["null_fit_method"],
                (
                    "poisson_coefficients_conditional_nb_ml"
                    if label == "sceptre"
                    else "direct_glm"
                ),
            )
            self.assertEqual(metadata["effect_fit_method"], "direct_glm")
            # SCEPTRE's staging by default: an empirical first stage screened at 0.02.
            self.assertEqual(metadata["resampling_policy"], "empirical_screen")
            self.assertEqual(metadata["screen"], 0.02)
            self.assertEqual(
                metadata["score_numerator"],
                "sceptre_uncorrected" if label == "sceptre" else "nuisance_projected",
            )
        path.write_text(json.dumps(dict(config, screen=0)))
        result = subprocess.run([BINARY, "sceptre", path], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.root / "result.run.json").read_text())
        self.assertEqual((metadata["resampling_policy"], metadata["screen"]), ("rapidglm", 0))
        result, _ = self.execute("fit", screen=0.02)
        self.assertEqual(result.returncode, 2)
        self.assertIn("screen is a scan or sceptre setting", result.stderr)
        for override in [
            dict(null_model="typo"),
            dict(family="poisson"),
            dict(dispersion="fixed"),
            dict(ridge=0.1),
            dict(screen=1.5),
        ]:
            path.write_text(json.dumps(dict(config, **override)))
            result = subprocess.run(
                [BINARY, "sceptre", path], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_tiled_sceptre_matches_a_single_run(self):
        cells, genes, targets = 120, 5, 4
        counts = [
            (g + 1, c + 1, (c * (g + 3) * 7919) % (g + 5) // 2 + (3 if g == 2 and c % 4 == 1 else 0))
            for g in range(genes)
            for c in range(cells)
        ]
        counts = [t for t in counts if t[2] > 0]
        calls = [(c + 1, c % 4 + 1) for c in range(cells) if (c // 4) % 3 != 0]
        base = dict(
            responses=self.file(
                "tiles.mtx",
                f"%%MatrixMarket matrix coordinate integer general\n{genes} {cells} {len(counts)}\n"
                + "".join(f"{g} {c} {v}\n" for g, c, v in counts),
            ),
            design=self.file(
                "tiles_design.tsv", "".join(f"1\t{(c % 7) / 7}\n" for c in range(cells))
            ),
            assignments=self.file(
                "tiles_guides.mtx",
                f"%%MatrixMarket matrix coordinate integer general\n{targets} {cells} {len(calls)}\n"
                + "".join(f"{t} {c} 1\n" for c, t in sorted(calls, key=lambda x: (x[1], x[0]))),
            ),
            guides=self.file(
                "tiles_guides.tsv",
                "id\ttarget\tnon_targeting\n"
                + "".join(f"g{t}\tT{t}\t0\n" for t in range(targets)),
            ),
            pairs=self.file(
                "tiles_pairs.tsv",
                "response\ttarget\tkind\n"
                + "".join(f"{g}\tT{t}\tdiscovery\n" for g in range(genes) for t in range(targets)),
            ),
            family="nb",
            moi="high",
            cell_qc=False,
            min_nonzero=1,
            budgets=[99, 499],
            screen=0.3,
            alpha=0.5,
            threads=4,
            seed=11,
        )
        path = self.root / "tiles.json"
        # Inputs older than a few seconds, as real ones are (fresh files are not
        # entered in a checkpoint's fingerprint record).
        old = time.time() - 100
        for key in ("responses", "design", "assignments", "guides", "pairs"):
            os.utime(base[key], (old, old))

        def run(**kwargs):
            path.write_text(json.dumps(dict(base, **kwargs)))
            return subprocess.run([BINARY, "sceptre", path], capture_output=True, text=True)

        modes = {"default": {}, "reuse": {"reuse_draws": True},
                 "screen": {"shared_screen": True, "shared_screen_p": 0.5},
                 "sequential": {"sequential_screen": True},
                 "both": {"shared_screen": True, "shared_screen_p": 0.5, "sequential_screen": True},
                 "early": {"sequential_screen": True, "early_stop_p": 0.1}}
        for name, mode in modes.items():
            single = run(**mode, output=str(self.root / f"single_{name}"))
            self.assertEqual(single.returncode, 0, single.stderr)
            expected = (self.root / f"single_{name}.tsv").read_bytes()
            rows = list(csv.DictReader(expected.decode().splitlines(), delimiter="\t"))
            self.assertGreater(sum(r["status"] == "ok" for r in rows), 10)
            record = json.loads((self.root / f"single_{name}.run.json").read_text())
            self.assertEqual(record["shared_screen"], mode.get("shared_screen", False))
            self.assertEqual(record["sequential_screen"], mode.get("sequential_screen", False))
            self.assertEqual(record["early_stop_p"], mode.get("early_stop_p", 0))
            # One process with checkpoints, and separate prepare/tile/merge phases.
            tiled = dict(
                mode, tile_targets=2, tile_responses=2,
                checkpoint_dir=str(self.root / f"ck_{name}"),
                output=str(self.root / f"tiled_{name}"),
            )
            result = run(**tiled)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.root / f"tiled_{name}.tsv").read_bytes(), expected)
            record = json.loads((self.root / f"tiled_{name}.run.json").read_text())
            self.assertEqual(record["tiles"]["count"], 6)
            phases = dict(tiled, checkpoint_dir=str(self.root / f"ph_{name}"),
                          output=str(self.root / f"phases_{name}"))
            self.assertEqual(run(**phases, phase="prepare").returncode, 0)
            for k in range(6):
                result = run(**dict(phases, threads=1), phase="tile", tile=k)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = run(**phases, phase="merge")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((self.root / f"phases_{name}.tsv").read_bytes(), expected)
        # A tiled run keeps its inputs' fingerprints with their sizes and times, and
        # later phases take them from there instead of hashing the files again.
        cache = pathlib.Path(phases["checkpoint_dir"]) / "inputs.tsv"
        lines = cache.read_text().splitlines()
        self.assertEqual(lines[0], "rapidglm-inputs-1")
        self.assertTrue(any(line.endswith(base["responses"]) for line in lines[1:]))
        merged = lambda: run(**phases, phase="merge")
        expected = (self.root / f"phases_{name}.tsv").read_bytes()
        record = json.loads((self.root / f"phases_{name}.run.json").read_text())
        self.assertEqual(merged().returncode, 0)
        self.assertEqual((self.root / f"phases_{name}.tsv").read_bytes(), expected)
        again = json.loads((self.root / f"phases_{name}.run.json").read_text())
        self.assertEqual(again["input_sha1"], record["input_sha1"])
        # The recorded fingerprint is what is used: a wrong one makes the plan differ.
        cache.write_text("\n".join(
            ("0" * 40 + line[40:]) if line.endswith(base["responses"]) else line
            for line in lines) + "\n")
        result = merged()
        self.assertEqual(result.returncode, 2)
        self.assertIn("holds a run with other inputs", result.stderr)
        cache.unlink()
        self.assertEqual(merged().returncode, 0)
        self.assertEqual(cache.read_text().splitlines(), lines)
        # A file that changed is hashed again: other content is refused, the same
        # content with a new modification time is accepted.
        pairs_path = pathlib.Path(base["pairs"])
        original = pairs_path.read_text()
        pairs_path.write_text(original.replace("0\tT0\tdiscovery\n", "", 1))
        result = merged()
        self.assertEqual(result.returncode, 2)
        self.assertIn("holds a run with other inputs", result.stderr)
        pairs_path.write_text(original)
        self.assertEqual(merged().returncode, 0)
        self.assertEqual((self.root / f"phases_{name}.tsv").read_bytes(), expected)
        # The screens end some pairs early and keep the default's records otherwise.
        read = lambda name: list(csv.DictReader(
            (self.root / f"single_{name}.tsv").read_text().splitlines(), delimiter="\t"))
        for name, threshold in (("screen", 0.5), ("sequential", 0.3), ("both", 0.3)):
            stopped = 0
            for a, b in zip(read("default"), read(name)):
                if a["status"] != "ok":
                    continue
                if (a["p_value"], a["draws"]) != (b["p_value"], b["draws"]):
                    self.assertLessEqual(int(b["draws"]), 99)
                    self.assertGreater(float(b["p_value"]), threshold)
                    stopped += 1
            self.assertGreater(stopped, 0, name)
        # The early stop ends pairs past the screen that are above its threshold,
        # and keeps the other records.
        stopped = kept = 0
        for a, b in zip(read("sequential"), read("early")):
            if a["status"] != "ok":
                continue
            if (a["p_value"], a["draws"]) != (b["p_value"], b["draws"]):
                self.assertEqual(int(a["draws"]), 499)
                self.assertGreater(int(b["draws"]), 99)
                self.assertGreater(float(b["p_value"]), 0.1)
                stopped += 1
            else:
                kept += int(a["draws"]) == 499
        self.assertGreater(stopped, 0)
        self.assertGreater(kept, 0)
        # crt_draw accepts the hazard sampler and records it.
        result = run(crt_draw="hazard", output=str(self.root / "hazard"))
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads((self.root / "hazard.run.json").read_text())
        self.assertEqual(record["crt_draw"], "hazard")
        self.assertIn(record["crt_draw_used"], ("hazard", "thinned"))
        self.assertEqual("no AES instructions" in result.stderr,
                         record["crt_draw_used"] == "thinned")
        result = run(crt_draw="other", output=str(self.root / "bad_crt_draw"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("crt_draw must be thinned/direct/hazard", result.stderr)
        # The skew-normal fit check: strict by default, recorded, and part of a tiled plan.
        record = json.loads((self.root / "single_default.run.json").read_text())
        self.assertEqual(record["fit_check"], "strict")
        result = run(fit_check="ks", output=str(self.root / "ks"))
        self.assertEqual(result.returncode, 0, result.stderr)
        ks_record = json.loads((self.root / "ks.run.json").read_text())
        self.assertEqual(ks_record["fit_check"], "ks")
        self.assertEqual(ks_record["confirm_draws"], 0)
        # Confirmation draws: by default ten times the last budget (here 499), recorded;
        # 0 turns them off; they need the strict check and must exceed the budgets.
        self.assertEqual(record["confirm_draws"], 4999)
        result = run(confirm_draws=0, output=str(self.root / "no_confirm"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "no_confirm.run.json").read_text())["confirm_draws"], 0)
        result = run(confirm_draws=999, output=str(self.root / "confirm"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((self.root / "confirm.run.json").read_text())["confirm_draws"], 999)
        result = run(confirm_draws=999, fit_check="ks", output=str(self.root / "bad_confirm"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("confirm_draws needs", result.stderr)
        result = run(confirm_draws=499, output=str(self.root / "bad_confirm"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must exceed the last budget", result.stderr)
        result = run(fit_check="loose", output=str(self.root / "bad_fit_check"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("fit_check must be strict/ks", result.stderr)
        result = run(fit_check="ks", phase="merge", checkpoint_dir=str(self.root / "ck_default"),
                     tile_targets=2, tile_responses=2, output=str(self.root / "other_check"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("holds a run with other inputs", result.stderr)
        result = run(early_stop_p=1, output=str(self.root / "bad_early_stop"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("early_stop_p must be in [0, 1)", result.stderr)
        # Every target keeps its sampler by default. With a budget too small for
        # them the run says so, gives the same output, and a tiled run without
        # tile sizes splits the targets into blocks that fit (here: single
        # targets), which later phases take from the plan whatever their budget.
        default_output = (self.root / "single_default.tsv").read_bytes()
        samplers = json.loads((self.root / "single_default.run.json").read_text())["samplers"]
        self.assertEqual(samplers["cached_targets"], targets)
        self.assertEqual(samplers["targets_rebuilt_per_pair"], 0)
        result = run(sampler_cache_mb=0, output=str(self.root / "nocache"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rebuilt for every pair", result.stderr)
        self.assertEqual((self.root / "nocache.tsv").read_bytes(), default_output)
        samplers = json.loads((self.root / "nocache.run.json").read_text())["samplers"]
        self.assertEqual(samplers["targets_rebuilt_per_pair"], targets)
        blocks = dict(checkpoint_dir=str(self.root / "ck_blocks"), output=str(self.root / "blocks"))
        result = run(sampler_cache_mb=0, phase="prepare", **blocks)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = (self.root / "ck_blocks" / "plan.txt").read_text()
        self.assertIn("tile_targets 1\n", plan)
        self.assertIn(f"tiles {targets}\n", plan)
        for k in range(targets):
            self.assertEqual(run(phase="tile", tile=k, **blocks).returncode, 0)
        result = run(phase="merge", **blocks)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "blocks.tsv").read_bytes(), default_output)
        record = json.loads((self.root / "blocks.run.json").read_text())
        self.assertEqual(record["tiles"]["tile_targets"], 1)
        # Counts are read into the compact count matrix; other responses are not.
        record = json.loads((self.root / "single_default.run.json").read_text())
        self.assertTrue(record["count_matrix"])
        real = self.file(
            "tiles_real.mtx",
            f"%%MatrixMarket matrix coordinate real general\n{genes} {cells} {len(counts)}\n"
            + "".join(f"{g} {c} {v + 0.5}\n" for g, c, v in counts),
        )
        result = run(responses=real, family="gaussian", null_model="direct",
                     output=str(self.root / "real"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads((self.root / "real.run.json").read_text())["count_matrix"])
        result = run(shared_screen_p=0.1, output=str(self.root / "bad"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shared_screen_p needs shared_screen", result.stderr)
        # Another run's settings are refused, and the keys need checkpoint_dir.
        result = run(**dict(tiled, seed=12))
        self.assertEqual(result.returncode, 2)
        self.assertIn("holds a run with other inputs", result.stderr)
        result = run(phase="merge", output=str(self.root / "x"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("phase needs checkpoint_dir", result.stderr)

    def test_null_model_is_not_silently_ignored_by_fit(self):
        result, _ = self.execute("fit", null_model="sceptre")
        self.assertEqual(result.returncode, 2)
        self.assertIn("null_model is a scan or sceptre setting", result.stderr)

    def test_regression_test_defaults_are_recorded(self):
        _, config = self.execute("fit")
        pathlib.Path(config["design"]).write_text("1\t0\n" * 3 + "1\t1\n" * 3)
        # A nonzero group effect, so LR and Wald statistics differ.
        pathlib.Path(config["responses"]).write_text(
            "%%MatrixMarket matrix coordinate integer general\n1 6 6\n"
            "1 1 1\n1 2 2\n1 3 3\n1 4 4\n1 5 6\n1 6 9\n"
        )
        path = self.root / "config.json"
        statistics = {}
        for command, test, expected in [
            ("pseudobulk-de", None, "lr"),
            ("pseudobulk-de", "lr", "lr"),
            ("pseudobulk-de", "wald", "wald"),
            ("dose-trend", None, "wald"),
        ]:
            run = dict(config, coefficient=1)
            if test:
                run["test"] = test
            path.write_text(json.dumps(run))
            r = subprocess.run([BINARY, command, path], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.loads((self.root / "result.run.json").read_text())
            self.assertEqual(record["test"], expected)
            row = next(
                csv.DictReader(
                    (self.root / "result.tsv").read_text().splitlines(), delimiter="\t"
                )
            )
            statistics[(command, test)] = row["statistic"]
        self.assertEqual(
            statistics[("pseudobulk-de", None)], statistics[("pseudobulk-de", "lr")]
        )
        self.assertNotEqual(
            statistics[("pseudobulk-de", "lr")], statistics[("pseudobulk-de", "wald")]
        )

    def nb_panel(self):
        # 12 features x 8 samples (feature-by-observation), deterministic counts.
        import random

        rng = random.Random(5)
        entries = []
        for j in range(12):
            for i in range(8):
                mean = (5 + 20 * j) * (1.5 if i % 2 else 1)
                v = sum(rng.random() < mean / 40 for _ in range(80)) + rng.randrange(3)
                if v:
                    entries.append(f"{j + 1} {i + 1} {v}")
        return dict(
            responses=self.file(
                "nb.mtx",
                "%%MatrixMarket matrix coordinate integer general\n"
                f"12 8 {len(entries)}\n" + "\n".join(entries) + "\n",
            ),
            design=self.file("nb_design.tsv", "1\t0\n1\t1\n" * 4),
            family="nb",
            coefficient=1,
        )

    def test_quasi_likelihood_is_the_nb_pseudobulk_default(self):
        tables = []
        for threads in (1, 3):
            r, _ = self.execute("pseudobulk-de", threads=threads, **self.nb_panel())
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.loads((self.root / "result.run.json").read_text())
            self.assertEqual(record["test"], "ql")
            self.assertEqual(record["ql_dispersion"], "cox_reid_adjusted_ml")
            self.assertEqual(record["ql_dispersion_features"], 12)
            self.assertFalse(record["ql_abundance_trend"])
            self.assertGreater(record["ql_df_prior"], 0)
            tables.append((self.root / "result.tsv").read_text())
        self.assertEqual(tables[0], tables[1])
        rows = list(csv.DictReader(tables[0].splitlines(), delimiter="\t"))
        self.assertEqual(len(rows), 12)
        for column in ("theta_feature", "ql_dispersion", "ql_prior", "ql_posterior"):
            self.assertIn(column, rows[0])
        self.assertEqual(rows[0]["df_residual"], "6")
        self.assertTrue(all(0 <= float(row["p_value"]) <= 1 for row in rows))

    def test_quasi_likelihood_and_cox_reid_misuse_is_rejected(self):
        panel = self.nb_panel()
        for extra, message in [
            (dict(test="ql", dispersion="shared"), "leave dispersion at ml"),
            (dict(test="ql", cox_reid=True), "omit cox_reid"),
            (dict(test="ql", family="poisson"), "requires unweighted, unpenalized NB"),
        ]:
            r, _ = self.execute("pseudobulk-de", **{**panel, **extra})
            self.assertEqual(r.returncode, 2, extra)
            self.assertIn(message, r.stderr)
        r, _ = self.execute("fit", **{**panel, "dispersion": "fixed", "cox_reid": True})
        self.assertEqual(r.returncode, 2)
        self.assertIn("Cox-Reid adjustment applies to NB fits with ML dispersion", r.stderr)
        r, _ = self.execute("fit", **{**panel, "cox_reid": True})
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_scan_pairs_all_by_all_and_records(self):
        panel = self.nb_panel()
        predictors = self.file(
            "predictors.mtx",
            "%%MatrixMarket matrix coordinate real general\n2 8 12\n"
            + "".join(f"1 {i + 1} {i % 2}\n" for i in range(8) if i % 2)
            + "".join(f"2 {i + 1} {0.5 * i - 1.75}\n" for i in range(8)),
        )
        r, _ = self.execute("scan", predictors=predictors, **panel)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = list(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
        self.assertEqual(len(rows), 12 * 2)
        self.assertEqual(rows[0]["family"], "all")
        record = json.loads((self.root / "result.run.json").read_text())
        self.assertEqual(
            (record["null_model"], record["calibration"], record["effect_fits"]),
            ("direct", "permutation", "significant"),
        )
        pairs = self.file("pairs.tsv", "response\tpredictor\tfamily\n0\t0\ta\n3\t1\tb\n")
        r, _ = self.execute(
            "scan", predictors=predictors, pairs=pairs, calibration="asymptotic", **panel
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = list(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
        self.assertEqual([(x["response"], x["predictor"], x["family"]) for x in rows],
                         [("0", "0", "a"), ("3", "1", "b")])
        self.assertTrue(all(x["draws"] == "0" for x in rows))

    def test_scan_misuse_is_rejected(self):
        panel = self.nb_panel()
        continuous = self.file(
            "cont.mtx", "%%MatrixMarket matrix coordinate real general\n1 8 2\n1 1 0.5\n1 2 2\n"
        )
        for extra, message in [
            (dict(calibration="bogus"), "calibration must be"),
            (dict(calibration="crt"), "CRT calibration requires binary predictors"),
            (dict(calibration="asymptotic", null_model="sceptre"), "asymptotic calibration needs"),
        ]:
            r, _ = self.execute("scan", predictors=continuous, **{**panel, **extra})
            self.assertEqual(r.returncode, 2, extra)
            self.assertIn(message, r.stderr)
        r, _ = self.execute("fit", predictors=continuous, **panel)
        self.assertEqual(r.returncode, 2)
        self.assertIn("are scan settings", r.stderr)

    def test_bulk_de_tables_records_and_thread_stability(self):
        panel = self.nb_panel()
        tables = []
        for threads in (1, 3):
            r, _ = self.execute("bulk-de", threads=threads, **panel)
            self.assertEqual(r.returncode, 0, r.stderr)
            tables.append((self.root / "result.tsv").read_text())
        self.assertEqual(tables[0], tables[1])
        rows = list(csv.DictReader(tables[0].splitlines(), delimiter="\t"))
        self.assertEqual(len(rows), 12)
        for column in ("baseMean", "log2FoldChange", "pvalue", "padj", "dispGeneEst", "dispFit",
                       "dispMAP", "dispersion", "maxCooks", "replaced"):
            self.assertIn(column, rows[0])
        record = json.loads((self.root / "result.run.json").read_text())
        self.assertEqual(record["test"], "wald")
        self.assertEqual(len(record["size_factors"]), 8)
        self.assertGreaterEqual(record["prior_variance"], 0.25)
        self.assertIn(record["trend"], ("parametric", "local", "mean"))
        filter_rows = (self.root / "result.filter.tsv").read_text().splitlines()
        self.assertEqual(filter_rows[0], "theta\trejections\tfit")
        self.assertEqual(len(filter_rows), 51)
        # Supplied size factors are used as given; the LRT runs on request.
        values = [0.8, 1.25] * 4
        r, _ = self.execute("bulk-de", test="lrt", independent_filtering=False,
                            size_factor_values=self.file("sf.txt", "\n".join(map(str, values))), **panel)
        self.assertEqual(r.returncode, 0, r.stderr)
        record = json.loads((self.root / "result.run.json").read_text())
        self.assertEqual(record["test"], "lrt")
        self.assertEqual(record["size_factors"], values)
        self.assertIsNone(record["filter_threshold"])
        self.assertEqual(record["lfc_prior_variance"], [])
        self.assertNotIn("log2FoldChangeShrunk", (self.root / "result.tsv").read_text().splitlines()[0])
        # Normal-prior shrinkage adds the posterior mode and SD; tests are unchanged.
        r, _ = self.execute("bulk-de", legacy_cooks=True, **panel)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((self.root / "result.run.json").read_text())["cooks_dispersion"],
                         "legacy_pydeseq2")
        r, _ = self.execute("pseudobulk-de", legacy_cooks=True, **panel)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("legacy_cooks are bulk-de settings", r.stderr)
        r, _ = self.execute("bulk-de", lfc_shrink="normal", **panel)
        self.assertEqual(r.returncode, 0, r.stderr)
        shrunk = list(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
        self.assertEqual([x["pvalue"] for x in shrunk], [x["pvalue"] for x in rows])
        self.assertTrue(all(abs(float(x["log2FoldChangeShrunk"])) <= abs(float(y["log2FoldChange"])) + 1e-9
                            for x, y in zip(shrunk, rows)))
        prior = json.loads((self.root / "result.run.json").read_text())["lfc_prior_variance"]
        self.assertEqual(len(prior), 2)
        self.assertEqual(prior[0], 1e6)

    def test_bulk_de_misuse_is_rejected(self):
        panel = self.nb_panel()
        for extra, message in [
            (dict(family="poisson"), "bulk-de fits NB models"),
            (dict(dispersion="shared"), "bulk-de fits NB models"),
            (dict(test="ql"), "bulk-de test must be wald/lrt"),
            (dict(size_factors="tmm"), "size_factors must be ratio/poscounts"),
            (dict(trend="loess"), "trend must be parametric/local/mean"),
            (dict(lfc_shrink="apeglm"), "lfc_shrink must be none/normal"),
        ]:
            r, _ = self.execute("bulk-de", **{**panel, **extra})
            self.assertEqual(r.returncode, 2, extra)
            self.assertIn(message, r.stderr)
        r, _ = self.execute("pseudobulk-de", **{**panel, "cooks_filter": False})
        self.assertEqual(r.returncode, 2)
        self.assertIn("are bulk-de settings", r.stderr)

    def test_contrasts_and_joint_tests(self):
        # Design (1, a, b) for three groups; (1, a, a + b) makes beta_a - beta_b column 1.
        import random

        rng = random.Random(9)
        entries = []
        for j in range(10):
            for i in range(12):
                mean = (10 + 5 * j) * (2 if 4 <= i < 8 and j % 2 else 1)
                v = sum(rng.random() < mean / 60 for _ in range(120)) + rng.randrange(3)
                if v:
                    entries.append(f"{j + 1} {i + 1} {v}")
        responses = self.file("groups.mtx", "%%MatrixMarket matrix coordinate integer general\n"
                              f"10 12 {len(entries)}\n" + "\n".join(entries) + "\n")
        rows = [(1, int(4 <= i < 8), int(i >= 8)) for i in range(12)]
        design = self.file("x.tsv", "".join(f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in rows))
        reparametrized = self.file("xr.tsv", "".join(f"{r[0]}\t{r[1]}\t{r[1] + r[2]}\n" for r in rows))
        for command, test in (("pseudobulk-de", "ql"), ("pseudobulk-de", "lr"), ("bulk-de", "wald")):
            base = dict(responses=responses, family="nb", test=test)
            r, _ = self.execute(command, design=design, contrast=[0, 1, -1], **base)
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.loads((self.root / "result.run.json").read_text())
            self.assertEqual(record["hypothesis"], [[0, 1, -1]])
            a = list(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
            r, _ = self.execute(command, design=reparametrized, coefficient=1, **base)
            self.assertEqual(r.returncode, 0, r.stderr)
            b = list(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
            key = "pvalue" if command == "bulk-de" else "p_value"
            for x, y in zip(a, b):
                self.assertAlmostEqual(float(x[key]), float(y[key]), places=6)
        # Joint tests: LR/QL/LRT only, with no single fold change.
        for command, test in (("pseudobulk-de", "ql"), ("bulk-de", "lrt")):
            r, _ = self.execute(command, design=design, coefficients=[1, 2], responses=responses,
                                family="nb", test=test)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads((self.root / "result.run.json").read_text())["hypothesis"],
                             [[0, 1, 0], [0, 0, 1]])
            row = next(csv.DictReader((self.root / "result.tsv").read_text().splitlines(), delimiter="\t"))
            self.assertEqual(row["log2FoldChange" if command == "bulk-de" else "log2fc"], "NA")
        for extra, message in [
            (dict(coefficients=[1, 2], test="wald"), "the Wald test takes one coefficient or contrast"),
            (dict(coefficient=1, contrast=[0, 1, -1]), "give at most one of coefficient"),
            (dict(contrast=[0, 1]), "contrast needs one weight per design column"),
            (dict(coefficients=[1, 1], test="lr"), "linearly independent"),
            (dict(coefficients=[0, 1, 2], test="lr"), "between 1 and p - 1 rows"),
        ]:
            r, _ = self.execute("pseudobulk-de", design=design, responses=responses, family="nb", **extra)
            self.assertEqual(r.returncode, 2, extra)
            self.assertIn(message, r.stderr)
        r, _ = self.execute("fit", design=design, responses=responses, family="nb", contrast=[0, 1, -1])
        self.assertEqual(r.returncode, 2)
        self.assertIn("are bulk-de, pseudobulk-de and dose-trend settings", r.stderr)

    def test_unsupported_dispersion_modes_are_rejected(self):
        extra = dict(
            assignments=self.file(
                "guides.mtx",
                "%%MatrixMarket matrix coordinate integer general\n1 6 3\n1 1 1\n1 3 1\n1 5 1\n",
            ),
            guides=self.file("guides.tsv", "id\ttarget\tnon_targeting\ng1\tt1\t0\n"),
            pairs=self.file("pairs.tsv", "response\ttarget\tkind\n0\tt1\tdiscovery\n"),
            group=self.file("group.txt", "0\n1\n0\n1\n0\n1\n"),
        )
        for command in ("sceptre", "da-lr"):
            for mode in ("shared", "trend"):
                result, _ = self.execute(command, dispersion=mode, **extra)
                self.assertEqual(result.returncode, 2, (command, mode))
                self.assertIn("not supported by the scan, sceptre or da-lr", result.stderr)

    def test_input_fingerprints_across_sha1_block_boundaries(self):
        for size in (6 * 2, 55, 56, 63, 64, 65, 119, 120, 128, 1000):
            text = "1\n" * 6
            text += " " * (size - len(text))
            result, config = self.execute(
                "fit", weights=self.file(f"weights{size}.txt", text)
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((self.root / "result.run.json").read_text())
            self.assertEqual(len(record["input_sha1"]), 4)
            for path, digest in record["input_sha1"].items():
                self.assertEqual(
                    digest, hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()
                )


if __name__ == "__main__":
    unittest.main()
