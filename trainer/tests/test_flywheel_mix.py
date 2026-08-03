from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import pytest
from lehome_train.flywheel.mix import build_mix_plan, verify_mix_plan
def _dataset(path:Path, *, frames:int, flywheel:bool=False, grade:str="A", holdout:bool=False)->Path:
    (path/"meta").mkdir(parents=True)
    (path/"manifest.json").write_text(json.dumps({"output_format":"groot_lerobot_v2.1_per_episode","frame_count":frames,"train_episode_ids":["0"],"source_revision":"a"*40}),encoding="utf-8")
    if flywheel: (path/"meta"/"materialization-provenance.json").write_text(json.dumps({"raw_manifest_verified":True,"raw_manifest_sha256":"b"*64,"quality_grade":grade,"raw_identity":{"release_stage":"public_unseen" if holdout else "seen"}}),encoding="utf-8")
    return path
def test_mix_targets_seventy_thirty_by_training_frames_and_freezes_cycles(tmp_path:Path)->None:
    plan=build_mix_plan(_dataset(tmp_path/"organizer",frames=700),_dataset(tmp_path/"new",frames=300,flywheel=True),seed=20260803)
    assert (plan.organizer_training_frames,plan.flywheel_training_frames)==(700,300)
    assert plan.source_weights=={"organizer":.7,"flywheel":.3}; assert len(plan.organizer_episode_ids)==700; assert len(plan.flywheel_episode_ids)==300
    assert plan.grade_weights=={"A":1.0,"B":.5}; verify_mix_plan(plan)
    with pytest.raises(ValueError,match="hash"): verify_mix_plan(replace(plan,sha256="0"*64))
@pytest.mark.parametrize(("grade","holdout","message"),[("A",True,"holdout"),("C",False,"grade")])
def test_mix_rejects_holdout_and_ineligible_grade(tmp_path:Path,grade:str,holdout:bool,message:str)->None:
    with pytest.raises(ValueError,match=message): build_mix_plan(_dataset(tmp_path/"org",frames=10),_dataset(tmp_path/"bad",frames=10,flywheel=True,grade=grade,holdout=holdout),seed=1)
