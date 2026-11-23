"""Hardware-free integration test: synthetic LeRobot v3 -> training -> prediction."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import torch

from algorithm.flow_matching.dataset import StitchedSequenceDataset
from scripts.hardware import build_command

ROOT=Path(__file__).resolve().parents[1]
JOINTS=[f"{n}.pos" for n in ("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper")]


def run(*args):
    r=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==0,r.stdout+"\n"+r.stderr
    return r.stdout


def synthetic_dataset(root):
    (root/"meta/episodes/chunk-000").mkdir(parents=True)
    (root/"data/chunk-000").mkdir(parents=True)
    (root/"videos/observation.images.wrist/chunk-000").mkdir(parents=True)
    n=60
    states=np.sin(np.arange(n)[:,None]/9+np.arange(6)[None,:]).astype(np.float32)*10
    actions=states+np.float32(.1)
    df=pd.DataFrame(dict(index=np.arange(n),episode_index=np.repeat(np.arange(3),20),
        frame_index=np.tile(np.arange(20),3),timestamp=np.tile(np.arange(20)/30,3),
        action=list(actions),**{"observation.state":list(states)}))
    df.to_parquet(root/"data/chunk-000/file-000.parquet")
    rows=[]
    for i in range(3):
        rows.append(dict(episode_index=i,length=20,dataset_from_index=i*20,dataset_to_index=(i+1)*20,
            **{"videos/observation.images.wrist/chunk_index":0,
               "videos/observation.images.wrist/file_index":0,
               "videos/observation.images.wrist/from_timestamp":i*20/30}))
    pd.DataFrame(rows).to_parquet(root/"meta/episodes/chunk-000/file-000.parquet")
    info=dict(total_frames=n,fps=30,features={"action":{"names":JOINTS}},
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    (root/"meta/info.json").write_text(json.dumps(info))
    with av.open(str(root/"videos/observation.images.wrist/chunk-000/file-000.mp4"),"w") as c:
        stream=c.add_stream("mpeg4",rate=30)
        stream.width,stream.height,stream.pix_fmt=640,480,"yuv420p"
        for i in range(n):
            image=np.full((480,640,3),[190+i%50,30,10],dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format="rgb24")):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    return df


def test_conversion_training_and_deployment_offline(tmp_path):
    raw=tmp_path/"raw"
    df=synthetic_dataset(raw)
    prepared=tmp_path/"prepared"
    run("-m","runner.flow_matching.prepare","--dataset",raw,"--out",prepared,"--val-episodes",1)
    manifest=json.loads((prepared/"manifest.json").read_text())
    assert len(manifest["train_episodes"])==2 and len(manifest["val_episodes"])==1
    assert not set(manifest["train_episodes"])&set(manifest["val_episodes"])
    selected=df[df.episode_index.isin(manifest["train_episodes"])]
    with np.load(prepared/"normalization.npz") as stats,np.load(prepared/"train.npz") as z:
        assert np.array_equal(stats["action_min"],np.stack(selected.action).min(0))
        reconstructed=(z["actions"]+1)/2*(stats["action_max"]-stats["action_min"]+1e-6)+stats["action_min"]
        np.testing.assert_allclose(reconstructed,np.stack(selected.action),atol=3e-6)
        assert z["images"].dtype==np.uint8 and z["images"].shape==(40,3,96,96)
        assert z["images"][:,0].mean()>z["images"][:,2].mean()
    ds=StitchedSequenceDataset(str(prepared/"train.npz"),horizon_steps=16,device="cpu")
    assert len(ds)==10
    for start,_ in ds.indices:
        assert start//20==(start+15)//20
    output=tmp_path/"run"
    run("-m","runner.flow_matching.train","--data",prepared,"--output",output,
        "--epochs",2,"--batch-size",4,"--device","cpu")
    ckpt=torch.load(output/"checkpoint/best_val.pt",map_location="cpu",weights_only=True)
    assert ckpt["global_step"]>0 and np.isfinite(ckpt["best_val"])
    result=json.loads(run("-m","runner.flow_matching.predict","--run",output,
        "--data",prepared/"val.npz","--device","cpu"))
    assert result["shape"]==[16,6] and np.isfinite(result["actions"]).all()
    offline=json.loads(run("-m","deployment.deploy","--run",output,
        "--offline-data",prepared/"val.npz","--no-preview"))
    assert offline["mode"]=="offline" and offline["shape"]==[16,6]
    run("-m","runner.flow_matching.train","--data",prepared,"--output",tmp_path/"resume",
        "--resume",output/"checkpoint/last.pt","--epochs",1,"--batch-size",4,"--device","cpu")
    resumed=torch.load(tmp_path/"resume/checkpoint/last.pt",map_location="cpu",weights_only=True)
    assert resumed["epoch"]==3 and resumed["global_step"]>ckpt["global_step"]


def test_record_resume_appends_and_preserves_task_argument(tmp_path):
    existing=tmp_path/"session"
    (existing/"meta").mkdir(parents=True)
    (existing/"meta/info.json").write_text("{}")
    args=SimpleNamespace(command="record",episodes=10,resume=existing,name=None,
        dataset_root=tmp_path,task='Put the "red" object left.',no_display=True)
    hardware=dict(leader_port="FAKE_LEADER",follower_port="FAKE_FOLLOWER",motor_model="hx30hm")
    cmd=build_command(args,hardware)
    assert "--resume=true" in cmd and "--dataset.num_episodes=10" in cmd
    assert f"--dataset.root={existing}" in cmd
    assert '--dataset.single_task=Put the "red" object left.' in cmd
    assert "--robot.motor_model=hx30hm" in cmd
    assert "--display_data=false" in cmd


def test_deployment_default_is_read_only(monkeypatch):
    from unittest.mock import patch
    import deployment.deploy as deploy
    monkeypatch.setattr(sys,"argv",["deploy.py","--port","FAKE_PORT"])
    with patch.object(deploy,"Policy"),patch.object(deploy,"live") as live:
        deploy.main()
        args=live.call_args.args[0]
        assert not args.execute and args.preview and args.duration==30
